"""CustomizableRLM -- RLM subclass with parametrizable sub-LLM context budget.

Replaces the hardcoded 500K chars capacity assumption in the system prompt
with a runtime parameter derived from sub_lm_context_tokens or litellm metadata.

Also catches AdapterParseError in generate_action calls to avoid expensive
JSONAdapter fallback retries when ChatAdapter partially parses the response.
"""

from __future__ import annotations

import logging
import threading
from typing import Any, TYPE_CHECKING

import dspy
from dspy.adapters.chat_adapter import ChatAdapter
from dspy.adapters.utils import translate_field_type
from dspy.primitives.code_interpreter import CodeInterpreter, CodeInterpreterError
from dspy.primitives.prediction import Prediction
from dspy.primitives.repl_types import REPLHistory, REPLVariable
from dspy.predict.rlm import RLM, _strip_code_fences
from dspy.utils.exceptions import AdapterParseError

from prompts import build_action_instructions

if TYPE_CHECKING:
    from dspy.signatures.signature import Signature

logger = logging.getLogger(__name__)

DEFAULT_CHAR_LIMIT = 500_000


class REPLTimeoutError(Exception):
    """Raised when the code interpreter exceeds repl_timeout."""


def _force_registration_reset(repl) -> None:
    """Clear PythonInterpreter's registration cache so the next execute()
    re-registers tool wrappers and the typed SUBMIT against the live sandbox.

    Upstream only resets these on the BrokenPipeError path (pre-write crash).
    When the Deno subprocess dies mid-execution (empty-readline path), the
    flags stay stale and the auto-respawned Deno runs without llm_query or
    the typed SUBMIT wrapper -- collapsing back to PYTHON_SETUP_CODE's default
    single-arg SUBMIT(output). Resetting unconditionally is cheap (one extra
    JSON-RPC register per iteration against a live process) and covers every
    subprocess restart mode.
    """
    if hasattr(repl, "_tools_registered"):
        repl._tools_registered = False
    if hasattr(repl, "_mounted_files"):
        repl._mounted_files = False


def _format_execute_error(e: Exception, input_arg_names) -> str:
    """Render an execute() exception as a REPL observation string.

    A "No output from Deno subprocess" CodeInterpreterError means the sandbox
    process crashed mid-execution. We surface that to the model explicitly so
    it knows its in-REPL state is gone while input variables still exist --
    otherwise it cascades into phantom NameError/TypeError loops until the
    iteration budget is exhausted.
    """
    if isinstance(e, CodeInterpreterError) and "No output from Deno subprocess" in str(e):
        args_list = ", ".join(input_arg_names)
        args_desc = args_list if args_list else "(none)"
        return (
            f"[Error] Interpreter process crashed and was restarted. "
            f"Input variables ({args_desc}) are still available, but any "
            f"intermediate variables you defined in previous steps have been "
            f"lost. Re-compute them if needed."
        )
    return f"[Error] {e}"


class CustomizableRLM(RLM):
    """RLM subclass with a parametrizable sub-LLM context budget.

    The character limit advertised to the orchestrator for each llm_query()
    call is derived from sub_lm_context_tokens (authoritative) or sub_lm
    (best-effort litellm lookup), with sub_lm_context_tokens taking precedence
    when both are supplied.  All other RLM constructor arguments are forwarded
    via **kwargs.
    """

    def __init__(
        self,
        signature: type[Signature] | str,
        *,
        sub_lm_context_tokens: int | None = None,
        small_model_tips: bool = False,
        safety_factor: float = 0.8,
        repl_timeout: int | float = 300,
        **kwargs,
    ) -> None:
        sub_lm = kwargs.get("sub_lm")
        self._sub_lm_context_chars = self.chars_for_sub_lm(
            sub_lm_context_tokens=sub_lm_context_tokens,
            lm=sub_lm,
            safety_factor=safety_factor,
        )
        self._small_model_tips = small_model_tips
        self.repl_timeout = repl_timeout

        super().__init__(signature, **kwargs)

    def _build_signatures(self):
        """Override of upstream RLM._build_signatures.

        Identical to upstream except the action-instructions body is sourced
        from prompts.build_action_instructions so the capacity figure and
        optional small-model tips are wired in via .format() placeholders
        instead of post-construction string surgery.
        """
        inputs_str = ", ".join(f"`{n}`" for n in self.signature.input_fields)
        final_output_names = ", ".join(self.signature.output_fields.keys())
        output_fields = "\n".join(
            f"- {translate_field_type(n, f)}"
            for n, f in self.signature.output_fields.items()
        )
        task_instructions = (
            f"{self.signature.instructions}\n\n" if self.signature.instructions else ""
        )
        tool_docs = self._format_tool_docs(self._user_tools)

        action_body = build_action_instructions(
            inputs=inputs_str,
            output_fields=output_fields,
            final_output_names=final_output_names,
            max_llm_calls=self.max_llm_calls,
            sub_lm_context_chars=self._sub_lm_context_chars,
            small_model_tips=self._small_model_tips,
        )

        action_sig = (
            dspy.Signature({}, task_instructions + action_body + tool_docs)
            .append("variables_info", dspy.InputField(desc="Metadata about the variables available in the REPL"), type_=str)
            .append("repl_history", dspy.InputField(desc="Previous REPL code executions and their outputs"), type_=REPLHistory)
            .append("iteration", dspy.InputField(desc="Current iteration number (1-indexed) out of max_iterations"), type_=str)
            .append("reasoning", dspy.OutputField(desc="Think step-by-step: what do you know? What remains? Plan your next action."), type_=str)
            .append("code", dspy.OutputField(desc="Python code to execute. Output it directly after the `[[ ## code ## ]]` marker; markdown code fences (```python ... ```) are optional and will be stripped."), type_=str)
        )

        extract_instructions = """Based on the REPL trajectory, extract the final outputs now.

            Review your trajectory to see what information you gathered and what values you computed, then provide the final outputs."""

        extended_task_instructions = ""
        if task_instructions:
            extended_task_instructions = "The trajectory was generated with the following objective: \n" + task_instructions + "\n"
        full_extract_instructions = extended_task_instructions + extract_instructions

        extract_sig = dspy.Signature(
            {**self.signature.output_fields},
            full_extract_instructions,
        )
        extract_sig = extract_sig.prepend("repl_history", dspy.InputField(desc="Your REPL interactions so far"), type_=REPLHistory)
        extract_sig = extract_sig.prepend("variables_info", dspy.InputField(desc="Metadata about the variables available in the REPL"), type_=str)

        return action_sig, extract_sig

    def _generate_action_no_fallback(self, **kwargs) -> Prediction:
        """Call generate_action with JSONAdapter fallback disabled.

        If ChatAdapter partially parses the response (e.g. reasoning but no
        code), builds a stub Prediction with code="" so the iteration can
        continue without an expensive JSONAdapter retry.
        """
        no_fallback_adapter = ChatAdapter(use_json_adapter_fallback=False)
        try:
            with dspy.settings.context(adapter=no_fallback_adapter):
                return self.generate_action(**kwargs)
        except AdapterParseError as e:
            reasoning = ""
            if e.parsed_result and isinstance(e.parsed_result, dict):
                reasoning = e.parsed_result.get("reasoning", "")
            logger.warning(
                "ChatAdapter parsed reasoning but missed code field; "
                "returning empty code instead of falling back to JSONAdapter"
            )
            return Prediction(reasoning=reasoning, code="")

    async def _agenerate_action_no_fallback(self, **kwargs) -> Prediction:
        """Async version of _generate_action_no_fallback."""
        no_fallback_adapter = ChatAdapter(use_json_adapter_fallback=False)
        try:
            with dspy.settings.context(adapter=no_fallback_adapter):
                return await self.generate_action.acall(**kwargs)
        except AdapterParseError as e:
            reasoning = ""
            if e.parsed_result and isinstance(e.parsed_result, dict):
                reasoning = e.parsed_result.get("reasoning", "")
            logger.warning(
                "ChatAdapter parsed reasoning but missed code field; "
                "returning empty code instead of falling back to JSONAdapter"
            )
            return Prediction(reasoning=reasoning, code="")

    def _execute_iteration(
        self,
        repl: CodeInterpreter,
        variables: list[REPLVariable],
        history: REPLHistory,
        iteration: int,
        input_args: dict[str, Any],
        output_field_names: list[str],
    ) -> Prediction | REPLHistory:
        """Execute one iteration, catching AdapterParseError instead of falling back."""
        variables_info = [variable.format() for variable in variables]
        action = self._generate_action_no_fallback(
            variables_info=variables_info,
            repl_history=history,
            iteration=f"{iteration + 1}/{self.max_iterations}",
        )
        if self.verbose:
            logger.info(
                f"RLM iteration {iteration + 1}/{self.max_iterations}\n"
                f"Reasoning: {action.reasoning}\nCode:\n{action.code}"
            )

        timed_out = False

        def _kill_deno():
            nonlocal timed_out
            timed_out = True
            repl.deno_process.kill()

        try:
            code = _strip_code_fences(action.code)
            _force_registration_reset(repl)
            timer = threading.Timer(self.repl_timeout, _kill_deno)
            timer.start()
            try:
                result = repl.execute(code, variables=dict(input_args))
            finally:
                timer.cancel()
        except (CodeInterpreterError, SyntaxError) as e:
            if timed_out:
                raise REPLTimeoutError(
                    f"Code interpreter timed out after {self.repl_timeout}s"
                ) from e
            result = _format_execute_error(e, input_args.keys())

        return self._process_execution_result(action, result, history, output_field_names)

    async def _aexecute_iteration(
        self,
        repl: CodeInterpreter,
        variables: list[REPLVariable],
        history: REPLHistory,
        iteration: int,
        input_args: dict[str, Any],
        output_field_names: list[str],
    ) -> Prediction | REPLHistory:
        """Async version: execute one iteration, catching AdapterParseError."""
        variables_info = [variable.format() for variable in variables]
        pred = await self._agenerate_action_no_fallback(
            variables_info=variables_info,
            repl_history=history,
            iteration=f"{iteration + 1}/{self.max_iterations}",
        )
        if self.verbose:
            logger.info(
                f"RLM iteration {iteration + 1}/{self.max_iterations}\n"
                f"Reasoning: {pred.reasoning}\nCode:\n{pred.code}"
            )

        timed_out = False

        def _kill_deno():
            nonlocal timed_out
            timed_out = True
            repl.deno_process.kill()

        try:
            code = _strip_code_fences(pred.code)
            _force_registration_reset(repl)
            timer = threading.Timer(self.repl_timeout, _kill_deno)
            timer.start()
            try:
                result = repl.execute(code, variables=dict(input_args))
            finally:
                timer.cancel()
        except (CodeInterpreterError, SyntaxError) as e:
            if timed_out:
                raise REPLTimeoutError(
                    f"Code interpreter timed out after {self.repl_timeout}s"
                ) from e
            result = _format_execute_error(e, input_args.keys())

        return self._process_execution_result(pred, result, history, output_field_names)

    @staticmethod
    def chars_for_sub_lm(
        sub_lm_context_tokens: int | None = None,
        lm: dspy.LM | None = None,
        safety_factor: float = 0.8,
    ) -> int:
        """Derive the char limit from a token count, an LM object, or both.

        If both are provided, sub_lm_context_tokens takes precedence.
        If neither is provided, returns the upstream default of 500,000.

        Args:
            sub_lm_context_tokens: Context window size in tokens (authoritative).
            lm: A dspy.LM instance; litellm metadata will be queried.
            safety_factor: Fraction of theoretical capacity to use (default 0.8).

        Returns:
            Character limit as an integer.
        """
        tokens = sub_lm_context_tokens

        if tokens is None and lm is not None:
            tokens = _get_context_tokens_from_lm(lm)

        if tokens is None:
            return DEFAULT_CHAR_LIMIT

        return int(tokens * 4 * safety_factor)


def _get_context_tokens_from_lm(lm: dspy.LM) -> int | None:
    """Best-effort context window lookup via litellm."""
    try:
        import litellm
        info = litellm.get_model_info(lm.model)
        max_input = info.get("max_input_tokens")
        if isinstance(max_input, int) and max_input > 0:
            return max_input
    except Exception as e:
        logger.warning("Could not look up context window for %s: %s", lm.model, e)
    return None
