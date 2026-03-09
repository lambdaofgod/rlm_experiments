"""CustomizableRLM -- RLM subclass with parametrizable sub-LLM context budget.

Replaces the hardcoded 500K chars capacity assumption in the system prompt
with a runtime parameter derived from sub_lm_context_tokens or litellm metadata.
"""

from __future__ import annotations

import logging
import re
from typing import TYPE_CHECKING

import dspy
from dspy.predict.rlm import RLM

if TYPE_CHECKING:
    from dspy.signatures.signature import Signature

logger = logging.getLogger(__name__)

DEFAULT_CHAR_LIMIT = 500_000


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
        **kwargs,
    ) -> None:
        sub_lm = kwargs.get("sub_lm")
        self._sub_lm_context_chars = self.chars_for_sub_lm(
            sub_lm_context_tokens=sub_lm_context_tokens,
            lm=sub_lm,
            safety_factor=safety_factor,
        )
        self._small_model_tips = small_model_tips

        super().__init__(signature, **kwargs)

        # Patch the capacity figure in the action signature instructions
        self._patch_instructions()

    def _patch_instructions(self) -> None:
        """Replace the 500K char capacity and optionally inject tips."""
        old_instructions = self.generate_action.signature.instructions
        if old_instructions is None:
            return

        chars = self._sub_lm_context_chars

        # Replace the capacity figure: "~500K char capacity" -> "~Nk char capacity"
        patched = re.sub(
            r"~500K char capacity",
            f"~{chars // 1000}K char capacity",
            old_instructions,
        )

        # Inject tips if requested
        if self._small_model_tips:
            patched = self._inject_tips(patched, chars)

        self.generate_action.signature = self.generate_action.signature.with_instructions(patched)

    def _inject_tips(self, instructions: str, chars: int) -> str:
        """Append batching guidance and optional context-length warning."""
        tips = []

        # Context-length warning for very small models (<= 100k chars)
        if chars <= 100_000:
            tokens_k = chars // 4000
            tips.append(
                f"IMPORTANT: You have a total context window of approximately "
                f"~{tokens_k}k tokens. Be very careful about context length limits. "
                f"The sub-LLMs you can query also have this same ~{tokens_k}k token "
                f"limit, so you must be conservative with how much context you send "
                f"in each call. For example, a viable strategy is to feed 2-3 "
                f"documents per sub-LLM query. Analyze your input data and see if "
                f"it is sufficient to just fit it in a few sub-LLM calls!"
            )

        # Batching block for small-to-medium models (<= 200k chars)
        if chars <= 200_000:
            tips.append(
                f"IMPORTANT: Be very careful about using 'llm_query' as it incurs "
                f"high runtime costs. Always batch as much information as reasonably "
                f"possible into each call (aim for around ~{chars:,} characters per "
                f"call). For example, if you have 1000 lines of information to "
                f"process, it's much better to split into chunks of N and call "
                f"'llm_query' on each chunk rather than making 1000 individual calls. "
                f"Minimize the number of 'llm_query' calls by batching related "
                f"information together."
            )

        if tips:
            instructions = instructions + "\n\n" + "\n\n".join(tips)
        return instructions

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
