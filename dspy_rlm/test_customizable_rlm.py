"""Tests for CustomizableRLM."""

import pytest

from customizable_rlm import CustomizableRLM, DEFAULT_CHAR_LIMIT
from prompts import build_action_instructions


# ---------------------------------------------------------------------------
# chars_for_sub_lm
# ---------------------------------------------------------------------------

class TestCharsForSubLm:
    def test_token_only(self):
        result = CustomizableRLM.chars_for_sub_lm(sub_lm_context_tokens=32_768)
        # 32768 * 4 * 0.8 = 104857.6 -> 104857
        assert result == 104_857

    def test_token_with_custom_safety(self):
        result = CustomizableRLM.chars_for_sub_lm(
            sub_lm_context_tokens=32_768, safety_factor=0.6
        )
        assert result == int(32_768 * 4 * 0.6)

    def test_neither_returns_default(self):
        result = CustomizableRLM.chars_for_sub_lm()
        assert result == DEFAULT_CHAR_LIMIT

    def test_token_takes_precedence_over_lm(self):
        """When both sub_lm_context_tokens and lm are provided, tokens win."""
        import dspy
        lm = dspy.LM("openai/fake-model", api_key="fake", api_base="http://localhost:1")
        result = CustomizableRLM.chars_for_sub_lm(
            sub_lm_context_tokens=16_384, lm=lm
        )
        # Should use 16384 regardless of what litellm returns for the model
        assert result == int(16_384 * 4 * 0.8)

    def test_lm_only_with_bad_model_falls_back(self):
        """If litellm can't find the model, returns default."""
        import dspy
        lm = dspy.LM("openai/nonexistent-model-xyz", api_key="fake", api_base="http://localhost:1")
        result = CustomizableRLM.chars_for_sub_lm(lm=lm)
        assert result == DEFAULT_CHAR_LIMIT


# ---------------------------------------------------------------------------
# Prompt patching
# ---------------------------------------------------------------------------

class TestPromptPatching:
    def _build(self, sub_lm_context_tokens=None, small_model_tips=False):
        chars = CustomizableRLM.chars_for_sub_lm(sub_lm_context_tokens=sub_lm_context_tokens)
        return build_action_instructions(
            inputs="`context`, `query`",
            output_fields="- answer (str)",
            final_output_names="answer",
            max_llm_calls=50,
            sub_lm_context_chars=chars,
            small_model_tips=small_model_tips,
        )

    def test_default_keeps_500k(self):
        instructions = self._build()
        assert "~500K char capacity" in instructions

    def test_custom_tokens_patches_capacity(self):
        instructions = self._build(sub_lm_context_tokens=32_768)
        # 32768 * 4 * 0.8 = 104857 -> 104K
        assert "~104K char capacity" in instructions
        assert "500K" not in instructions

    def test_small_model_tips_batching_block(self):
        instructions = self._build(sub_lm_context_tokens=32_768, small_model_tips=True)
        assert "high runtime costs" in instructions
        assert "batch as much information" in instructions
        assert "104,857 characters per call" in instructions

    def test_small_model_tips_context_warning(self):
        """<= 100k chars should get the context-length warning."""
        # 25000 * 4 * 0.8 = 80000 chars, well under 100k threshold
        instructions = self._build(sub_lm_context_tokens=25_000, small_model_tips=True)
        chars = CustomizableRLM.chars_for_sub_lm(sub_lm_context_tokens=25_000)
        assert chars <= 100_000
        assert "total context window" in instructions
        assert "conservative with how much context" in instructions

    def test_medium_model_no_context_warning(self):
        """130k tokens -> ~416k chars, but with safety -> 130000*4*0.8=416000.
        Actually let's use 50k tokens -> 160k chars. > 100k so no context warning."""
        instructions = self._build(sub_lm_context_tokens=50_000, small_model_tips=True)
        chars = CustomizableRLM.chars_for_sub_lm(sub_lm_context_tokens=50_000)
        assert chars > 100_000
        assert chars <= 200_000
        # Should have batching block but NOT context-length warning
        assert "batch as much information" in instructions
        assert "total context window" not in instructions

    def test_large_model_no_tips(self):
        """Large model with small_model_tips=True but > 200k chars: no tips injected."""
        instructions = self._build(sub_lm_context_tokens=200_000, small_model_tips=True)
        chars = CustomizableRLM.chars_for_sub_lm(sub_lm_context_tokens=200_000)
        assert chars > 200_000
        assert "high runtime costs" not in instructions

    def test_no_tips_without_flag(self):
        """small_model_tips=False should never inject tips."""
        instructions = self._build(sub_lm_context_tokens=32_768, small_model_tips=False)
        assert "high runtime costs" not in instructions
        assert "total context window" not in instructions


# ---------------------------------------------------------------------------
# Interpreter crash recovery
# ---------------------------------------------------------------------------

class TestInterpreterCrashRecovery:
    """When the deno sandbox process dies mid-run (e.g. an unhandled async error
    during a heavy llm_query_batched call), PythonInterpreter auto-respawns on
    the next execute() but skips re-registration because _tools_registered stays
    True. The freshly spawned process ends up with only PYTHON_SETUP_CODE's
    default single-arg SUBMIT(output), so typed SUBMIT(answer=...) calls fail
    with "unexpected keyword argument", which is what broke the real trace.
    """

    def test_typed_submit_survives_deno_process_death(self):
        from dspy.primitives.python_interpreter import PythonInterpreter, FinalOutput
        from customizable_rlm import _force_registration_reset

        interp = PythonInterpreter(output_fields=[{"name": "answer"}])
        try:
            interp.execute("pass")

            before = interp.execute("SUBMIT(answer='before')")
            assert isinstance(before, FinalOutput)
            assert before.output == {"answer": "before"}

            interp.deno_process.kill()
            interp.deno_process.wait()

            # The fix: reset registration flags so the next execute()
            # re-registers against the fresh deno that _ensure_deno_process
            # auto-spawns. _execute_iteration calls this on every iteration.
            _force_registration_reset(interp)

            after = interp.execute("SUBMIT(answer='after')")
            assert isinstance(after, FinalOutput)
            assert after.output == {"answer": "after"}
        finally:
            interp.shutdown()

    def test_dead_readline_error_becomes_crash_observation(self):
        """When repl.execute raises the 'No output from Deno subprocess' error,
        the REPL observation handed back to the model must explicitly name the
        crash and tell it which variables are still available."""
        from dspy.primitives.code_interpreter import CodeInterpreterError
        from customizable_rlm import _format_execute_error

        crash = CodeInterpreterError("No output from Deno subprocess. Stderr: ")
        obs = _format_execute_error(crash, ["context", "query"])
        assert obs.startswith("[Error] Interpreter process crashed")
        assert "context, query" in obs
        assert "intermediate variables" in obs

    def test_other_errors_keep_plain_format(self):
        """Non-crash CodeInterpreterErrors must round-trip as the existing
        '[Error] <exc>' format so downstream handling is unchanged."""
        from dspy.primitives.code_interpreter import CodeInterpreterError
        from customizable_rlm import _format_execute_error

        other = CodeInterpreterError("TypeError: foo")
        assert _format_execute_error(other, ["context"]) == f"[Error] {other}"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
