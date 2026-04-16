"""Tests for CustomizableRLM."""

import pytest

from customizable_rlm import CustomizableRLM, DEFAULT_CHAR_LIMIT


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
