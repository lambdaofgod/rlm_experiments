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
# Prompt patching
# ---------------------------------------------------------------------------

class TestPromptPatching:
    def _make_rlm(self, sub_lm_context_tokens=None, small_model_tips=False, **kw):
        return CustomizableRLM(
            "context, query -> answer",
            sub_lm_context_tokens=sub_lm_context_tokens,
            small_model_tips=small_model_tips,
            **kw,
        )

    def test_default_keeps_500k(self):
        rlm = self._make_rlm()
        instructions = rlm.generate_action.signature.instructions
        assert "~500K char capacity" in instructions

    def test_custom_tokens_patches_capacity(self):
        rlm = self._make_rlm(sub_lm_context_tokens=32_768)
        instructions = rlm.generate_action.signature.instructions
        # 32768 * 4 * 0.8 = 104857 -> 104K
        assert "~104K char capacity" in instructions
        assert "500K" not in instructions

    def test_small_model_tips_batching_block(self):
        rlm = self._make_rlm(sub_lm_context_tokens=32_768, small_model_tips=True)
        instructions = rlm.generate_action.signature.instructions
        assert "high runtime costs" in instructions
        assert "batch as much information" in instructions
        assert "104,857 characters per call" in instructions

    def test_small_model_tips_context_warning(self):
        """<= 100k chars should get the context-length warning."""
        # 25000 * 4 * 0.8 = 80000 chars, well under 100k threshold
        rlm = self._make_rlm(sub_lm_context_tokens=25_000, small_model_tips=True)
        instructions = rlm.generate_action.signature.instructions
        chars = CustomizableRLM.chars_for_sub_lm(sub_lm_context_tokens=25_000)
        assert chars <= 100_000
        assert "total context window" in instructions
        assert "conservative with how much context" in instructions

    def test_medium_model_no_context_warning(self):
        """130k tokens -> ~416k chars, but with safety -> 130000*4*0.8=416000.
        Actually let's use 50k tokens -> 160k chars. > 100k so no context warning."""
        rlm = self._make_rlm(sub_lm_context_tokens=50_000, small_model_tips=True)
        instructions = rlm.generate_action.signature.instructions
        chars = CustomizableRLM.chars_for_sub_lm(sub_lm_context_tokens=50_000)
        assert chars > 100_000
        assert chars <= 200_000
        # Should have batching block but NOT context-length warning
        assert "batch as much information" in instructions
        assert "total context window" not in instructions

    def test_large_model_no_tips(self):
        """Large model with small_model_tips=True but > 200k chars: no tips injected."""
        rlm = self._make_rlm(sub_lm_context_tokens=200_000, small_model_tips=True)
        instructions = rlm.generate_action.signature.instructions
        chars = CustomizableRLM.chars_for_sub_lm(sub_lm_context_tokens=200_000)
        assert chars > 200_000
        assert "high runtime costs" not in instructions

    def test_no_tips_without_flag(self):
        """small_model_tips=False should never inject tips."""
        rlm = self._make_rlm(sub_lm_context_tokens=32_768, small_model_tips=False)
        instructions = rlm.generate_action.signature.instructions
        assert "high runtime costs" not in instructions
        assert "total context window" not in instructions


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
