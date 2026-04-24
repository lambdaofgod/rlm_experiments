"""Tests for FenceTolerantChatAdapter."""

import dspy
import pytest
from dspy.utils.exceptions import AdapterParseError

from adapters import FenceTolerantChatAdapter


@pytest.fixture
def action_signature():
    """Mimics the action signature from CustomizableRLM._build_signatures:
    reasoning + code output fields. Inputs are simplified for parse testing."""
    return dspy.Signature(
        "variables_info: str, repl_history: str, iteration: str -> reasoning: str, code: str"
    )


@pytest.fixture
def answer_signature():
    """Signature without a `code` field (e.g. the extract signature)."""
    return dspy.Signature("context: str, query: str -> answer: str")


class TestFenceTolerantParse:
    """The three failing LM-response shapes observed in Phoenix traces:
    reasoning present, code marker missing, markdown fence present."""

    def test_reasoning_and_fence_and_completed_marker(self, action_signature):
        completion = (
            "[[ ## reasoning ## ]]\n"
            "I found the BALANCE SHEETS section. From the output, I can see:\n\n"
            "**December 28, 2024:**\n- Total current assets: $19,049 million\n\n"
            "Now I need to calculate net working capital.\n\n"
            "```python\n"
            "nwc_2024 = 19049 - 7281\n"
            "print(nwc_2024)\n"
            "```\n\n"
            "[[ ## completed ## ]]\n"
        )
        result = FenceTolerantChatAdapter().parse(action_signature, completion)
        assert "reasoning" in result
        assert "I found the BALANCE SHEETS" in result["reasoning"]
        assert result["code"] == "nwc_2024 = 19049 - 7281\nprint(nwc_2024)"

    def test_reasoning_and_fence_without_completed_marker(self, action_signature):
        completion = (
            "[[ ## reasoning ## ]]\n"
            "Let me try a different approach.\n\n"
            "```python\n"
            "for i in range(3):\n"
            "    print(i)\n"
            "```\n"
        )
        result = FenceTolerantChatAdapter().parse(action_signature, completion)
        assert result["reasoning"].startswith("Let me try a different approach.")
        assert result["code"] == "for i in range(3):\n    print(i)"

    def test_reasoning_and_fence_with_duplicate_submit_markers(self, action_signature):
        completion = (
            "[[ ## reasoning ## ]]\n"
            "Based on my analysis...\n\n"
            "```python\n"
            "SUBMIT(answer=['A', 'B'])\n"
            "```\n\n"
            "[[ ## completed ## ]]\n\n"
            "[[ ## SUBMIT ## ]]\n"
            "done\n\n"
            "[[ ## SUBMIT ## ]]\n"
        )
        result = FenceTolerantChatAdapter().parse(action_signature, completion)
        assert "Based on my analysis" in result["reasoning"]
        assert result["code"] == "SUBMIT(answer=['A', 'B'])"

    def test_bare_fence_without_language_tag(self, action_signature):
        completion = (
            "[[ ## reasoning ## ]]\n"
            "Quick check.\n\n"
            "```\n"
            "x = 42\n"
            "```\n"
        )
        result = FenceTolerantChatAdapter().parse(action_signature, completion)
        assert result["code"] == "x = 42"


class TestFenceTolerantPassThrough:
    """Cases where the adapter should behave exactly like the stock ChatAdapter."""

    def test_happy_path_with_code_marker(self, action_signature):
        """When the LM obeys the marker convention, parse should succeed
        without invoking the fence fallback."""
        completion = (
            "[[ ## reasoning ## ]]\n"
            "Plan: print hello.\n\n"
            "[[ ## code ## ]]\n"
            "print('hello')\n\n"
            "[[ ## completed ## ]]\n"
        )
        result = FenceTolerantChatAdapter().parse(action_signature, completion)
        assert result["reasoning"] == "Plan: print hello."
        assert result["code"] == "print('hello')"

    def test_signature_without_code_field_reraises(self, answer_signature):
        """If the signature has no `code` field, the adapter should not try
        to synthesize one -- the original AdapterParseError must propagate."""
        completion = (
            "[[ ## reasoning ## ]]\n"
            "some reasoning without the answer field\n"
        )
        with pytest.raises(AdapterParseError):
            FenceTolerantChatAdapter().parse(answer_signature, completion)

    def test_multiple_missing_fields_reraises(self):
        """If more than just `code` is missing, the adapter does not attempt
        recovery -- the caller (or stock JSONAdapter fallback) handles it."""
        sig = dspy.Signature(
            "x: str -> reasoning: str, code: str, summary: str"
        )
        completion = (
            "[[ ## reasoning ## ]]\n"
            "only reasoning here\n\n"
            "```python\n"
            "x = 1\n"
            "```\n"
        )
        with pytest.raises(AdapterParseError):
            FenceTolerantChatAdapter().parse(sig, completion)

    def test_code_missing_but_no_fence_reraises(self, action_signature):
        """If code is missing and there is no fence to extract from, re-raise."""
        completion = (
            "[[ ## reasoning ## ]]\n"
            "Just prose, no code block.\n\n"
            "[[ ## completed ## ]]\n"
        )
        with pytest.raises(AdapterParseError):
            FenceTolerantChatAdapter().parse(action_signature, completion)
