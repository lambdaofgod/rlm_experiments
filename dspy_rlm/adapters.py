"""Local DSPy adapter subclasses."""

from __future__ import annotations

import logging
import re
from typing import Any, TYPE_CHECKING

from dspy.adapters.chat_adapter import ChatAdapter
from dspy.utils.exceptions import AdapterParseError

if TYPE_CHECKING:
    from dspy.signatures.signature import Signature

logger = logging.getLogger(__name__)

_FENCE_IN_BODY_PATTERN = re.compile(r"```(?:python|py)?\s*\n(.*?)\n```", re.DOTALL)


class FenceTolerantChatAdapter(ChatAdapter):
    """ChatAdapter that recovers the `code` field from a markdown fence.

    When the sub-LM skips the `[[ ## code ## ]]` marker but emits a
    ```python ... ``` fence inside the reasoning body, upstream
    ChatAdapter.parse fails with AdapterParseError. This subclass retries
    the parse by extracting the fence contents into the `code` field
    when `code` is the only missing output field. Any other missing-field
    shape re-raises the original error untouched.
    """

    def parse(self, signature: type[Signature], completion: str) -> dict[str, Any]:
        try:
            return super().parse(signature, completion)
        except AdapterParseError as e:
            if "code" not in signature.output_fields:
                raise
            parsed = e.parsed_result if isinstance(e.parsed_result, dict) else {}
            missing = set(signature.output_fields) - parsed.keys()
            if missing != {"code"}:
                raise
            match = _FENCE_IN_BODY_PATTERN.search(completion)
            if not match:
                raise
            parsed["code"] = match.group(1)
            logger.debug(
                "ChatAdapter missed [[ ## code ## ]] marker; recovered code "
                "from markdown fence (%d chars)",
                len(parsed["code"]),
            )
            return parsed
