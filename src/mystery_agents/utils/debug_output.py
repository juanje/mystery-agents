"""Debug logging of raw model responses (enabled with --debug)."""

import json

from langchain_core.messages import BaseMessage
from pydantic import BaseModel

PREVIEW_CHARS = 1000


def _preview(text: str) -> str:
    """Truncate long text, noting the original length."""
    if len(text) <= PREVIEW_CHARS:
        return text
    return f"{text[:PREVIEW_CHARS]}\n... (truncated, total length: {len(text)} chars)"


def log_model_response(
    raw: BaseMessage | None,
    parsed: BaseModel | None,
    parsing_error: BaseException | None = None,
) -> None:
    """
    Print a model response for debugging.

    Args:
        raw: Raw message returned by the model
        parsed: Structured output parsed from the response, if any
        parsing_error: Error raised while parsing the response, if any
    """
    print("\n" + "=" * 80)
    print("[DEBUG] Model response")
    print("=" * 80)

    if raw is None:
        print("[DEBUG] (no raw message)")
    else:
        print(f"[DEBUG] Message type: {type(raw).__name__}")
        # .text joins only the text parts, skipping blocks such as thought signatures
        text = raw.text
        print("-" * 80)
        print(_preview(text) if text else "(empty content)")
        print("-" * 80)

    if parsed is not None:
        print(f"[DEBUG] ✓ Structured output: {type(parsed).__name__}")
        print(_preview(json.dumps(parsed.model_dump(), indent=2, default=str)))
    else:
        print("[DEBUG] ✗ No structured output")

    if parsing_error is not None:
        print(f"[DEBUG] ✗ Parsing error: {parsing_error}")

    print("=" * 80 + "\n")
