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


def describe_response(raw: BaseMessage | None) -> str:
    """
    Why a model response ended and how long it was, for a failure message.

    A response cut at the output limit (often a model repeating itself) looks like any
    other parsing failure unless the finish reason and token count are shown.
    """
    if raw is None:
        return "no response"
    meta = raw.response_metadata or {}
    reason = (
        meta.get("finish_reason")
        or meta.get("stop_reason")
        or (meta.get("incomplete_details") or {}).get("reason")
        or meta.get("status")
        or "unknown"
    )
    usage = getattr(raw, "usage_metadata", None) or {}
    tokens = usage.get("output_tokens")
    reasoning = (usage.get("output_token_details") or {}).get("reasoning")
    detail = f"finish reason {reason}"
    if tokens is not None:
        detail += f", {tokens} output tokens"
    if reasoning:
        detail += f" ({reasoning} reasoning)"
    return detail
