"""Tests for debug output of model responses."""

import pytest
from langchain_core.messages import AIMessage
from pydantic import BaseModel

from mystery_agents.utils.debug_output import PREVIEW_CHARS, log_model_response


class _Output(BaseModel):
    value: str


def test_logs_raw_and_parsed_output(capsys: pytest.CaptureFixture[str]) -> None:
    """Both the raw content and the structured output are printed."""
    log_model_response(AIMessage(content='{"value": "x"}'), _Output(value="x"))

    out = capsys.readouterr().out
    assert '{"value": "x"}' in out
    assert "✓ Structured output: _Output" in out


def test_logs_parsing_error(capsys: pytest.CaptureFixture[str]) -> None:
    """A parsing failure is reported."""
    log_model_response(AIMessage(content="not json"), None, ValueError("bad json"))

    out = capsys.readouterr().out
    assert "✗ No structured output" in out
    assert "Parsing error: bad json" in out


def test_truncates_long_content(capsys: pytest.CaptureFixture[str]) -> None:
    """Long responses are truncated to a preview."""
    log_model_response(AIMessage(content="a" * (PREVIEW_CHARS + 50)), None)

    assert f"total length: {PREVIEW_CHARS + 50} chars" in capsys.readouterr().out


def test_handles_missing_raw_message(capsys: pytest.CaptureFixture[str]) -> None:
    """A missing raw message does not crash the logger."""
    log_model_response(None, None)

    assert "(no raw message)" in capsys.readouterr().out
