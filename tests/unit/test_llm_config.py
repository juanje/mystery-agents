"""Tests for LLMConfig."""

import pytest

from mystery_agents.config import LLMConfig
from mystery_agents.utils.constants import LLM_REQUEST_TIMEOUT


def test_temperature_unset_by_default(monkeypatch: pytest.MonkeyPatch) -> None:
    """Gemini 3.6+ ignores custom sampling, so no temperature is sent by default."""
    monkeypatch.delenv("LLM_TEMPERATURE_TIER2", raising=False)

    model = LLMConfig.get_model("tier2")

    assert model.temperature is None  # type: ignore[attr-defined]


def test_temperature_env_override(monkeypatch: pytest.MonkeyPatch) -> None:
    """An explicit environment override is still honored."""
    monkeypatch.setenv("LLM_TEMPERATURE_TIER1", "0.4")

    model = LLMConfig.get_model("tier1")

    assert model.temperature == 0.4  # type: ignore[attr-defined]


def test_models_have_request_timeout() -> None:
    """A dropped connection must fail instead of hanging the run forever."""
    model = LLMConfig.get_model("tier3")

    assert model.timeout == LLM_REQUEST_TIMEOUT  # type: ignore[attr-defined]
