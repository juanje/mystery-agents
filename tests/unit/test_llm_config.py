"""Tests for LLMConfig."""

import pytest

from mystery_agents.config import LLMConfig


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
