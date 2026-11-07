"""Tests for LLMConfig."""

import pytest
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_openai import ChatOpenAI
from openai.lib._pydantic import to_strict_json_schema
from pydantic import BaseModel

from mystery_agents.config import LLMConfig, api_key_var, get_provider
from mystery_agents.utils.constants import LLM_MODELS, LLM_REQUEST_TIMEOUT


@pytest.fixture(autouse=True)
def _default_provider(monkeypatch: pytest.MonkeyPatch) -> None:
    """Each test starts from the defaults, whatever the developer's .env says."""
    for var in ("LLM_PROVIDER", "LLM_MODEL_TIER1", "LLM_MODEL_TIER2", "LLM_MODEL_TIER3"):
        monkeypatch.delenv(var, raising=False)


def test_openai_is_the_default_provider() -> None:
    """Without LLM_PROVIDER, text tiers use OpenAI's default model."""
    model = LLMConfig.get_model("tier1")

    assert get_provider() == "openai"
    assert isinstance(model, ChatOpenAI)
    assert model.model_name == LLM_MODELS["openai"]["tier1"]
    assert api_key_var() == "OPENAI_API_KEY"


def test_google_provider_and_model_override(monkeypatch: pytest.MonkeyPatch) -> None:
    """LLM_PROVIDER=google switches back to Gemini; LLM_MODEL_TIER* still overrides a tier."""
    monkeypatch.setenv("LLM_PROVIDER", "Google")
    monkeypatch.setenv("LLM_MODEL_TIER2", "gemini-custom")

    tier1 = LLMConfig.get_model("tier1")
    tier2 = LLMConfig.get_model("tier2")

    assert isinstance(tier1, ChatGoogleGenerativeAI)
    assert tier1.model.endswith(LLM_MODELS["google"]["tier1"])
    assert tier2.model.endswith("gemini-custom")
    assert api_key_var() == "GOOGLE_API_KEY"


def test_unknown_provider_is_rejected(monkeypatch: pytest.MonkeyPatch) -> None:
    """A typo in LLM_PROVIDER fails loudly instead of silently using another provider."""
    monkeypatch.setenv("LLM_PROVIDER", "openia")

    with pytest.raises(ValueError, match="LLM_PROVIDER"):
        LLMConfig.get_model("tier1")


@pytest.mark.parametrize("provider", ["openai", "google"])
def test_temperature_unset_by_default(provider: str, monkeypatch: pytest.MonkeyPatch) -> None:
    """No temperature is sent unless overridden: each model keeps its own default."""
    monkeypatch.setenv("LLM_PROVIDER", provider)
    monkeypatch.delenv("LLM_TEMPERATURE_TIER2", raising=False)

    model = LLMConfig.get_model("tier2")

    assert model.temperature is None  # type: ignore[attr-defined]


def test_temperature_env_override(monkeypatch: pytest.MonkeyPatch) -> None:
    """An explicit environment override is still honored."""
    monkeypatch.setenv("LLM_TEMPERATURE_TIER1", "0.4")

    model = LLMConfig.get_model("tier1")

    assert model.temperature == 0.4  # type: ignore[attr-defined]


@pytest.mark.parametrize("provider", ["openai", "google"])
def test_models_have_request_timeout(provider: str, monkeypatch: pytest.MonkeyPatch) -> None:
    """A dropped connection must fail instead of hanging the run forever."""
    monkeypatch.setenv("LLM_PROVIDER", provider)

    model = LLMConfig.get_model("tier3")

    timeout = model.request_timeout if isinstance(model, ChatOpenAI) else model.timeout  # type: ignore[attr-defined]
    assert timeout == LLM_REQUEST_TIMEOUT


def _response_schemas() -> list[type[BaseModel]]:
    """Every schema an agent asks a model to answer with, the per-game ones included."""
    from mystery_agents.agents.a2_5_visual_style import A2_5Output
    from mystery_agents.agents.a2_world import A2Output
    from mystery_agents.agents.a3_characters import A3Output
    from mystery_agents.agents.a4_relationships import A4Output
    from mystery_agents.agents.a5_crime import A5Output
    from mystery_agents.agents.a7_4_presence import output_schema_for_rooms
    from mystery_agents.agents.a7_5_personal_timelines import SheetsOutput
    from mystery_agents.agents.a8_content import CluesOutput, GuideOutput, SolutionOutput
    from mystery_agents.agents.v1_world_validator import V1Output
    from mystery_agents.models.state import GlobalTimeline, KillerSelection, ValidationReport

    return [
        A2Output,
        A2_5Output,
        A3Output,
        A4Output,
        A5Output,
        GlobalTimeline,
        KillerSelection,
        output_schema_for_rooms(["Biblioteca", "Salón"]),
        SheetsOutput,
        ValidationReport,
        CluesOutput,
        GuideOutput,
        SolutionOutput,
        V1Output,
    ]


@pytest.mark.parametrize("schema", _response_schemas(), ids=lambda schema: schema.__name__)
def test_response_schemas_work_in_openai_strict_mode(schema: type[BaseModel]) -> None:
    """OpenAI runs structured output in strict mode; a schema it can't take fails here, not mid-game."""
    strict = to_strict_json_schema(schema)

    assert strict["additionalProperties"] is False
    assert set(strict["required"]) == set(strict["properties"])
