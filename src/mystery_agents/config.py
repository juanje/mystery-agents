"""LLM configuration for multi-tier agent system."""

import os
from typing import Literal

from langchain_core.language_models import BaseChatModel
from langchain_google_genai import ChatGoogleGenerativeAI

from mystery_agents.utils.constants import (
    DRY_RUN_DUMMY_API_KEY,
    LLM_MODEL_TIER1,
    LLM_MODEL_TIER2,
    LLM_MODEL_TIER3,
    LLM_REQUEST_TIMEOUT,
)

Tier = Literal["tier1", "tier2", "tier3"]

DEFAULT_MODELS: dict[Tier, str] = {
    "tier1": LLM_MODEL_TIER1,
    "tier2": LLM_MODEL_TIER2,
    "tier3": LLM_MODEL_TIER3,
}


def _optional_temperature(env_var: str) -> float | None:
    """Read an optional temperature override from the environment."""
    value = os.getenv(env_var)
    return float(value) if value else None


class LLMConfig:
    """
    Abstracts LLM configuration for multi-tier agent system.

    Provides a clean abstraction layer that allows switching between different
    LLM providers without changing agent code.

    Currently uses Google Gemini models. To switch to other providers:
    - OpenAI: from langchain_openai import ChatOpenAI
    - Anthropic: from langchain_anthropic import ChatAnthropic
    - Ollama: from langchain_ollama import ChatOllama
    """

    @staticmethod
    def get_model(tier: Tier) -> BaseChatModel:
        """
        Get LLM model for the specified tier.

        This method creates NEW instances each time. For cached instances,
        use LLMCache.get_model() instead (recommended for production).

        Args:
            tier: Model tier selection
                - tier1: World, crime, plot logic, the suspects' sheets, written content
                  and its reviews
                - tier2: World review, visual style, characters, relationships and the
                  shared timeline
                - tier3: Placeholder for agents that never call the model (images, packaging)

        Returns:
            Configured chat model instance

        Note:
            Model names and temperatures can be overridden via environment variables:
            - LLM_MODEL_TIER1, LLM_MODEL_TIER2, LLM_MODEL_TIER3
            - LLM_TEMPERATURE_TIER1, LLM_TEMPERATURE_TIER2, LLM_TEMPERATURE_TIER3
            Default model names are configured in constants.py. Temperature is not
            sent unless overridden: Gemini 3.6+ models use fixed sampling defaults
            and ignore custom sampling parameters.
        """
        api_key = os.getenv("GOOGLE_API_KEY")
        if not api_key:
            # In dry_run mode, agents use mocks, so API key not needed
            # Use a dummy key that will fail gracefully if actually used
            api_key = DRY_RUN_DUMMY_API_KEY

        number = tier.removeprefix("tier")
        return ChatGoogleGenerativeAI(
            model=os.getenv(f"LLM_MODEL_TIER{number}", DEFAULT_MODELS[tier]),
            google_api_key=api_key,
            # None keeps the model's own default
            temperature=_optional_temperature(f"LLM_TEMPERATURE_TIER{number}"),
            timeout=LLM_REQUEST_TIMEOUT,
        )
