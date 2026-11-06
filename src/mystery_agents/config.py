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
)


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
    def get_model(tier: Literal["tier1", "tier2", "tier3"]) -> BaseChatModel:
        """
        Get LLM model for the specified tier.

        This method creates NEW instances each time. For cached instances,
        use LLMCache.get_model() instead (recommended for production).

        Args:
            tier: Model tier selection
                - tier1: Logic/creativity
                - tier2: Content generation
                - tier3: Validation/simple tasks

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

        # Get model names from environment or use defaults
        model_tier1 = os.getenv("LLM_MODEL_TIER1", LLM_MODEL_TIER1)
        model_tier2 = os.getenv("LLM_MODEL_TIER2", LLM_MODEL_TIER2)
        model_tier3 = os.getenv("LLM_MODEL_TIER3", LLM_MODEL_TIER3)

        # Temperatures are optional overrides (None keeps the model's own default)
        temp_tier1 = _optional_temperature("LLM_TEMPERATURE_TIER1")
        temp_tier2 = _optional_temperature("LLM_TEMPERATURE_TIER2")
        temp_tier3 = _optional_temperature("LLM_TEMPERATURE_TIER3")

        models: dict[str, BaseChatModel] = {
            "tier1": ChatGoogleGenerativeAI(
                model=model_tier1,
                google_api_key=api_key,
                temperature=temp_tier1,
            ),
            "tier2": ChatGoogleGenerativeAI(
                model=model_tier2,
                google_api_key=api_key,
                temperature=temp_tier2,
            ),
            "tier3": ChatGoogleGenerativeAI(
                model=model_tier3,
                google_api_key=api_key,
                temperature=temp_tier3,
            ),
        }
        return models[tier]
