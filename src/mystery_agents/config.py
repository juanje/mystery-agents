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
    LLM_TEMPERATURE_TIER1,
    LLM_TEMPERATURE_TIER2,
    LLM_TEMPERATURE_TIER3,
)


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
            Default values are configured in constants.py
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

        # Get temperatures from environment or use defaults
        temp_tier1 = float(os.getenv("LLM_TEMPERATURE_TIER1", str(LLM_TEMPERATURE_TIER1)))
        temp_tier2 = float(os.getenv("LLM_TEMPERATURE_TIER2", str(LLM_TEMPERATURE_TIER2)))
        temp_tier3 = float(os.getenv("LLM_TEMPERATURE_TIER3", str(LLM_TEMPERATURE_TIER3)))

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
