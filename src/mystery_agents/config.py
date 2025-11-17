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
            Model names and temperatures are configured in constants.py
        """
        api_key = os.getenv("GOOGLE_API_KEY")
        if not api_key:
            # In dry_run mode, agents use mocks, so API key not needed
            # Use a dummy key that will fail gracefully if actually used
            api_key = DRY_RUN_DUMMY_API_KEY

        models: dict[str, BaseChatModel] = {
            "tier1": ChatGoogleGenerativeAI(
                model=LLM_MODEL_TIER1,
                google_api_key=api_key,
                temperature=LLM_TEMPERATURE_TIER1,
            ),
            "tier2": ChatGoogleGenerativeAI(
                model=LLM_MODEL_TIER2,
                google_api_key=api_key,
                temperature=LLM_TEMPERATURE_TIER2,
            ),
            "tier3": ChatGoogleGenerativeAI(
                model=LLM_MODEL_TIER3,
                google_api_key=api_key,
                temperature=LLM_TEMPERATURE_TIER3,
            ),
        }
        return models[tier]
