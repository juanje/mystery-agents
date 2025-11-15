"""LLM configuration for multi-tier agent system."""

import os
from typing import Literal

from langchain_core.language_models import BaseChatModel
from langchain_google_genai import ChatGoogleGenerativeAI


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

        Args:
            tier: Model tier selection
                - tier1: Logic/creativity (gemini-2.5-pro)
                - tier2: Content generation (gemini-2.5-pro)
                - tier3: Validation/simple tasks (gemini-2.5-flash)

        Returns:
            Configured chat model instance

        Note:
            Tier 1 and Tier 2 now use gemini-2.5-pro for better performance.
            Tier 3 uses gemini-2.5-flash for faster validation tasks.
        """
        api_key = os.getenv("GOOGLE_API_KEY")
        if not api_key:
            # In dry_run mode, agents use mocks, so API key not needed
            # Use a dummy key that will fail gracefully if actually used
            api_key = "dry-run-dummy-key"

        models: dict[str, BaseChatModel] = {
            "tier1": ChatGoogleGenerativeAI(
                model="gemini-2.5-pro",
                google_api_key=api_key,
                temperature=0.6,
            ),
            "tier2": ChatGoogleGenerativeAI(
                model="gemini-2.5-pro",
                google_api_key=api_key,
                temperature=0.7,
            ),
            "tier3": ChatGoogleGenerativeAI(
                model="gemini-2.5-flash",
                google_api_key=api_key,
                temperature=0.3,
            ),
        }
        return models[tier]
