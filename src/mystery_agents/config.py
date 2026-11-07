"""LLM configuration for multi-tier agent system."""

import os
from typing import Any, Literal

import openai
from langchain_core.language_models import BaseChatModel
from langchain_core.runnables import Runnable
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_openai import ChatOpenAI
from pydantic import BaseModel

from mystery_agents.utils.constants import (
    DRY_RUN_DUMMY_API_KEY,
    LLM_API_KEY_VARS,
    LLM_MODELS,
    LLM_PROVIDER_DEFAULT,
    LLM_REQUEST_TIMEOUT,
)

Tier = Literal["tier1", "tier2", "tier3"]
Provider = Literal["openai", "google"]
# Attempts for an OpenAI call that answers 403 while a model's access propagates
OPENAI_ACCESS_RETRIES = 5
PROVIDERS: tuple[Provider, ...] = ("openai", "google")


def _optional_temperature(env_var: str) -> float | None:
    """Read an optional temperature override from the environment."""
    value = os.getenv(env_var)
    return float(value) if value else None


def get_provider() -> Provider:
    """
    The text provider chosen with LLM_PROVIDER ("openai" or "google").

    Raises:
        ValueError: If LLM_PROVIDER names another provider
    """
    provider = os.getenv("LLM_PROVIDER", LLM_PROVIDER_DEFAULT).strip().lower()
    if provider not in PROVIDERS:
        raise ValueError(f"LLM_PROVIDER must be one of {', '.join(PROVIDERS)}, not {provider!r}")
    return provider


def api_key_var() -> str:
    """Name of the environment variable with the active text provider's API key."""
    return LLM_API_KEY_VARS[get_provider()]


def structured_output(llm: BaseChatModel, schema: type[BaseModel]) -> Runnable[Any, Any]:
    """
    Wrap a model so it answers with the given schema, keeping the raw message.

    Native JSON schema output avoids function calling (agents call no tools), and
    include_raw keeps the raw message for debugging and for reporting parsing
    failures. OpenAI runs it in strict mode, which guarantees the answer follows the
    schema; every game schema was checked to work that way.
    """
    if isinstance(llm, ChatOpenAI):
        # Newly enabled models answered 403 "does not have access" on a share of calls
        # for a while; a few retries ride it out, and a real denial still fails after them
        return llm.with_structured_output(
            schema, method="json_schema", include_raw=True, strict=True
        ).with_retry(
            retry_if_exception_type=(openai.PermissionDeniedError,),
            stop_after_attempt=OPENAI_ACCESS_RETRIES,
        )
    return llm.with_structured_output(schema, method="json_schema", include_raw=True)


class LLMConfig:
    """
    Abstracts LLM configuration for multi-tier agent system.

    Agents get a provider-agnostic BaseChatModel; LLM_PROVIDER chooses OpenAI (the
    default) or Google Gemini for every text tier. Image generation always uses Gemini.
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
            Default model names per provider are configured in constants.py. Temperature
            is not sent unless overridden: Gemini 3.6+ models use fixed sampling
            defaults and ignore custom sampling parameters.
        """
        provider = get_provider()
        # In dry_run mode agents use mocks, so a dummy key is enough; it fails if used
        api_key = os.getenv(LLM_API_KEY_VARS[provider]) or DRY_RUN_DUMMY_API_KEY
        number = tier.removeprefix("tier")
        model = os.getenv(f"LLM_MODEL_TIER{number}", LLM_MODELS[provider][tier])
        # None keeps the model's own default
        temperature = _optional_temperature(f"LLM_TEMPERATURE_TIER{number}")

        if provider == "openai":
            return ChatOpenAI(
                model=model,
                api_key=api_key,  # type: ignore[arg-type]
                temperature=temperature,
                timeout=LLM_REQUEST_TIMEOUT,
                # GPT-6 models are only served by the Responses API: Chat Completions
                # answers 403 "does not have access to model" for them
                use_responses_api=True,
            )
        return ChatGoogleGenerativeAI(
            model=model,
            google_api_key=api_key,
            temperature=temperature,
            timeout=LLM_REQUEST_TIMEOUT,
        )
