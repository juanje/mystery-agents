"""Caching utilities for LLMs and Agents to improve performance and reduce costs."""

from __future__ import annotations

import logging
from typing import Any, Literal

from langchain_core.language_models import BaseChatModel

logger = logging.getLogger(__name__)


class LLMCache:
    """
    Singleton cache for LLM instances to avoid creating duplicate models.

    Benefits:
    - Reduces initialization overhead
    - Ensures consistent model configuration across the application
    - Reduces memory footprint (1 instance per tier instead of 20+)
    """

    _instance: LLMCache | None = None
    _cache: dict[str, BaseChatModel] = {}

    def __new__(cls) -> LLMCache:
        """Ensure singleton pattern."""
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance

    @classmethod
    def get_model(cls, tier: Literal["tier1", "tier2", "tier3"]) -> BaseChatModel:
        """
        Get cached LLM model instance for the specified tier.

        Args:
            tier: Model tier selection (tier1, tier2, or tier3)

        Returns:
            Cached chat model instance
        """
        cache = cls()._cache

        if tier not in cache:
            # Import here to avoid circular dependency
            from mystery_agents.config import LLMConfig

            logger.info(f"[Cache] Creating new LLM instance for {tier}")
            cache[tier] = LLMConfig.get_model(tier)
        else:
            logger.debug(f"[Cache] Reusing cached LLM instance for {tier}")

        return cache[tier]

    @classmethod
    def clear(cls) -> None:
        """Clear the LLM cache (useful for testing)."""
        logger.info("[Cache] Clearing LLM cache")
        cls()._cache.clear()

    @classmethod
    def cache_stats(cls) -> dict[str, Any]:
        """
        Get cache statistics.

        Returns:
            Dictionary with cache size and tier information
        """
        return {
            "cached_models": len(cls()._cache),
            "tiers": list(cls()._cache.keys()),
        }


class AgentFactory:
    """
    Singleton factory for agent instances to avoid creating duplicates.

    Benefits:
    - Agents are stateless, so we can safely reuse them
    - Reduces LLM instance creation (through LLMCache)
    - Reduces create_agent() calls from LangChain
    - Improves performance in retry loops (V1, V2)
    """

    _instance: AgentFactory | None = None
    _cache: dict[str, Any] = {}

    def __new__(cls) -> AgentFactory:
        """Ensure singleton pattern."""
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance

    @classmethod
    def get_agent(cls, agent_class: type[Any]) -> Any:
        """
        Get cached agent instance for the specified class.

        Args:
            agent_class: Agent class to instantiate (e.g., CharactersAgent)

        Returns:
            Cached agent instance
        """
        cache = cls()._cache
        agent_name = agent_class.__name__

        if agent_name not in cache:
            logger.info(f"[Cache] Creating new agent instance: {agent_name}")
            cache[agent_name] = agent_class()
        else:
            logger.debug(f"[Cache] Reusing cached agent: {agent_name}")

        return cache[agent_name]

    @classmethod
    def clear(cls) -> None:
        """Clear the agent cache (useful for testing)."""
        logger.info("[Cache] Clearing agent cache")
        cls()._cache.clear()

    @classmethod
    def cache_stats(cls) -> dict[str, Any]:
        """
        Get cache statistics.

        Returns:
            Dictionary with cache size and agent names
        """
        return {
            "cached_agents": len(cls()._cache),
            "agents": list(cls()._cache.keys()),
        }


def clear_all_caches() -> None:
    """
    Clear all application caches.

    Useful for:
    - Testing to ensure clean state
    - Forcing fresh LLM instances after configuration changes
    """
    logger.info("[Cache] Clearing all caches")
    LLMCache.clear()
    AgentFactory.clear()


def get_cache_stats() -> dict[str, Any]:
    """
    Get statistics for all caches.

    Returns:
        Dictionary with cache statistics
    """
    return {
        "llm_cache": LLMCache.cache_stats(),
        "agent_cache": AgentFactory.cache_stats(),
    }
