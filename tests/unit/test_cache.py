"""Tests for caching utilities (LLMCache and AgentFactory)."""

import pytest

from mystery_agents.agents.a2_world import WorldAgent
from mystery_agents.agents.a3_characters import CharactersAgent
from mystery_agents.agents.v1_validator import ValidationAgent
from mystery_agents.utils.cache import (
    AgentFactory,
    LLMCache,
    clear_all_caches,
    get_cache_stats,
)


@pytest.fixture(autouse=True)
def clear_caches_before_test() -> None:
    """Clear all caches before each test to ensure clean state."""
    clear_all_caches()


@pytest.fixture
def mock_google_api_key(monkeypatch: pytest.MonkeyPatch) -> None:
    """Set a mock API key for testing."""
    monkeypatch.setenv("GOOGLE_API_KEY", "test-api-key-for-cache-testing")


def test_llm_cache_singleton() -> None:
    """Test that LLMCache is a singleton."""
    cache1 = LLMCache()
    cache2 = LLMCache()
    assert cache1 is cache2


def test_llm_cache_returns_same_instance(mock_google_api_key: None) -> None:
    """Test that LLMCache returns the same LLM instance for the same tier."""
    llm1 = LLMCache.get_model("tier1")
    llm2 = LLMCache.get_model("tier1")

    # Should be the exact same object (not just equal)
    assert llm1 is llm2


def test_llm_cache_different_tiers(mock_google_api_key: None) -> None:
    """Test that LLMCache returns different instances for different tiers."""
    tier1_llm = LLMCache.get_model("tier1")
    tier2_llm = LLMCache.get_model("tier2")
    tier3_llm = LLMCache.get_model("tier3")

    # Each tier should have a different LLM instance
    assert tier1_llm is not tier2_llm
    assert tier1_llm is not tier3_llm
    assert tier2_llm is not tier3_llm


def test_llm_cache_stats(mock_google_api_key: None) -> None:
    """Test cache statistics."""
    # Initially empty
    stats = LLMCache.cache_stats()
    assert stats["cached_models"] == 0
    assert stats["tiers"] == []

    # After requesting tier1
    LLMCache.get_model("tier1")
    stats = LLMCache.cache_stats()
    assert stats["cached_models"] == 1
    assert "tier1" in stats["tiers"]

    # After requesting tier2 and tier3
    LLMCache.get_model("tier2")
    LLMCache.get_model("tier3")
    stats = LLMCache.cache_stats()
    assert stats["cached_models"] == 3
    assert set(stats["tiers"]) == {"tier1", "tier2", "tier3"}


def test_llm_cache_clear() -> None:
    """Test clearing the LLM cache."""
    # Populate cache
    LLMCache.get_model("tier1")
    assert LLMCache.cache_stats()["cached_models"] == 1

    # Clear and verify
    LLMCache.clear()
    stats = LLMCache.cache_stats()
    assert stats["cached_models"] == 0
    assert stats["tiers"] == []


def test_agent_factory_singleton() -> None:
    """Test that AgentFactory is a singleton."""
    factory1 = AgentFactory()
    factory2 = AgentFactory()
    assert factory1 is factory2


def test_agent_factory_returns_same_instance(mock_google_api_key: None) -> None:
    """Test that AgentFactory returns the same agent instance for the same class."""
    agent1 = AgentFactory.get_agent(WorldAgent)
    agent2 = AgentFactory.get_agent(WorldAgent)

    # Should be the exact same object
    assert agent1 is agent2


def test_agent_factory_different_classes(mock_google_api_key: None) -> None:
    """Test that AgentFactory returns different instances for different classes."""
    world_agent = AgentFactory.get_agent(WorldAgent)
    char_agent = AgentFactory.get_agent(CharactersAgent)
    val_agent = AgentFactory.get_agent(ValidationAgent)

    # Each agent class should have a different instance
    assert world_agent is not char_agent
    assert world_agent is not val_agent
    assert char_agent is not val_agent


def test_agent_factory_reuses_llm(mock_google_api_key: None) -> None:
    """Test that agents created by AgentFactory share the same LLM instances."""
    # Get tier1 LLM directly
    tier1_llm = LLMCache.get_model("tier1")

    # Get agents that use tier1
    world_agent = AgentFactory.get_agent(WorldAgent)
    val_agent = AgentFactory.get_agent(ValidationAgent)

    # Both agents should use the same LLM instance
    assert world_agent.llm is tier1_llm
    assert val_agent.llm is tier1_llm


def test_agent_factory_stats(mock_google_api_key: None) -> None:
    """Test agent factory statistics."""
    # Initially empty
    stats = AgentFactory.cache_stats()
    assert stats["cached_agents"] == 0
    assert stats["agents"] == []

    # After creating some agents
    AgentFactory.get_agent(WorldAgent)
    AgentFactory.get_agent(CharactersAgent)

    stats = AgentFactory.cache_stats()
    assert stats["cached_agents"] == 2
    assert "WorldAgent" in stats["agents"]
    assert "CharactersAgent" in stats["agents"]


def test_agent_factory_clear(mock_google_api_key: None) -> None:
    """Test clearing the agent factory cache."""
    # Populate cache
    AgentFactory.get_agent(WorldAgent)
    assert AgentFactory.cache_stats()["cached_agents"] == 1

    # Clear and verify
    AgentFactory.clear()
    stats = AgentFactory.cache_stats()
    assert stats["cached_agents"] == 0
    assert stats["agents"] == []


def test_clear_all_caches(mock_google_api_key: None) -> None:
    """Test clearing all caches at once."""
    # Populate both caches
    LLMCache.get_model("tier1")
    AgentFactory.get_agent(WorldAgent)

    assert LLMCache.cache_stats()["cached_models"] == 1
    assert AgentFactory.cache_stats()["cached_agents"] == 1

    # Clear all
    clear_all_caches()

    assert LLMCache.cache_stats()["cached_models"] == 0
    assert AgentFactory.cache_stats()["cached_agents"] == 0


def test_get_cache_stats(mock_google_api_key: None) -> None:
    """Test getting combined cache statistics."""
    # Initially empty
    stats = get_cache_stats()
    assert stats["llm_cache"]["cached_models"] == 0
    assert stats["agent_cache"]["cached_agents"] == 0

    # Populate caches
    LLMCache.get_model("tier1")
    LLMCache.get_model("tier2")
    AgentFactory.get_agent(WorldAgent)

    # Check combined stats
    stats = get_cache_stats()
    assert stats["llm_cache"]["cached_models"] == 2
    assert stats["agent_cache"]["cached_agents"] == 1


def test_cache_performance_benefit(mock_google_api_key: None) -> None:
    """Test that caching provides performance benefit (not creating duplicate instances)."""
    # Request the same agent multiple times (simulating retry loops)
    agents = [AgentFactory.get_agent(WorldAgent) for _ in range(10)]

    # All should be the exact same instance
    assert all(agent is agents[0] for agent in agents)

    # Cache should only have 1 agent
    stats = AgentFactory.cache_stats()
    assert stats["cached_agents"] == 1

    # Similarly for LLMs
    llms = [LLMCache.get_model("tier1") for _ in range(10)]
    assert all(llm is llms[0] for llm in llms)

    llm_stats = LLMCache.cache_stats()
    assert llm_stats["cached_models"] == 1
