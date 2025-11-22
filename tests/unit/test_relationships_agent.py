"""Unit tests for RelationshipsAgent (A4)."""

import pytest

from mystery_agents.agents.a2_world import WorldAgent
from mystery_agents.agents.a3_characters import CharactersAgent
from mystery_agents.agents.a4_relationships import RelationshipsAgent
from mystery_agents.models.state import GameConfig, GameState, MetaInfo, PlayerConfig
from mystery_agents.utils.constants import TEST_DEFAULT_DURATION, TEST_DEFAULT_PLAYERS


@pytest.fixture
def state_with_characters() -> GameState:
    """Create a state with world and characters."""
    state = GameState(
        meta=MetaInfo(),
        config=GameConfig(
            players=PlayerConfig(total=TEST_DEFAULT_PLAYERS),
            host_gender="male",
            duration_minutes=TEST_DEFAULT_DURATION,
            dry_run=True,
        ),
    )

    # Generate world
    world_agent = WorldAgent()
    state = world_agent.run(state)

    # Generate characters
    char_agent = CharactersAgent()
    state = char_agent.run(state)

    return state


def test_relationships_agent_initialization() -> None:
    """Test RelationshipsAgent initializes correctly."""
    agent = RelationshipsAgent()

    assert agent.llm is not None
    assert agent.response_format is not None


def test_relationships_agent_get_system_prompt(state_with_characters: GameState) -> None:
    """Test get_system_prompt returns non-empty string."""
    agent = RelationshipsAgent()

    prompt = agent.get_system_prompt(state_with_characters)

    assert isinstance(prompt, str)
    assert len(prompt) > 0


def test_relationships_agent_mock_output_with_sufficient_characters(
    state_with_characters: GameState,
) -> None:
    """Test _mock_output generates relationships when enough characters exist."""
    agent = RelationshipsAgent()

    # Ensure we have at least 2 characters
    assert len(state_with_characters.characters) >= 2

    result = agent._mock_output(state_with_characters)

    assert len(result.relationships) >= 1
    assert result.relationships[0].from_character_id == state_with_characters.characters[0].id
    assert result.relationships[0].to_character_id == state_with_characters.characters[1].id
    assert result.relationships[0].type == "professional"


def test_relationships_agent_mock_output_with_many_characters() -> None:
    """Test _mock_output generates multiple relationships with 3+ characters."""
    state = GameState(
        meta=MetaInfo(),
        config=GameConfig(
            players=PlayerConfig(total=6),  # More characters
            host_gender="male",
            duration_minutes=TEST_DEFAULT_DURATION,
            dry_run=True,
        ),
    )

    # Generate world and characters
    world_agent = WorldAgent()
    state = world_agent.run(state)

    char_agent = CharactersAgent()
    state = char_agent.run(state)

    assert len(state.characters) >= 3

    # Generate relationships
    agent = RelationshipsAgent()
    result = agent._mock_output(state)

    # Should have at least 2 relationships
    assert len(result.relationships) >= 2
    assert result.relationships[1].type == "rivalry"


def test_relationships_agent_mock_output_with_insufficient_characters() -> None:
    """Test _mock_output handles case with insufficient characters."""
    state = GameState(
        meta=MetaInfo(),
        config=GameConfig(
            players=PlayerConfig(total=4),  # Minimum 4 characters
            host_gender="male",
            duration_minutes=TEST_DEFAULT_DURATION,
            dry_run=True,
        ),
    )

    # Generate world and characters
    world_agent = WorldAgent()
    state = world_agent.run(state)

    char_agent = CharactersAgent()
    state = char_agent.run(state)

    # Manually set only 1 character to test edge case
    if len(state.characters) > 1:
        state.characters = state.characters[:1]

    # Generate relationships
    agent = RelationshipsAgent()
    result = agent._mock_output(state)

    # Should return state without error (no relationships added if < 2 chars)
    assert len(result.relationships) == 0


def test_relationships_agent_run_dry_run(state_with_characters: GameState) -> None:
    """Test run method in dry run mode."""
    agent = RelationshipsAgent()

    result = agent.run(state_with_characters)

    assert len(result.relationships) >= 0
    # In dry run, should use mock output
    if len(result.characters) >= 2:
        assert len(result.relationships) >= 1
