"""Unit tests for validator agents (V1, V2)."""

import pytest

from mystery_agents.agents.a2_world import WorldAgent
from mystery_agents.agents.a3_characters import CharactersAgent
from mystery_agents.agents.a4_relationships import RelationshipsAgent
from mystery_agents.agents.a5_crime import CrimeAgent
from mystery_agents.agents.a6_timeline import TimelineAgent
from mystery_agents.agents.a7_killer_selection import KillerSelectionAgent
from mystery_agents.agents.v1_world_validator import WorldValidatorAgent
from mystery_agents.agents.v2_game_logic_validator import GameLogicValidatorAgent
from mystery_agents.models.state import (
    GameConfig,
    GameState,
    MetaInfo,
    PlayerConfig,
    ValidationReport,
    WorldValidation,
)
from mystery_agents.utils.constants import TEST_DEFAULT_DURATION, TEST_DEFAULT_PLAYERS


@pytest.fixture
def state_with_world() -> GameState:
    """Create a state with world."""
    state = GameState(
        meta=MetaInfo(),
        config=GameConfig(
            players=PlayerConfig(total=TEST_DEFAULT_PLAYERS),
            host_gender="male",
            duration_minutes=TEST_DEFAULT_DURATION,
            dry_run=True,
        ),
    )

    world_agent = WorldAgent()
    return world_agent.run(state)


@pytest.fixture
def state_with_full_game() -> GameState:
    """Create a state with full game setup including killer selection."""
    state = GameState(
        meta=MetaInfo(),
        config=GameConfig(
            players=PlayerConfig(total=TEST_DEFAULT_PLAYERS),
            host_gender="male",
            duration_minutes=TEST_DEFAULT_DURATION,
            dry_run=True,
        ),
    )

    # Build full pipeline
    world_agent = WorldAgent()
    state = world_agent.run(state)

    chars_agent = CharactersAgent()
    state = chars_agent.run(state)

    rels_agent = RelationshipsAgent()
    state = rels_agent.run(state)

    crime_agent = CrimeAgent()
    state = crime_agent.run(state)

    timeline_agent = TimelineAgent()
    state = timeline_agent.run(state)

    killer_agent = KillerSelectionAgent()
    state = killer_agent.run(state)

    return state


def test_world_validator_initialization() -> None:
    """Test WorldValidatorAgent initializes correctly."""
    agent = WorldValidatorAgent()

    assert agent.llm is not None
    assert agent.response_format is not None


def test_world_validator_get_system_prompt(state_with_world: GameState) -> None:
    """Test WorldValidatorAgent get_system_prompt."""
    agent = WorldValidatorAgent()

    prompt = agent.get_system_prompt(state_with_world)

    assert isinstance(prompt, str)
    assert len(prompt) > 0


def test_world_validator_mock_output(state_with_world: GameState) -> None:
    """Test WorldValidatorAgent _mock_output."""
    agent = WorldValidatorAgent()

    result = agent._mock_output(state_with_world)

    assert result.world_validation is not None
    assert isinstance(result.world_validation, WorldValidation)
    assert result.world_validation.is_coherent is True
    assert len(result.world_validation.issues) == 0


def test_world_validator_run_dry_run(state_with_world: GameState) -> None:
    """Test WorldValidatorAgent run in dry run mode."""
    agent = WorldValidatorAgent()

    result = agent.run(state_with_world)

    assert result.world_validation is not None
    assert result.world_validation.is_coherent is True


def test_validation_agent_initialization() -> None:
    """Test GameLogicValidatorAgent initializes correctly."""
    agent = GameLogicValidatorAgent()

    assert agent.llm is not None
    assert agent.response_format is not None


def test_validation_agent_get_system_prompt(state_with_full_game: GameState) -> None:
    """Test GameLogicValidatorAgent get_system_prompt."""
    agent = GameLogicValidatorAgent()

    prompt = agent.get_system_prompt(state_with_full_game)

    assert isinstance(prompt, str)
    assert len(prompt) > 0


def test_validation_agent_mock_output(state_with_full_game: GameState) -> None:
    """Test GameLogicValidatorAgent _mock_output."""
    agent = GameLogicValidatorAgent()

    result = agent._mock_output(state_with_full_game)

    assert result.validation is not None
    assert isinstance(result.validation, ValidationReport)
    assert result.validation.is_consistent is True
    assert len(result.validation.issues) == 0


def test_validation_agent_run_dry_run(state_with_full_game: GameState) -> None:
    """Test GameLogicValidatorAgent run in dry run mode."""
    agent = GameLogicValidatorAgent()

    result = agent.run(state_with_full_game)

    assert result.validation is not None
    assert result.validation.is_consistent is True


def test_world_validator_without_world() -> None:
    """Test WorldValidatorAgent with missing world data."""
    state = GameState(
        meta=MetaInfo(),
        config=GameConfig(
            players=PlayerConfig(total=TEST_DEFAULT_PLAYERS),
            host_gender="male",
            duration_minutes=TEST_DEFAULT_DURATION,
            dry_run=True,
        ),
    )

    agent = WorldValidatorAgent()
    result = agent._mock_output(state)

    # Should still return valid validation in mock mode
    assert result.world_validation is not None
    assert result.world_validation.is_coherent is True


def test_validation_agent_without_full_state() -> None:
    """Test GameLogicValidatorAgent with incomplete state."""
    state = GameState(
        meta=MetaInfo(),
        config=GameConfig(
            players=PlayerConfig(total=TEST_DEFAULT_PLAYERS),
            host_gender="male",
            duration_minutes=TEST_DEFAULT_DURATION,
            dry_run=True,
        ),
    )

    agent = GameLogicValidatorAgent()
    result = agent._mock_output(state)

    # Should still return valid validation in mock mode
    assert result.validation is not None
    assert result.validation.is_consistent is True
