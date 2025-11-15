"""Unit tests for workflow functions and routing logic."""

from unittest.mock import patch

import pytest

from mystery_agents.graph.workflow import (
    should_retry_validation,
    should_retry_world_validation,
)
from mystery_agents.models.state import (
    GameConfig,
    GameState,
    MetaInfo,
    PlayerConfig,
    ValidationReport,
    WorldValidation,
)
from mystery_agents.utils.constants import TEST_DEFAULT_DURATION, TEST_DEFAULT_PLAYERS


@pytest.fixture(autouse=True)
def mock_google_api_key(monkeypatch: pytest.MonkeyPatch) -> None:
    """Mock GOOGLE_API_KEY for all tests."""
    monkeypatch.setenv("GOOGLE_API_KEY", "test-api-key-for-testing")


@pytest.fixture
def basic_state() -> GameState:
    """Create a basic game state for testing."""
    return GameState(
        meta=MetaInfo(),
        config=GameConfig(
            players=PlayerConfig(total=TEST_DEFAULT_PLAYERS),
            host_gender="male",
            duration_minutes=TEST_DEFAULT_DURATION,
            dry_run=True,
        ),
    )


def test_should_retry_validation_passes(basic_state: GameState) -> None:
    """Test should_retry_validation when validation passes."""
    basic_state.validation = ValidationReport(is_consistent=True, issues=[], suggested_fixes=[])

    result = should_retry_validation(basic_state)
    assert result == "pass"


def test_should_retry_validation_retry_available(basic_state: GameState) -> None:
    """Test should_retry_validation when retry is available."""
    basic_state.validation = ValidationReport(is_consistent=False, issues=[], suggested_fixes=[])
    basic_state.retry_count = 0

    result = should_retry_validation(basic_state)
    assert result == "retry"


def test_should_retry_validation_max_retries_exceeded(basic_state: GameState) -> None:
    """Test should_retry_validation when max retries exceeded."""
    basic_state.validation = ValidationReport(is_consistent=False, issues=[], suggested_fixes=[])
    basic_state.retry_count = 3  # At max

    result = should_retry_validation(basic_state)
    assert result == "fail"


def test_should_retry_validation_no_validation(basic_state: GameState) -> None:
    """Test should_retry_validation when validation is None."""
    basic_state.validation = None

    result = should_retry_validation(basic_state)
    assert result == "fail"


def test_should_retry_world_validation_passes(basic_state: GameState) -> None:
    """Test should_retry_world_validation when validation passes."""
    basic_state.world_validation = WorldValidation(is_coherent=True, issues=[])

    result = should_retry_world_validation(basic_state)
    assert result == "pass"


def test_should_retry_world_validation_retry_available(basic_state: GameState) -> None:
    """Test should_retry_world_validation when retry is available."""
    basic_state.world_validation = WorldValidation(is_coherent=False, issues=["Test issue"])
    basic_state.world_retry_count = 0

    result = should_retry_world_validation(basic_state)
    assert result == "retry"


def test_should_retry_world_validation_max_retries_exceeded(basic_state: GameState) -> None:
    """Test should_retry_world_validation when max retries exceeded."""
    basic_state.world_validation = WorldValidation(is_coherent=False, issues=["Test issue"])
    basic_state.world_retry_count = 2  # At max (max_world_retries is 2)

    result = should_retry_world_validation(basic_state)
    assert result == "fail"


def test_should_retry_world_validation_no_validation(basic_state: GameState) -> None:
    """Test should_retry_world_validation when validation is None."""
    basic_state.world_validation = None

    result = should_retry_world_validation(basic_state)
    assert result == "fail"


def test_node_functions_dry_run(basic_state: GameState) -> None:
    """Test that node functions work in dry run mode."""
    from mystery_agents.graph.workflow import (
        a2_world_node,
        a3_characters_node,
        a4_relationships_node,
        a5_crime_node,
        a6_timeline_node,
        a7_killer_node,
    )

    # Test world node
    state = a2_world_node(basic_state)
    assert state.world is not None

    # Test characters node
    state = a3_characters_node(state)
    assert len(state.characters) > 0

    # Test relationships node
    state = a4_relationships_node(state)
    assert len(state.relationships) >= 0

    # Test crime node
    state = a5_crime_node(state)
    assert state.crime is not None

    # Test timeline node
    state = a6_timeline_node(state)
    assert state.timeline_global is not None

    # Test killer selection node
    state = a7_killer_node(state)
    assert state.killer_selection is not None


def test_validation_node_with_click_output(basic_state: GameState) -> None:
    """Test validation node outputs correct click messages."""
    from mystery_agents.agents.a2_world import WorldAgent
    from mystery_agents.graph.workflow import v1_validator_node, v2_world_validator_node

    # Prepare state with world for world validation
    world_agent = WorldAgent()
    state_with_world = world_agent.run(basic_state)

    # Test world validation node with click output
    with patch("mystery_agents.graph.workflow.click.echo") as mock_echo:
        # Reset retry count before test
        state_with_world.world_retry_count = 0
        result = v2_world_validator_node(state_with_world)

        # Should have printed validation message
        assert mock_echo.call_count >= 1
        # In dry run mode, validation passes so retry_count gets reset to 0
        assert result.world_retry_count == 0  # Reset after successful validation

    # Test full validation node with click output
    from mystery_agents.agents.a3_characters import CharactersAgent
    from mystery_agents.agents.a4_relationships import RelationshipsAgent
    from mystery_agents.agents.a5_crime import CrimeAgent
    from mystery_agents.agents.a6_timeline import TimelineAgent
    from mystery_agents.agents.a7_killer_selection import KillerSelectionAgent

    # Build full state
    chars_agent = CharactersAgent()
    state_with_chars = chars_agent.run(state_with_world)

    rel_agent = RelationshipsAgent()
    state_with_rels = rel_agent.run(state_with_chars)

    crime_agent = CrimeAgent()
    state_with_crime = crime_agent.run(state_with_rels)

    timeline_agent = TimelineAgent()
    state_with_timeline = timeline_agent.run(state_with_crime)

    killer_agent = KillerSelectionAgent()
    state_with_killer = killer_agent.run(state_with_timeline)

    with patch("mystery_agents.graph.workflow.click.echo") as mock_echo:
        result = v1_validator_node(state_with_killer)

        # Should have printed validation message
        assert mock_echo.call_count >= 1
        # In dry run mode, validation passes so retry_count gets reset to 0
        assert result.retry_count == 0  # Reset after successful validation
