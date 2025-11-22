"""Unit tests for KillerSelectionAgent (A7)."""

import pytest

from mystery_agents.agents.a2_world import WorldAgent
from mystery_agents.agents.a3_characters import CharactersAgent
from mystery_agents.agents.a4_relationships import RelationshipsAgent
from mystery_agents.agents.a5_crime import CrimeAgent
from mystery_agents.agents.a6_timeline import TimelineAgent
from mystery_agents.agents.a7_killer_selection import KillerSelectionAgent
from mystery_agents.models.state import GameConfig, GameState, MetaInfo, PlayerConfig
from mystery_agents.utils.constants import TEST_DEFAULT_DURATION, TEST_DEFAULT_PLAYERS


@pytest.fixture
def state_with_timeline() -> GameState:
    """Create a state with full game setup including timeline."""
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

    return state


def test_killer_selection_agent_initialization() -> None:
    """Test KillerSelectionAgent initializes correctly."""
    agent = KillerSelectionAgent()

    assert agent.llm is not None
    assert agent.response_format is not None


def test_killer_selection_agent_get_system_prompt(state_with_timeline: GameState) -> None:
    """Test get_system_prompt returns non-empty string."""
    agent = KillerSelectionAgent()

    prompt = agent.get_system_prompt(state_with_timeline)

    assert isinstance(prompt, str)
    assert len(prompt) > 0


def test_killer_selection_agent_mock_output(state_with_timeline: GameState) -> None:
    """Test _mock_output generates killer selection."""
    agent = KillerSelectionAgent()

    result = agent._mock_output(state_with_timeline)

    assert result.killer_selection is not None
    assert result.killer_selection.killer_id == state_with_timeline.characters[0].id
    assert result.killer_selection.rationale != ""
    assert len(result.killer_selection.modified_events) > 0
    assert result.killer_selection.truth_narrative != ""


def test_killer_selection_agent_mock_output_with_no_characters() -> None:
    """Test _mock_output handles case with no characters."""
    state = GameState(
        meta=MetaInfo(),
        config=GameConfig(
            players=PlayerConfig(total=TEST_DEFAULT_PLAYERS),
            host_gender="male",
            duration_minutes=TEST_DEFAULT_DURATION,
            dry_run=True,
        ),
    )

    agent = KillerSelectionAgent()
    result = agent._mock_output(state)

    assert result.killer_selection is not None
    assert result.killer_selection.killer_id == "mock-killer"


def test_killer_selection_agent_run_dry_run(state_with_timeline: GameState) -> None:
    """Test run method in dry run mode."""
    agent = KillerSelectionAgent()

    result = agent.run(state_with_timeline)

    assert result.killer_selection is not None
    assert result.killer_selection.killer_id != ""


def test_killer_selection_agent_run_validates_crime_exists() -> None:
    """Test run method validates that crime exists."""
    state = GameState(
        meta=MetaInfo(),
        config=GameConfig(
            players=PlayerConfig(total=TEST_DEFAULT_PLAYERS),
            host_gender="male",
            duration_minutes=TEST_DEFAULT_DURATION,
            dry_run=True,  # Use dry run to avoid API calls
        ),
    )

    # Set up characters but no crime
    world_agent = WorldAgent()
    state = world_agent.run(state)

    chars_agent = CharactersAgent()
    state = chars_agent.run(state)

    # Manually remove crime to test validation
    state.crime = None
    # Disable dry run to trigger the validation path
    state.config.dry_run = False

    # Try to run killer selection without crime
    agent = KillerSelectionAgent()

    with pytest.raises(ValueError, match="Crime specification is required"):
        agent.run(state)


def test_killer_selection_formats_timeline_correctly(state_with_timeline: GameState) -> None:
    """Test that timeline is formatted correctly in the prompt."""
    agent = KillerSelectionAgent()

    # This test verifies the internal logic by checking run() works
    result = agent.run(state_with_timeline)

    # Should have processed timeline and selected killer
    assert result.killer_selection is not None
    assert result.killer_selection.killer_id in [c.id for c in state_with_timeline.characters]


def test_killer_selection_validates_killer_id(state_with_timeline: GameState) -> None:
    """Test that killer ID is validated against character list."""
    # This is tested implicitly in run() which validates the killer_id
    # is in the characters list and falls back to first character if not

    agent = KillerSelectionAgent()
    result = agent.run(state_with_timeline)

    # Killer must be one of the characters
    killer_ids = [c.id for c in state_with_timeline.characters]
    assert result.killer_selection is not None
    assert result.killer_selection.killer_id in killer_ids
