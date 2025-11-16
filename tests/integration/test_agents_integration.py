"""Integration tests for agents (test full agent initialization and execution)."""

from pathlib import Path
from unittest.mock import patch

import pytest

from mystery_agents.agents.a1_config import ConfigWizardAgent
from mystery_agents.agents.a2_world import WorldAgent
from mystery_agents.agents.a5_crime import CrimeAgent
from mystery_agents.agents.a7_killer_selection import KillerSelectionAgent
from mystery_agents.agents.v2_game_logic_validator import GameLogicValidatorAgent
from mystery_agents.models.state import (
    CrimeSpec,
    GameConfig,
    GameState,
    GlobalTimeline,
    MetaInfo,
    PlayerConfig,
    ValidationReport,
    WorldBible,
)
from mystery_agents.utils.constants import (
    HOST_GUIDE_FILENAME,
    TEST_DEFAULT_DURATION,
    TEST_DEFAULT_PLAYERS,
)


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
            dry_run=True,  # Use dry run to avoid LLM calls
            debug_model=False,
        ),
    )


@pytest.fixture
def state_with_world(basic_state: GameState) -> GameState:
    """Create a state with world."""
    agent = WorldAgent()
    return agent.run(basic_state)


@pytest.fixture
def state_with_crime(state_with_characters: GameState) -> GameState:
    """Create a state with crime."""
    agent = CrimeAgent()
    return agent.run(state_with_characters)


@pytest.fixture
def state_with_characters(state_with_world: GameState) -> GameState:
    """Create a state with characters."""
    from mystery_agents.agents.a3_characters import CharactersAgent

    agent = CharactersAgent()
    return agent.run(state_with_world)


@pytest.mark.slow
def test_config_wizard_dry_run(basic_state: GameState) -> None:
    """Test config wizard in dry run mode."""
    # Mock click prompts (dry_run is now passed from CLI, not asked interactively)
    with patch("mystery_agents.agents.a1_config.click.prompt") as mock_prompt:
        mock_prompt.side_effect = [
            "es",  # language
            "Spain",  # country
            "",  # region (optional, empty string)
            1,  # epoch
            1,  # theme
            # tone is now fixed (no prompt)
            3,  # male characters
            3,  # female characters (total = 6 = TEST_DEFAULT_PLAYERS)
            "male",  # host gender
            TEST_DEFAULT_DURATION,  # duration
            2,  # difficulty
        ]

        agent = ConfigWizardAgent()
        result = agent.run(basic_state)

        assert result.config.language == "es"
        assert result.config.players.total == TEST_DEFAULT_PLAYERS
        assert result.config.players.male == 3
        assert result.config.players.female == 3
        assert result.config.dry_run is True  # Preserved from initial state


@pytest.mark.slow
def test_world_agent_dry_run(basic_state: GameState) -> None:
    """Test world agent in dry run mode."""
    agent = WorldAgent()
    result = agent.run(basic_state)

    # Should generate mock world
    assert result.world is not None
    assert isinstance(result.world, WorldBible)
    assert result.world.location_name != ""


@pytest.mark.slow
def test_crime_agent_dry_run(state_with_characters: GameState) -> None:
    """Test crime agent in dry run mode."""
    agent = CrimeAgent()
    result = agent.run(state_with_characters)

    # Should generate mock crime
    assert result.crime is not None
    assert isinstance(result.crime, CrimeSpec)
    assert result.crime.victim.name != ""


@pytest.mark.slow
def test_killer_selection_dry_run(state_with_crime: GameState) -> None:
    """Test killer selection agent in dry run mode."""
    # Add timeline for completeness
    from mystery_agents.agents.a6_timeline import TimelineAgent

    timeline_agent = TimelineAgent()
    state_with_timeline = timeline_agent.run(state_with_crime)

    # Select killer
    agent = KillerSelectionAgent()
    result = agent.run(state_with_timeline)

    assert result.killer_selection is not None
    assert result.killer_selection.killer_id != ""

    # Verify killer is in characters list
    killer_ids = [c.id for c in result.characters]
    assert result.killer_selection.killer_id in killer_ids


@pytest.mark.slow
def test_validation_agent_dry_run(state_with_crime: GameState) -> None:
    """Test validation agent in dry run mode."""
    agent = GameLogicValidatorAgent()
    result = agent.run(state_with_crime)

    assert result.validation is not None
    assert isinstance(result.validation, ValidationReport)

    # In dry run mode, validation always passes
    assert result.validation.is_consistent is True


@pytest.mark.slow
def test_characters_generation(state_with_world: GameState) -> None:
    """Test characters generation."""
    from mystery_agents.agents.a3_characters import CharactersAgent

    agent = CharactersAgent()
    result = agent.run(state_with_world)

    # Should generate correct number of characters
    assert len(result.characters) == state_with_world.config.players.total

    # Each character should have required fields
    for char in result.characters:
        assert char.name != ""
        assert char.role != ""
        assert char.relation_to_victim != ""
        assert char.motive_for_crime is not None
        assert char.costume_suggestion is not None  # MVP requirement


@pytest.mark.slow
def test_timeline_generation(state_with_crime: GameState) -> None:
    """Test timeline generation."""
    from mystery_agents.agents.a6_timeline import TimelineAgent

    agent = TimelineAgent()
    result = agent.run(state_with_crime)

    assert result.timeline_global is not None
    assert isinstance(result.timeline_global, GlobalTimeline)
    assert len(result.timeline_global.time_blocks) > 0


@pytest.mark.slow
def test_content_generation(state_with_crime: GameState) -> None:
    """Test content generation."""
    from mystery_agents.agents.a6_timeline import TimelineAgent
    from mystery_agents.agents.a8_content import ContentGenerationAgent

    # Need timeline and killer selection
    timeline_agent = TimelineAgent()
    state_with_timeline = timeline_agent.run(state_with_crime)

    killer_agent = KillerSelectionAgent()
    state_with_killer = killer_agent.run(state_with_timeline)

    # Generate content
    agent = ContentGenerationAgent()
    result = agent.run(state_with_killer)

    assert result.host_guide is not None
    assert result.audio_script is not None
    assert len(result.clues) > 0


@pytest.mark.slow
def test_packaging(state_with_crime: GameState, tmp_path: Path) -> None:
    """Test packaging agent logic without actual file I/O."""
    from mystery_agents.agents.a6_timeline import TimelineAgent
    from mystery_agents.agents.a8_content import ContentGenerationAgent
    from mystery_agents.agents.a9_packaging import PackagingAgent

    # Need full pipeline
    timeline_agent = TimelineAgent()
    state_with_timeline = timeline_agent.run(state_with_crime)

    killer_agent = KillerSelectionAgent()
    state_with_killer = killer_agent.run(state_with_timeline)

    content_agent = ContentGenerationAgent()
    state_with_content = content_agent.run(state_with_killer)

    # Mock the packaging agent's internal write methods to avoid I/O
    test_output_dir = str(tmp_path / "test_output")
    agent = PackagingAgent()

    # Mock all write methods and directory creation to avoid disk I/O
    # Using _ prefix for intentionally unused mocks (they prevent real I/O)
    with (
        patch.object(agent, "_write_host_guide") as mock_write_host,
        patch.object(agent, "_write_audio_script") as mock_write_audio,
        patch.object(agent, "_write_solution") as _mock_write_solution,
        patch.object(agent, "_write_character_sheet") as _mock_write_char,
        patch.object(agent, "_write_invitation") as _mock_write_inv,
        patch.object(agent, "_write_clue_clean") as _mock_write_clue_clean,
        patch.object(agent, "_write_clue_reference") as _mock_write_clue_ref,
        patch.object(agent, "_write_readme") as _mock_write_readme,
        patch.object(agent, "_create_zip") as _mock_create_zip,
        patch.object(agent, "_generate_all_pdfs") as _mock_generate_pdfs,
        patch("pathlib.Path.mkdir") as _mock_mkdir,  # Mock directory creation
        patch("pathlib.Path.write_text") as _mock_write_text,  # Mock file writing
    ):
        result = agent.run(state_with_content, output_dir=test_output_dir)

        # Verify packaging logic (not file I/O)
        assert result.packaging is not None
        assert result.packaging.host_guide_file is not None
        assert result.packaging.host_guide_file.name == HOST_GUIDE_FILENAME
        assert len(result.packaging.individual_player_packages) > 0
        assert len(result.packaging.individual_player_packages) == len(
            state_with_content.characters
        )

        # Verify mocks were called (but no actual I/O happened)
        if state_with_content.host_guide:
            mock_write_host.assert_called_once()
        if state_with_content.audio_script:
            mock_write_audio.assert_called_once()
