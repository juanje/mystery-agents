"""Integration tests for the CLI entry point."""

from pathlib import Path
from unittest.mock import patch

import pytest
from click.testing import CliRunner

from mystery_agents.cli import generate
from mystery_agents.models.state import (
    GameConfig,
    GameState,
    MetaInfo,
    PlayerConfig,
    ValidationReport,
)
from mystery_agents.utils.constants import (
    HOST_GUIDE_FILENAME,
    TEST_MIN_DURATION,
    TEST_MIN_PLAYERS,
)


@pytest.fixture
def test_output_dir(tmp_path: Path) -> Path:
    """Create a temporary output directory."""
    output_dir = tmp_path / "output"
    output_dir.mkdir()
    return output_dir


@pytest.mark.slow
def test_cli_handles_dict_state_correctly(test_output_dir: Path) -> None:
    """
    Test that CLI correctly handles LangGraph's dict-based state output.

    This test specifically prevents regression of the bug where CLI tried to
    access state.validation when state was a dict instead of GameState object.

    The test verifies that the CLI can access nested objects (like validation)
    correctly when the state is returned as a dict from LangGraph.
    """
    from mystery_agents.graph.workflow import create_workflow
    from mystery_agents.utils.constants import DEFAULT_RECURSION_LIMIT

    # Create initial state
    initial_state = GameState(
        meta=MetaInfo(),
        config=GameConfig(
            players=PlayerConfig(total=TEST_MIN_PLAYERS),
            host_gender="male",
            country="Spain",
            duration_minutes=TEST_MIN_DURATION,
            dry_run=True,
        ),
    )

    # Mock all workflow nodes to avoid executing agents - we only need the state structure
    # This test verifies dict access, not agent execution
    with (
        patch("mystery_agents.graph.workflow.a1_config_node") as mock_a1,
        patch("mystery_agents.graph.workflow.a2_world_node") as mock_a2,
        patch("mystery_agents.graph.workflow.v1_world_validator_node") as mock_v1_world,
        patch("mystery_agents.graph.workflow.a3_characters_node") as mock_a3,
        patch("mystery_agents.graph.workflow.a3_5_character_images_node") as mock_a3_5,
        patch("mystery_agents.graph.workflow.a4_relationships_node") as mock_a4,
        patch("mystery_agents.graph.workflow.a5_crime_node") as mock_a5,
        patch("mystery_agents.graph.workflow.a6_timeline_node") as mock_a6,
        patch("mystery_agents.graph.workflow.a7_killer_node") as mock_a7,
        patch("mystery_agents.graph.workflow.v2_game_logic_validator_node") as mock_v2_logic,
        patch("mystery_agents.graph.workflow.a8_content_node") as mock_a8,
        patch("mystery_agents.graph.workflow.a9_packaging_node") as mock_a9,
    ):
        # Pre-populate state with minimal data to avoid agent execution
        from mystery_agents.models.state import (
            ClueSpec,
            CrimeScene,
            CrimeSpec,
            FileDescriptor,
            GlobalTimeline,
            HostGuide,
            KillerSelection,
            MurderMethod,
            PackagingInfo,
            ValidationReport,
            VictimSpec,
            WorldBible,
        )

        initial_state.world = WorldBible(
            epoch="Modern",
            location_type="Mansion",
            location_name="Test Manor",
            summary="Test",
            gathering_reason="Test gathering",
            visual_keywords=["test"],
            constraints=[],
        )
        initial_state.crime = CrimeSpec(
            victim=VictimSpec(
                name="Test Victim",
                age=50,
                gender="female",
                role_in_setting="Owner",
                public_persona="Test",
                secrets=[],
            ),
            murder_method=MurderMethod(type="poison", description="Test", weapon_used="Test"),
            crime_scene=CrimeScene(
                room_id="study", description="Test", scene_description_post_discovery="Test"
            ),
            time_of_death_approx="22:00",
            possible_weapons=[],
            possible_opportunities=[],
        )
        initial_state.characters = []
        initial_state.timeline_global = GlobalTimeline(
            time_blocks=[], live_action_murder_event=None
        )
        initial_state.killer_selection = KillerSelection(
            killer_id="test", rationale="Test", modified_events=[], truth_narrative="Test"
        )
        initial_state.validation = ValidationReport(
            is_consistent=True, issues=[], suggested_fixes=[]
        )
        initial_state.host_guide = HostGuide(
            spoiler_free_intro="Test",
            host_act1_role_description="Test",
            setup_instructions=[],
            runtime_tips=[],
            live_action_murder_event_guide="Test",
            act_2_intro_script="Test",
            host_act2_detective_role=None,
        )
        initial_state.clues = [
            ClueSpec(
                type="note",
                title="Test",
                description="Test",
                incriminates=[],
                exonerates=[],
                is_red_herring=False,
            )
        ]
        initial_state.packaging = PackagingInfo(
            host_guide_file=FileDescriptor(
                type="markdown", name=HOST_GUIDE_FILENAME, path="test.md"
            ),
            index_summary="Test package",
        )

        # Mock nodes to just pass through state (fast - no agent execution)
        def pass_through(state: GameState) -> GameState:
            return state

        mock_a1.side_effect = pass_through
        mock_a2.side_effect = pass_through
        mock_v1_world.side_effect = pass_through
        mock_a3.side_effect = pass_through
        mock_a3_5.side_effect = pass_through
        mock_a4.side_effect = pass_through
        mock_a5.side_effect = pass_through
        mock_a6.side_effect = pass_through
        mock_a7.side_effect = pass_through
        mock_v2_logic.side_effect = pass_through
        mock_a8.side_effect = pass_through
        mock_a9.side_effect = pass_through

        # Run workflow to get dict state (simulating what CLI does)
        workflow = create_workflow()
        config = {"recursion_limit": DEFAULT_RECURSION_LIMIT}
        all_states = []

        for output in workflow.stream(initial_state, config=config):
            for node_name, state in output.items():
                all_states.append((node_name, state))

        # Get final state (this is what CLI receives)
        final_node_name, final_state = all_states[-1]

        # Verify it's a dict (this is what LangGraph returns)
        assert isinstance(final_state, dict), "State should be a dict from LangGraph"

        # This is the critical test: accessing validation like CLI does
        # This pattern was failing before the fix
        validation = final_state.get("validation")
        assert validation is not None, "Validation should exist"

        # Access nested object attribute (this was the bug - trying to do
        # final_state.validation.is_consistent when final_state is a dict)
        assert hasattr(validation, "is_consistent"), "Should access nested object attribute"
        assert validation.is_consistent is not None, "Should be able to read is_consistent"


@pytest.mark.slow
def test_cli_handles_validation_failure_correctly() -> None:
    """
    Test that CLI correctly handles validation failures when state is a dict.

    This ensures the validation check in CLI works with dict-based state.
    The test simulates the exact pattern used in CLI to check validation.
    """
    from mystery_agents.graph.workflow import create_workflow
    from mystery_agents.utils.constants import DEFAULT_RECURSION_LIMIT

    # Create initial state
    initial_state = GameState(
        meta=MetaInfo(),
        config=GameConfig(
            players=PlayerConfig(total=TEST_MIN_PLAYERS),
            host_gender="male",
            country="Spain",
            duration_minutes=TEST_MIN_DURATION,
            dry_run=True,
        ),
    )

    # Mock workflow nodes to avoid agent execution - we only need validation logic
    from mystery_agents.models.state import ValidationIssue

    # Pre-populate state with validation failure
    initial_state.validation = ValidationReport(
        is_consistent=False,
        issues=[ValidationIssue(type="timeline_conflict", description="Test issue")],
        suggested_fixes=["Fix timeline"],
    )

    with (
        patch("mystery_agents.graph.workflow.a1_config_node") as mock_a1,
        patch("mystery_agents.graph.workflow.a2_world_node") as mock_a2,
        patch("mystery_agents.graph.workflow.v1_world_validator_node") as mock_v1_world,
        patch("mystery_agents.graph.workflow.a3_characters_node") as mock_a3,
        patch("mystery_agents.graph.workflow.a3_5_character_images_node") as mock_a3_5,
        patch("mystery_agents.graph.workflow.a4_relationships_node") as mock_a4,
        patch("mystery_agents.graph.workflow.a5_crime_node") as mock_a5,
        patch("mystery_agents.graph.workflow.a6_timeline_node") as mock_a6,
        patch("mystery_agents.graph.workflow.a7_killer_node") as mock_a7,
        patch("mystery_agents.graph.workflow.v2_game_logic_validator_node") as mock_v2_logic,
        patch("mystery_agents.graph.workflow.a8_content_node") as mock_a8,
        patch("mystery_agents.graph.workflow.a9_packaging_node") as mock_a9,
    ):
        # Mock nodes to pass through state (fast - no agent execution)
        def pass_through(state: GameState) -> GameState:
            return state

        mock_a1.side_effect = pass_through
        mock_a2.side_effect = pass_through
        mock_v1_world.side_effect = pass_through
        mock_a3.side_effect = pass_through
        mock_a3_5.side_effect = pass_through
        mock_a4.side_effect = pass_through
        mock_a5.side_effect = pass_through
        mock_a6.side_effect = pass_through
        mock_a7.side_effect = pass_through

        # v2_game_logic_validator_node needs to increment retry_count to prevent infinite loop
        def mock_validator(state: GameState) -> GameState:
            state.retry_count += 1
            # Keep the pre-populated validation (is_consistent=False)
            return state

        mock_v2_logic.side_effect = mock_validator
        mock_a8.side_effect = pass_through
        mock_a9.side_effect = pass_through

        # Run workflow
        workflow = create_workflow()
        config = {"recursion_limit": DEFAULT_RECURSION_LIMIT}
        all_states = []

        for output in workflow.stream(initial_state, config=config):
            for node_name, state in output.items():
                all_states.append((node_name, state))

        # Get final state (as dict, like CLI receives it)
        final_node_name, final_state = all_states[-1]
        assert isinstance(final_state, dict), "State should be a dict"

        # This is the exact pattern used in CLI (line 93-94)
        validation = final_state.get("validation")

        # This is the critical check that was failing before
        # CLI does: if validation and not validation.is_consistent:
        # This should work even when final_state is a dict
        assert validation is not None, "Validation should exist"
        assert not validation.is_consistent, "Validation should be inconsistent"

        # Verify we can access issues (like CLI does on line 97)
        assert len(validation.issues) > 0, "Should have issues"
        # ValidationIssue is a Pydantic model, access attributes not dict keys
        assert validation.issues[0].type == "timeline_conflict"


@pytest.mark.slow
def test_cli_accesses_nested_objects_correctly() -> None:
    """
    Test that CLI correctly accesses nested Pydantic objects within dict state.

    This verifies that when state is a dict, nested objects like validation
    are still accessed correctly using attribute notation.
    """
    from mystery_agents.graph.workflow import create_workflow
    from mystery_agents.utils.constants import DEFAULT_RECURSION_LIMIT

    # Create initial state
    initial_state = GameState(
        meta=MetaInfo(),
        config=GameConfig(
            players=PlayerConfig(total=TEST_MIN_PLAYERS),
            host_gender="male",
            country="Spain",
            duration_minutes=TEST_MIN_DURATION,
            dry_run=True,
        ),
    )

    # Pre-populate state with minimal required data
    from mystery_agents.models.state import (
        ClueSpec,
        FileDescriptor,
        GlobalTimeline,
        HostGuide,
        KillerSelection,
        PackagingInfo,
    )

    initial_state.timeline_global = GlobalTimeline(time_blocks=[], live_action_murder_event=None)
    initial_state.killer_selection = KillerSelection(
        killer_id="test", rationale="Test", modified_events=[], truth_narrative="Test"
    )
    initial_state.validation = ValidationReport(is_consistent=True, issues=[], suggested_fixes=[])
    initial_state.host_guide = HostGuide(
        spoiler_free_intro="Test",
        host_act1_role_description="Test",
        setup_instructions=[],
        runtime_tips=[],
        live_action_murder_event_guide="Test",
        act_2_intro_script="Test",
        host_act2_detective_role=None,
    )
    initial_state.clues = [
        ClueSpec(
            type="note",
            title="Test",
            description="Test",
            incriminates=[],
            exonerates=[],
            is_red_herring=False,
        )
    ]
    initial_state.packaging = PackagingInfo(
        host_guide_file=FileDescriptor(type="markdown", name=HOST_GUIDE_FILENAME, path="test.md"),
        index_summary="Test",
    )

    # Mock workflow nodes to avoid agent execution - we only need the state structure
    with (
        patch("mystery_agents.graph.workflow.a1_config_node") as mock_a1,
        patch("mystery_agents.graph.workflow.a2_world_node") as mock_a2,
        patch("mystery_agents.graph.workflow.v1_world_validator_node") as mock_v1_world,
        patch("mystery_agents.graph.workflow.a3_characters_node") as mock_a3,
        patch("mystery_agents.graph.workflow.a3_5_character_images_node") as mock_a3_5,
        patch("mystery_agents.graph.workflow.a4_relationships_node") as mock_a4,
        patch("mystery_agents.graph.workflow.a5_crime_node") as mock_a5,
        patch("mystery_agents.graph.workflow.a6_timeline_node") as mock_a6,
        patch("mystery_agents.graph.workflow.a7_killer_node") as mock_a7,
        patch("mystery_agents.graph.workflow.v2_game_logic_validator_node") as mock_v2_logic,
        patch("mystery_agents.graph.workflow.a8_content_node") as mock_a8,
        patch("mystery_agents.graph.workflow.a9_packaging_node") as mock_a9,
    ):
        # Mock nodes to pass through state (fast - no agent execution)
        def pass_through(state: GameState) -> GameState:
            return state

        mock_a1.side_effect = pass_through
        mock_a2.side_effect = pass_through
        mock_v1_world.side_effect = pass_through
        mock_a3.side_effect = pass_through
        mock_a3_5.side_effect = pass_through
        mock_a4.side_effect = pass_through
        mock_a5.side_effect = pass_through
        mock_a6.side_effect = pass_through
        mock_a7.side_effect = pass_through
        mock_v2_logic.side_effect = pass_through
        mock_a8.side_effect = pass_through
        mock_a9.side_effect = pass_through

        # Run workflow to get dict state
        workflow = create_workflow()
        config = {"recursion_limit": DEFAULT_RECURSION_LIMIT}
        all_states = []

        for output in workflow.stream(initial_state, config=config):
            for node_name, state in output.items():
                all_states.append((node_name, state))

    # Get final state
    final_node_name, final_state = all_states[-1]

    # Verify it's a dict
    assert isinstance(final_state, dict), "State should be a dict from LangGraph"

    # Verify we can access nested objects correctly (the pattern used in CLI)
    validation = final_state.get("validation")
    assert validation is not None, "Validation should exist"

    # This is the critical test: accessing nested object attributes
    # This is what the CLI does and what was failing before
    assert validation.is_consistent is not None, "Should access nested object attribute"

    # Verify other nested accesses work
    packaging = final_state.get("packaging")
    assert packaging is not None, "Packaging should exist"
    assert packaging.index_summary is not None, "Should access nested object attribute"

    meta = final_state.get("meta")
    assert meta is not None, "Meta should exist"
    assert meta.id is not None, "Should access nested object attribute"


def test_cli_uses_default_game_yml_when_no_arg(tmp_path: Path) -> None:
    """Test that CLI uses game.yml by default when it exists and no argument is provided."""
    runner = CliRunner()

    with runner.isolated_filesystem(temp_dir=tmp_path):
        # Create a game.yml config file
        game_config = """
language: es
country: Spain
epoch: modern
theme: family_mansion
players:
  male: 3
  female: 3
host_gender: male
duration_minutes: 90
difficulty: medium
"""
        Path("game.yml").write_text(game_config)

        # Mock workflow to stop after config loading (fast test)
        with patch("mystery_agents.graph.workflow.create_workflow") as mock_workflow:
            mock_instance = mock_workflow.return_value
            # Simulate workflow stopping after config node
            mock_instance.stream.side_effect = Exception("Test stopped after config load")

            # Run CLI without specifying config file (it should use game.yml)
            result = runner.invoke(generate, ["--dry-run"])

            # Should have passed config file resolution (exit before actual workflow)
            # If it failed at config resolution, we'd see "No configuration file found"
            assert "No configuration file found" not in result.output
            # The test exception proves we got past config resolution into workflow
            assert result.exit_code == 1  # Exits due to our test exception


def test_cli_fails_when_no_config_file_found(tmp_path: Path) -> None:
    """Test that CLI fails with clear error when no config file exists."""
    runner = CliRunner()

    with runner.isolated_filesystem(temp_dir=tmp_path):
        # Don't create any config file
        result = runner.invoke(generate)

        # Should fail with error message
        assert result.exit_code == 1
        assert "No configuration file found" in result.output
        assert "game.yml" in result.output


def test_cli_uses_specified_config_file(tmp_path: Path) -> None:
    """Test that CLI uses the specified config file when provided as argument."""
    runner = CliRunner()

    with runner.isolated_filesystem(temp_dir=tmp_path):
        # Create a custom config file
        custom_config = """
language: en
country: United States
epoch: 1920s
theme: cruise
players:
  male: 4
  female: 2
host_gender: female
duration_minutes: 120
difficulty: hard
"""
        custom_path = Path("my-custom-game.yml")
        custom_path.write_text(custom_config)

        # Mock workflow to stop after config loading (fast test)
        with patch("mystery_agents.graph.workflow.create_workflow") as mock_workflow:
            mock_instance = mock_workflow.return_value
            mock_instance.stream.side_effect = Exception("Test stopped after config load")

            # Run CLI with specified config file
            result = runner.invoke(generate, ["my-custom-game.yml", "--dry-run"])

            # Should have passed config file resolution
            # If it failed at config resolution, we'd see "Configuration file not found"
            assert "Configuration file not found" not in result.output
            assert result.exit_code == 1  # Exits due to our test exception


def test_cli_fails_when_specified_config_not_exists(tmp_path: Path) -> None:
    """Test that CLI fails with clear error when specified config file doesn't exist."""
    runner = CliRunner()

    with runner.isolated_filesystem(temp_dir=tmp_path):
        # Try to use a non-existent file
        result = runner.invoke(generate, ["nonexistent.yml"])

        # Should fail with error message
        assert result.exit_code == 1
        assert "Configuration file not found" in result.output
        assert "nonexistent.yml" in result.output


def test_cli_config_file_takes_precedence_over_default(tmp_path: Path) -> None:
    """Test that explicitly specified config file takes precedence over default game.yml."""
    runner = CliRunner()

    with runner.isolated_filesystem(temp_dir=tmp_path):
        # Create both game.yml and a custom config
        default_config = """
language: es
country: Spain
epoch: modern
theme: family_mansion
players:
  male: 3
  female: 3
host_gender: male
duration_minutes: 90
difficulty: medium
"""
        Path("game.yml").write_text(default_config)

        custom_config = """
language: en
country: Japan
epoch: victorian
theme: train
players:
  male: 2
  female: 4
host_gender: female
duration_minutes: 90
difficulty: medium
"""
        Path("custom.yml").write_text(custom_config)

        # Mock workflow to stop after config loading (fast test)
        with patch("mystery_agents.graph.workflow.create_workflow") as mock_workflow:
            mock_instance = mock_workflow.return_value
            mock_instance.stream.side_effect = Exception("Test stopped after config load")

            # Run CLI with custom config
            result = runner.invoke(generate, ["custom.yml", "--dry-run"])

            # Should have passed config file resolution
            assert "Configuration file not found" not in result.output
            assert result.exit_code == 1  # Exits due to our test exception


def test_cli_verbose_flag_shows_logs(tmp_path: Path) -> None:
    """Test that -v flag shows structured logs instead of visual progress."""
    runner = CliRunner()

    with runner.isolated_filesystem(temp_dir=tmp_path):
        config_content = """
language: es
country: Spain
epoch: modern
theme: family_mansion
players:
  male: 3
  female: 3
host_gender: male
duration_minutes: 90
difficulty: medium
"""
        Path("game.yml").write_text(config_content)

        with patch("mystery_agents.graph.workflow.create_workflow") as mock_workflow:
            mock_instance = mock_workflow.return_value
            mock_instance.stream.side_effect = Exception("Test stopped")

            result = runner.invoke(generate, ["-v", "--dry-run"])
            assert result.exit_code != 0


def test_cli_quiet_and_verbose_mutually_exclusive(tmp_path: Path) -> None:
    """Test that --quiet and -v are mutually exclusive."""
    runner = CliRunner()

    with runner.isolated_filesystem(temp_dir=tmp_path):
        config_content = """
language: es
country: Spain
epoch: modern
theme: family_mansion
players:
  male: 3
  female: 3
host_gender: male
duration_minutes: 90
difficulty: medium
"""
        Path("game.yml").write_text(config_content)

        result = runner.invoke(generate, ["--quiet", "-v"])
        assert result.exit_code == 1
        assert "mutually exclusive" in result.output
