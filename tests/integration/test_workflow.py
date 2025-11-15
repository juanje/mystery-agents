"""Integration tests for the full workflow."""

from pathlib import Path
from unittest.mock import patch

import pytest

from mystery_agents.graph.workflow import create_workflow
from mystery_agents.models.state import GameConfig, GameState, MetaInfo, PlayerConfig
from mystery_agents.utils.constants import (
    DEFAULT_RECURSION_LIMIT,
    HOST_GUIDE_FILENAME,
    TEST_DEFAULT_DURATION,
    TEST_MIN_DURATION,
    TEST_MIN_PLAYERS,
)


@pytest.fixture
def test_output_dir(tmp_path: Path) -> Path:
    """Create a temporary output directory."""
    output_dir = tmp_path / "output"
    output_dir.mkdir()
    return output_dir


def test_workflow_dry_run(test_output_dir: Path) -> None:
    """Test the complete workflow in dry run mode (fast, no LLMs)."""
    # Mock all CLI prompts to run non-interactively (dry_run is now passed from CLI, not asked interactively)
    with (
        patch("mystery_agents.agents.a1_config.click.prompt") as mock_prompt,
        patch("mystery_agents.agents.a9_packaging.PackagingAgent.run") as mock_package,
    ):
        # Configure mocks for A1 config wizard
        mock_prompt.side_effect = [
            "es",  # language
            "Spain",  # country
            "",  # region (optional)
            1,  # epoch (modern)
            1,  # theme (family_mansion)
            1,  # tone (cluedo_like)
            TEST_MIN_PLAYERS,  # total players (minimum)
            "male",  # host gender
            0,  # male
            0,  # female
            TEST_MIN_DURATION,  # duration (minimum)
            1,  # difficulty (easy)
        ]

        # Mock packaging to avoid file I/O in test - tests logic, not I/O
        def mock_packaging_run(state: GameState, output_dir: str = "./output") -> GameState:
            from mystery_agents.models.state import FileDescriptor, PackagingInfo

            state.packaging = PackagingInfo(
                host_guide_file=FileDescriptor(
                    type="markdown",
                    name=HOST_GUIDE_FILENAME,
                    path=str(test_output_dir / HOST_GUIDE_FILENAME),
                ),
                index_summary="Test game package",
            )
            return state

        mock_package.side_effect = mock_packaging_run

        # Create initial state
        initial_state = GameState(
            meta=MetaInfo(),
            config=GameConfig(
                players=PlayerConfig(total=TEST_MIN_PLAYERS),
                host_gender="male",
                country="Spain",
                duration_minutes=TEST_DEFAULT_DURATION,
                dry_run=True,
                debug_model=False,
            ),
        )

        # Create and run workflow
        workflow = create_workflow()

        # Execute workflow with increased recursion limit for validation retries
        config = {"recursion_limit": DEFAULT_RECURSION_LIMIT}
        all_states = []
        for output in workflow.stream(initial_state, config=config):
            # Each output is a dict like {"node_name": state}
            for node_name, state in output.items():
                all_states.append((node_name, state))

        # Verify we got outputs
        assert len(all_states) > 0, "Workflow produced no outputs"

        # Get the final state from the last node (should be a9_packaging)
        final_node_name, final_state = all_states[-1]

        # LangGraph returns state as dict (serialized Pydantic model)
        assert isinstance(final_state, dict), f"Expected dict, got {type(final_state)}"

        # Verify state completeness (state is dict, nested objects are Pydantic)
        assert final_state["config"] is not None
        assert final_state["world"] is not None
        assert final_state["crime"] is not None
        assert len(final_state["characters"]) == TEST_MIN_PLAYERS  # Should match config
        assert final_state["timeline_global"] is not None
        assert final_state["killer_selection"] is not None
        assert final_state["validation"] is not None
        assert (
            final_state["validation"].is_consistent is True
        )  # Nested object uses attribute access
        assert final_state["host_guide"] is not None
        assert final_state["audio_script"] is not None
        assert len(final_state["clues"]) > 0
        assert final_state["packaging"] is not None


def test_validation_retry_loop() -> None:
    """Test that the validation retry loop works correctly."""
    from mystery_agents.graph.workflow import should_retry_validation
    from mystery_agents.models.state import ValidationReport

    # Note: retry_count is incremented in v1_validator_node, not in should_retry_validation
    # This test verifies that should_retry_validation correctly routes based on retry_count

    # Scenario 1: First attempt fails, can retry (retry_count will be 1 after validator node)
    state = GameState(
        meta=MetaInfo(),
        config=GameConfig(
            players=PlayerConfig(total=TEST_MIN_PLAYERS),
            host_gender="male",
            country="Spain",
            duration_minutes=TEST_DEFAULT_DURATION,
        ),
    )
    state.validation = ValidationReport(is_consistent=False, issues=[], suggested_fixes=[])
    state.retry_count = 0  # Before validator node increments it

    result = should_retry_validation(state)
    assert result == "retry", "Should allow retry when retry_count < max_retries"

    # Scenario 2: After validator node increments, still can retry
    state.retry_count = 1  # After first validator run
    state.validation = ValidationReport(is_consistent=False, issues=[], suggested_fixes=[])
    result = should_retry_validation(state)
    assert result == "retry", "Should allow retry when retry_count < max_retries"

    # Scenario 3: After second validator run
    state.retry_count = 2
    state.validation = ValidationReport(is_consistent=False, issues=[], suggested_fixes=[])
    result = should_retry_validation(state)
    assert result == "retry", "Should allow retry when retry_count < max_retries"

    # Scenario 4: Max retries reached, should fail
    state.retry_count = 3  # At max_retries
    state.validation = ValidationReport(is_consistent=False, issues=[], suggested_fixes=[])
    result = should_retry_validation(state)
    assert result == "fail", "Should fail when retry_count >= max_retries"
    assert state.retry_count == 3  # Doesn't increment past max

    # Scenario 5: Validation passes (regardless of retry_count)
    state.validation = ValidationReport(is_consistent=True, issues=[], suggested_fixes=[])
    result = should_retry_validation(state)
    assert result == "pass", "Should pass when validation is consistent"


def test_workflow_structure() -> None:
    """Test that the workflow has the correct structure."""
    workflow = create_workflow()

    # Verify workflow was compiled successfully
    assert workflow is not None

    # The workflow should have our agents as nodes
    # LangGraph's compiled graph doesn't expose node names easily,
    # but we can verify it compiled without errors
    assert hasattr(workflow, "stream")
    assert hasattr(workflow, "invoke")
