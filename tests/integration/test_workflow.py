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


@pytest.mark.slow
def test_workflow_dry_run(test_output_dir: Path, tmp_path: Path) -> None:
    """Test the complete workflow in dry run mode (fast, no LLMs)."""
    import yaml

    # Create a temporary YAML config file
    config_data = {
        "language": "es",
        "country": "Spain",
        "epoch": "modern",
        "theme": "family_mansion",
        "players": {"male": 2, "female": 2},
        "host_gender": "male",
        "duration_minutes": TEST_MIN_DURATION,
        "difficulty": "easy",
    }
    config_file = tmp_path / "test_config.yml"
    with open(config_file, "w", encoding="utf-8") as f:
        yaml.dump(config_data, f)

    # Mock packaging to avoid file I/O in test - tests logic, not I/O
    with patch("mystery_agents.agents.a9_packaging.PackagingAgent.run") as mock_package:

        def mock_packaging_run(state: GameState, output_dir: str = "./output") -> GameState:
            from mystery_agents.models.state import FileDescriptor, PackagingInfo

            state.packaging = PackagingInfo(
                host_guide_file=FileDescriptor(
                    type="markdown",
                    name=HOST_GUIDE_FILENAME,
                    path=str(test_output_dir / HOST_GUIDE_FILENAME),
                ),
            )
            return state

        mock_package.side_effect = mock_packaging_run

        # Create initial state with config_file set
        initial_state = GameState(
            meta=MetaInfo(),
            config=GameConfig(
                players=PlayerConfig(total=TEST_MIN_PLAYERS),
                host_gender="male",
                country="Spain",
                duration_minutes=TEST_DEFAULT_DURATION,
                dry_run=True,
                debug_model=False,
                config_file=str(config_file),  # Add config_file path
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
    assert final_state["validation"].is_consistent is True  # Nested object uses attribute access
    assert final_state["host_guide"] is not None
    assert len(final_state["clues"]) > 0
    assert final_state["packaging"] is not None


def test_validation_retry_loop() -> None:
    """Test that the validation retry loop works correctly."""
    from mystery_agents.graph.workflow import should_retry_validation
    from mystery_agents.models.state import ValidationReport

    # Note: retry_count is incremented in v2_game_logic_validator_node, not in should_retry_validation
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


def test_node_retried_on_structured_output_error(tmp_path: Path) -> None:
    """A node whose LLM output doesn't fit the schema is retried by the graph."""
    import yaml

    from mystery_agents.agents.a2_world import WorldAgent
    from mystery_agents.agents.base import StructuredOutputError

    config_file = tmp_path / "game.yml"
    config_file.write_text(
        yaml.dump(
            {
                "language": "en",
                "country": "United States",
                "epoch": "modern",
                "theme": "family_mansion",
                "players": {"male": 2, "female": 2},
                "host_gender": "male",
            }
        )
    )
    state = GameState(
        meta=MetaInfo(),
        config=GameConfig(
            players=PlayerConfig(total=TEST_MIN_PLAYERS),
            duration_minutes=TEST_MIN_DURATION,
            dry_run=True,
            generate_images=False,
            config_file=str(config_file),
            output_dir=str(tmp_path / "output"),
        ),
    )

    original_run = WorldAgent.run
    calls = 0

    def flaky_run(self: WorldAgent, state: GameState) -> GameState:
        nonlocal calls
        calls += 1
        if calls == 1:
            raise StructuredOutputError("bad sample")
        return original_run(self, state)

    with (
        patch.object(WorldAgent, "run", flaky_run),
        patch(
            "mystery_agents.agents.a9_packaging.PackagingAgent.run", side_effect=lambda s, **_: s
        ),
    ):
        final = create_workflow().invoke(state, config={"recursion_limit": DEFAULT_RECURSION_LIMIT})

    assert calls == 2
    assert final["world"] is not None


def test_character_images_reach_packaging(tmp_path: Path) -> None:
    """Portraits from the parallel image branch are attached before packaging."""
    import yaml

    config_file = tmp_path / "game.yml"
    config_file.write_text(
        yaml.dump(
            {
                "language": "en",
                "country": "United States",
                "epoch": "modern",
                "theme": "family_mansion",
                "players": {"male": 2, "female": 2},
                "host_gender": "male",
            }
        )
    )
    state = GameState(
        meta=MetaInfo(),
        config=GameConfig(
            players=PlayerConfig(total=TEST_MIN_PLAYERS),
            duration_minutes=TEST_MIN_DURATION,
            dry_run=True,
            generate_images=True,
            config_file=str(config_file),
            output_dir=str(tmp_path / "output"),
        ),
    )
    packaged: list[GameState] = []

    def capture_packaging(s: GameState, **_: object) -> GameState:
        packaged.append(s)
        return s

    with patch(
        "mystery_agents.agents.a9_packaging.PackagingAgent.run", side_effect=capture_packaging
    ):
        create_workflow().invoke(state, config={"recursion_limit": DEFAULT_RECURSION_LIMIT})

    assert len(packaged) == 1
    characters = packaged[0].characters
    assert characters
    assert all(char.image_path for char in characters)


def test_content_rewritten_when_review_fails(tmp_path: Path) -> None:
    """A failed content review sends the game back to A8 before packaging."""
    import yaml

    from mystery_agents.agents.a8_content import ContentGenerationAgent
    from mystery_agents.agents.v3_content_validator import ContentValidatorAgent
    from mystery_agents.models.state import ValidationIssue, ValidationReport

    config_file = tmp_path / "game.yml"
    config_file.write_text(
        yaml.dump(
            {
                "language": "en",
                "country": "United States",
                "epoch": "modern",
                "theme": "family_mansion",
                "players": {"male": 2, "female": 2},
                "host_gender": "male",
            }
        )
    )
    state = GameState(
        meta=MetaInfo(),
        config=GameConfig(
            players=PlayerConfig(total=TEST_MIN_PLAYERS),
            duration_minutes=TEST_MIN_DURATION,
            dry_run=True,
            generate_images=False,
            config_file=str(config_file),
            output_dir=str(tmp_path / "output"),
        ),
    )

    original_content_run = ContentGenerationAgent.run
    content_calls = 0
    reviews = 0

    def counting_content_run(self: ContentGenerationAgent, s: GameState) -> GameState:
        nonlocal content_calls
        content_calls += 1
        return original_content_run(self, s)

    def review_failing_once(self: ContentValidatorAgent, s: GameState) -> GameState:
        nonlocal reviews
        reviews += 1
        failed = reviews == 1
        s.content_validation = ValidationReport(
            is_consistent=not failed,
            issues=[ValidationIssue(type="contradiction", description="Times differ")]
            if failed
            else [],
            suggested_fixes=[],
        )
        return s

    with (
        patch.object(ContentGenerationAgent, "run", counting_content_run),
        patch.object(ContentValidatorAgent, "run", review_failing_once),
        patch(
            "mystery_agents.agents.a9_packaging.PackagingAgent.run", side_effect=lambda s, **_: s
        ),
    ):
        final = create_workflow().invoke(state, config={"recursion_limit": DEFAULT_RECURSION_LIMIT})

    assert content_calls == 2
    assert reviews == 2
    assert final["content_validation"].is_consistent
