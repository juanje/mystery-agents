"""Integration tests for checkpointing and resuming generations."""

import logging
from pathlib import Path
from typing import Any
from unittest.mock import patch

import pytest
import yaml

from mystery_agents.agents.a2_world import WorldAgent
from mystery_agents.agents.a8_content import ContentGenerationAgent
from mystery_agents.graph.checkpoints import checkpoints_path, open_checkpointer
from mystery_agents.graph.workflow import create_workflow
from mystery_agents.models.state import GameConfig, GameState, MetaInfo, PlayerConfig
from mystery_agents.utils.constants import (
    DEFAULT_RECURSION_LIMIT,
    TEST_MIN_DURATION,
    TEST_MIN_PLAYERS,
)


@pytest.fixture
def initial_state(tmp_path: Path) -> GameState:
    """Dry-run state for a small game written under tmp_path."""
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
    return GameState(
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


def test_resume_after_failure_skips_completed_nodes(
    initial_state: GameState, caplog: pytest.LogCaptureFixture
) -> None:
    """A failed run resumes from its last checkpoint without redoing earlier nodes."""
    output_dir = initial_state.config.output_dir
    config: Any = {
        "recursion_limit": DEFAULT_RECURSION_LIMIT,
        "configurable": {"thread_id": initial_state.meta.id[:8]},
    }
    original_world_run = WorldAgent.run
    world_calls = 0

    def counting_world_run(self: WorldAgent, state: GameState) -> GameState:
        nonlocal world_calls
        world_calls += 1
        return original_world_run(self, state)

    caplog.set_level(logging.WARNING)
    with (
        patch.object(WorldAgent, "run", counting_world_run),
        patch(
            "mystery_agents.agents.a9_packaging.PackagingAgent.run", side_effect=lambda s, **_: s
        ),
    ):
        with open_checkpointer(output_dir) as checkpointer:
            workflow = create_workflow(checkpointer=checkpointer)
            with patch.object(ContentGenerationAgent, "run", side_effect=RuntimeError("API down")):
                with pytest.raises(RuntimeError, match="API down"):
                    workflow.invoke(initial_state, config=config)

            assert workflow.get_state(config).next == ("a8_content",)

        # Reopen the database, as a new CLI process would
        with open_checkpointer(output_dir) as checkpointer:
            final = create_workflow(checkpointer=checkpointer).invoke(None, config=config)

    assert world_calls == 1
    assert final["clues"]
    assert checkpoints_path(output_dir).exists()
    assert not [r for r in caplog.records if "Blocked deserialization" in r.getMessage()]
