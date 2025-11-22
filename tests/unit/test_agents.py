"""Unit tests for agent functionality (pure unit tests, no agent initialization)."""

from mystery_agents.models.state import (
    GameConfig,
    GameState,
    MetaInfo,
    PlayerConfig,
    ValidationReport,
)
from mystery_agents.utils.constants import (
    TEST_DEFAULT_DURATION,
    TEST_DEFAULT_PLAYERS,
)


def test_state_initialization() -> None:
    """Test that GameState initializes correctly."""
    state = GameState(
        meta=MetaInfo(),
        config=GameConfig(
            players=PlayerConfig(total=TEST_DEFAULT_PLAYERS),
            host_gender="male",
            country="Spain",
            duration_minutes=TEST_DEFAULT_DURATION,
        ),
    )

    assert state.meta.id is not None
    assert state.config.players.total == TEST_DEFAULT_PLAYERS
    assert state.world is None
    assert state.crime is None
    assert len(state.characters) == 0
    assert state.retry_count == 0
    assert state.max_retries == 3  # From model default


def test_retry_counter() -> None:
    """Test that retry counter works correctly."""
    state = GameState(
        meta=MetaInfo(),
        config=GameConfig(
            players=PlayerConfig(total=TEST_DEFAULT_PLAYERS),
            host_gender="male",
            country="Spain",
            duration_minutes=TEST_DEFAULT_DURATION,
        ),
    )

    assert state.retry_count == 0

    # Simulate retry
    state.retry_count += 1
    assert state.retry_count == 1

    # Check max retries (from model default)
    assert state.max_retries == 3


def test_validation_routing_logic() -> None:
    """Test the validation routing function logic."""
    from mystery_agents.graph.workflow import should_retry_validation

    # Note: retry_count is incremented in v2_game_logic_validator_node, not in should_retry_validation
    # This test verifies that should_retry_validation correctly routes based on retry_count

    # Case 1: Validation passes
    state = GameState(
        meta=MetaInfo(),
        config=GameConfig(
            players=PlayerConfig(total=TEST_DEFAULT_PLAYERS),
            host_gender="male",
            country="Spain",
            duration_minutes=TEST_DEFAULT_DURATION,
        ),
    )
    state.validation = ValidationReport(is_consistent=True, issues=[], suggested_fixes=[])

    result = should_retry_validation(state)
    assert result == "pass"

    # Case 2: Validation fails but can retry (retry_count will be incremented in validator node)
    state.validation = ValidationReport(is_consistent=False, issues=[], suggested_fixes=[])
    state.retry_count = 0  # Before validator node increments it

    result = should_retry_validation(state)
    assert result == "retry"
    # Note: retry_count is NOT incremented here - it's incremented in v2_game_logic_validator_node
    assert state.retry_count == 0

    # Case 3: Max retries exceeded
    state.retry_count = 3  # Already at max
    state.validation = ValidationReport(is_consistent=False, issues=[], suggested_fixes=[])
    result = should_retry_validation(state)
    assert result == "fail"
