"""Unit tests for ContentValidatorAgent (V3) and its routing."""

from unittest.mock import patch

import pytest

from mystery_agents.agents.v3_content_validator import (
    ContentValidatorAgent,
    find_suspect_names_in_titles,
)
from mystery_agents.graph.workflow import should_retry_content_validation
from mystery_agents.models.state import (
    CharacterSpec,
    ClueSpec,
    GameConfig,
    GameState,
    MetaInfo,
    PlayerConfig,
    ValidationReport,
)
from mystery_agents.utils.constants import TEST_DEFAULT_DURATION, TEST_DEFAULT_PLAYERS


@pytest.fixture
def state() -> GameState:
    """State with one suspect and two clues, one naming her in the title."""
    return GameState(
        meta=MetaInfo(),
        config=GameConfig(
            players=PlayerConfig(total=TEST_DEFAULT_PLAYERS),
            duration_minutes=TEST_DEFAULT_DURATION,
        ),
        characters=[
            CharacterSpec(
                id="char-1",
                name="Dra. Sofía del Castillo",
                gender="female",
                age_range="40-50",
                role="Art expert",
                public_description="Scholarly",
                relation_to_victim="Advisor",
            )
        ],
        clues=[
            ClueSpec(id="clue-1", type="note", title="Torn letter", description="..."),
            ClueSpec(id="clue-2", type="object", title="El maletín de Castillo", description="..."),
        ],
    )


def test_find_suspect_names_in_titles(state: GameState) -> None:
    """Only the clue whose title names a suspect is flagged."""
    issues = find_suspect_names_in_titles(state)

    assert len(issues) == 1
    assert issues[0].type == "spoiler"
    assert issues[0].related_ids == ["clue-2", "char-1"]


def test_run_adds_title_issues_to_llm_report(state: GameState) -> None:
    """A passing LLM review still fails if a clue title names a suspect."""
    agent = ContentValidatorAgent()
    passing = ValidationReport(is_consistent=True, issues=[], suggested_fixes=[])

    with patch.object(agent, "invoke", return_value=passing):
        result = agent.run(state)

    assert result.content_validation is not None
    assert result.content_validation.is_consistent is False
    assert [issue.type for issue in result.content_validation.issues] == ["spoiler"]


def test_mock_output_passes(state: GameState) -> None:
    """Dry run always passes content review."""
    state.config.dry_run = True

    result = ContentValidatorAgent().run(state)

    assert result.content_validation is not None
    assert result.content_validation.is_consistent is True


@pytest.mark.parametrize(
    ("is_consistent", "retry_count", "expected"),
    [(True, 0, "pass"), (False, 1, "retry"), (False, 2, "retry"), (False, 3, "fail")],
)
def test_should_retry_content_validation(
    state: GameState, is_consistent: bool, retry_count: int, expected: str
) -> None:
    """Failed reviews rewrite the content until retries run out, then package anyway."""
    state.content_validation = ValidationReport(
        is_consistent=is_consistent, issues=[], suggested_fixes=[]
    )
    state.content_retry_count = retry_count

    assert should_retry_content_validation(state) == expected
