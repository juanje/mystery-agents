"""Unit tests for ContentValidatorAgent (V3) and its routing."""

from unittest.mock import patch

import pytest

from mystery_agents.agents.v3_content_validator import (
    ContentValidatorAgent,
    find_killer_in_guiding_questions,
    find_over_obvious_clues,
    find_rule_violations,
    find_suspect_names_in_titles,
)
from mystery_agents.graph.workflow import should_retry_content_validation
from mystery_agents.models.state import (
    CharacterSpec,
    ClueSpec,
    DetectiveRole,
    GameConfig,
    GameState,
    HostGuide,
    KillerSelection,
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


def _game_with_killer() -> GameState:
    """Four suspects, killer 'Dr. Álvaro Manrique', with a host guide and three clues."""
    names = ["Dr. Álvaro Manrique", "Elena Ojeda", "Mateo Suárez", "Beatriz Castillo"]
    return GameState(
        meta=MetaInfo(),
        config=GameConfig(
            players=PlayerConfig(total=TEST_DEFAULT_PLAYERS),
            duration_minutes=TEST_DEFAULT_DURATION,
        ),
        characters=[
            CharacterSpec(
                id=f"char-{n}",
                name=name,
                gender="male",
                age_range="40-50",
                role="Guest",
                public_description="A guest",
                relation_to_victim="Friend",
            )
            for n, name in enumerate(names)
        ],
        killer_selection=KillerSelection(killer_id="char-0", rationale="R", truth_narrative="T"),
        clues=[
            ClueSpec(id="c1", type="object", title="Wine glass", description="..."),
            ClueSpec(
                id="c2", type="note", title="Deeds", description="...", incriminates=["char-1"]
            ),
            ClueSpec(
                id="c3",
                type="note",
                title="Customs log",
                description="...",
                incriminates=["char-2"],
            ),
        ],
        host_guide=HostGuide(
            spoiler_free_intro="Intro",
            host_act2_detective_role=DetectiveRole(
                public_description="Sharp",
                clues_to_reveal=[],
                final_solution_script="It was him.",
                guiding_questions=[
                    "Who was near the library between 22:08 and 22:15?",
                    "What was the doctor Álvaro doing in the gallery at 22:16?",
                ],
            ),
        ),
    )


def test_find_killer_in_guiding_questions() -> None:
    """Questions naming the killer are spoilers; neutral ones are fine."""
    issues = find_killer_in_guiding_questions(_game_with_killer())

    assert len(issues) == 1
    assert issues[0].type == "spoiler"
    assert "Álvaro" in issues[0].description


def test_find_over_obvious_clue_that_clears_everyone_else() -> None:
    """A clue pointing only at the killer while clearing the rest solves the case alone."""
    state = _game_with_killer()
    state.clues.append(
        ClueSpec(
            id="c4",
            type="object",
            title="Prescription bottle",
            description="Found next to the doctor's bag.",
            incriminates=["char-0"],
            exonerates=["char-1", "char-2", "char-3"],
        )
    )

    issues = find_over_obvious_clues(state)

    assert [issue.type for issue in issues] == ["over_obvious"]
    assert "Clue 4" in issues[0].description


def test_find_over_obvious_when_most_clues_point_at_killer() -> None:
    """More than half the clues against the killer is too easy."""
    state = _game_with_killer()
    state.clues = [
        ClueSpec(
            id=f"k{n}", type="note", title=f"Note {n}", description="...", incriminates=["char-0"]
        )
        for n in range(3)
    ] + state.clues[:2]

    issues = find_over_obvious_clues(state)

    assert any("3 of 5 clues point at the killer" in issue.description for issue in issues)


def test_balanced_game_has_no_rule_violations() -> None:
    """A game with neutral titles and questions and balanced clues passes the checks."""
    state = _game_with_killer()
    assert state.host_guide is not None and state.host_guide.host_act2_detective_role is not None
    state.host_guide.host_act2_detective_role.guiding_questions = ["Who was in the corridor?"]

    assert find_rule_violations(state) == []


def test_killer_question_check_ignores_shared_surname() -> None:
    """A question about the killer's relative (same surname) is not flagged."""
    state = _game_with_killer()
    state.characters[1].name = "Mateo Manrique"
    assert state.host_guide is not None and state.host_guide.host_act2_detective_role is not None
    state.host_guide.host_act2_detective_role.guiding_questions = [
        "What was Mateo Manrique doing in the cellar at 22:05?"
    ]

    assert find_killer_in_guiding_questions(state) == []


def test_title_check_ignores_family_surname(state: GameState) -> None:
    """A title with a surname several characters share doesn't single out one suspect."""
    state.characters.append(
        state.characters[0].model_copy(update={"id": "char-2", "name": "Ana Castillo"})
    )
    state.clues = [
        ClueSpec(id="clue-3", type="note", title="Castillo Foundation audit", description="...")
    ]

    assert find_suspect_names_in_titles(state) == []
