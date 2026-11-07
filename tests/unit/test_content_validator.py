"""Unit tests for ContentValidatorAgent (V3) and its routing."""

from unittest.mock import patch

import pytest

from mystery_agents.agents.v3_content_validator import (
    ContentValidatorAgent,
    drop_revealing_questions,
    find_clue_notes_naming_killer,
    find_clue_objects_worn_by_one,
    find_clues_naming_killer,
    find_killer_in_guiding_questions,
    find_over_obvious_clues,
    find_rule_violations,
    find_suspect_names_in_titles,
)
from mystery_agents.graph.workflow import (
    route_after_game_validation,
    should_retry_content_validation,
)
from mystery_agents.models.state import (
    CharacterSpec,
    ClueSolutionEntry,
    ClueSpec,
    DeductionPlan,
    DetectiveRole,
    EvidencePiece,
    GameConfig,
    GameState,
    HostGuide,
    KillerSelection,
    MetaInfo,
    PlayerConfig,
    ValidationIssue,
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


def test_timeline_issues_are_sent_back_to_their_author(state: GameState) -> None:
    """A8 can't change the players' sheets, so timeline issues retry through A7.5 first."""
    state.content_validation = ValidationReport(
        is_consistent=False,
        issues=[ValidationIssue(type="contradiction", description="x", fix_in="timelines")],
    )
    state.content_retry_count = 1

    assert should_retry_content_validation(state) == "retry_timelines"

    state.content_retry_count = 3
    assert should_retry_content_validation(state) == "fail"


def test_logic_issues_in_sheets_rewrite_only_the_sheets(state: GameState) -> None:
    """V2 issues inside personal timelines don't need a new global timeline."""
    state.validation = ValidationReport(
        is_consistent=False,
        issues=[ValidationIssue(type="timeline_conflict", description="x", fix_in="timelines")],
    )
    state.retry_count = 1

    assert route_after_game_validation(state) == ["a7_5_revise_for_logic"]

    state.validation.issues[0].fix_in = "content"
    assert route_after_game_validation(state) == ["a6_timeline"]


def test_out_of_retries_with_only_sheet_issues_keeps_the_game(state: GameState) -> None:
    """The plot holds when only the sheets still disagree, so the game isn't thrown away."""
    from langgraph.graph import END

    state.validation = ValidationReport(
        is_consistent=False,
        issues=[ValidationIssue(type="timeline_conflict", description="x", fix_in="timelines")],
    )
    state.retry_count = state.max_retries

    assert route_after_game_validation(state) == ["a8_content", "a3_5_character_images"]

    state.validation.issues.append(ValidationIssue(type="logic_gap", description="y"))
    assert route_after_game_validation(state) == [END]


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


def test_questions_naming_the_killer_are_dropped() -> None:
    """Questions that name the killer are removed; the others stay."""
    state = _game_with_killer()
    detective = state.host_guide.host_act2_detective_role
    detective.guiding_questions = [
        "¿Dónde estaba Manrique a las 22:10?",
        "¿Quién pasó por el pasillo entre las 22:10 y las 22:20?",
    ]

    removed = drop_revealing_questions(state)

    assert removed == ["¿Dónde estaba Manrique a las 22:10?"]
    assert detective.guiding_questions == [
        "¿Quién pasó por el pasillo entre las 22:10 y las 22:20?"
    ]


def test_questions_citing_the_decisive_sightings_are_dropped() -> None:
    """Asking about the minute of a breaking sighting gives it away; a wide window doesn't."""
    state = _game_with_killer()
    assert state.killer_selection is not None
    state.killer_selection.deduction_plan = DeductionPlan(
        breaking_pieces=[
            EvidencePiece(
                kind="observation",
                holder_id="char-1",
                about_id="char-0",
                time="22:12",
                description="x",
            )
        ],
        means_piece=EvidencePiece(
            kind="clue", holder_id="", about_id="char-0", time="21:00", description="x"
        ),
        red_herrings=[],
    )
    detective = state.host_guide.host_act2_detective_role
    detective.guiding_questions = [
        "¿Qué vio Elena desde el salón?",
        "¿Quién cruzaba el patio hacia las 22:13?",
        "¿Dónde estaba cada uno entre las 22:12 y las 22:30?",
    ]

    removed = drop_revealing_questions(state)

    assert removed == ["¿Quién cruzaba el patio hacia las 22:13?"]


def test_clue_notes_naming_the_killer_are_flagged() -> None:
    """The host reads the notes before the reveal, so they can't say whom a clue convicts."""
    state = _game_with_killer()
    state.host_guide.host_act2_detective_role.clues_to_reveal = [
        ClueSolutionEntry(clue_id="c1", how_to_interpret="Contradice la coartada de Manrique."),
        ClueSolutionEntry(clue_id="c2", how_to_interpret="Sugiere un motivo económico."),
    ]

    issues = find_clue_notes_naming_killer(state)

    assert [issue.related_ids[0] for issue in issues] == ["c1"]


def test_clues_that_identify_the_killer_alone_are_flagged() -> None:
    """A clue naming only the killer, or their unique initials on an object, solves the case."""
    state = _game_with_killer()
    state.clues[0].description = "La suela coincide con los zapatos de Manrique."
    state.clues[1].description = "Un pañuelo bordado con las iniciales Á.M. junto a la puerta."
    state.clues[2].description = "Una lista de deudores: Manrique, Elena Ojeda y Mateo Suárez."

    issues = find_clues_naming_killer(state)

    named = [issue for issue in issues if "names the killer" in issue.description]
    assert [issue.related_ids[0] for issue in named] == [state.clues[0].id]
    assert any("initials" in issue.description for issue in issues)


def test_clue_objects_only_one_costume_explains_are_flagged() -> None:
    """Cufflinks only the killer wears name them; once others wear them too, they don't."""
    state = _game_with_killer()
    state.characters[0].costume_suggestion = "Traje gris con gemelos de oro"
    state.characters[1].costume_suggestion = "Vestido largo de seda"
    state.clues[0].description = "Un gemelo de oro junto al cuerpo."

    issues = find_clue_objects_worn_by_one(state)

    assert [issue.related_ids for issue in issues] == [["c1"]]
    assert "gemelo" in issues[0].description

    state.costume_additions = {"char-2": "gemelos de oro"}
    assert find_clue_objects_worn_by_one(state) == []


def test_incriminating_clue_with_an_object_only_the_victim_wears_is_flagged() -> None:
    """The victim's own cufflink at the scene can't point at a suspect."""
    from mystery_agents.models.state import CrimeScene, CrimeSpec, MurderMethod, VictimSpec

    state = _game_with_killer()
    state.crime = CrimeSpec(
        victim=VictimSpec(
            name="Don Tomás",
            age=70,
            gender="male",
            role_in_setting="Host",
            public_persona="x",
            costume_suggestion="Frac con gemelos grabados",
        ),
        murder_method=MurderMethod(type="other", description="x", weapon_used="x"),
        crime_scene=CrimeScene(room_id="Study", description="x"),
        time_of_death_approx="22:00",
    )
    state.clues[0].description = "Un gemelo grabado en el balcón."
    state.clues[0].incriminates = ["char-0"]

    issues = find_clue_objects_worn_by_one(state)

    assert len(issues) == 1 and "victim" in issues[0].description
