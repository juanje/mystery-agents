"""Unit tests for PersonalTimelinesAgent (A7.5)."""

from unittest.mock import patch

import pytest

from mystery_agents.agents.a7_5_personal_timelines import (
    PersonalTimelinesAgent,
    SheetsOutput,
    SlotText,
    SuspectSheet,
)
from mystery_agents.agents.base import StructuredOutputError
from mystery_agents.models.state import (
    CharacterSpec,
    GameConfig,
    GameState,
    KillerSelection,
    MetaInfo,
    PersonalEvent,
    PersonalTimeline,
    PlayerConfig,
    PresenceGrid,
    PresenceSlot,
    SuspectPresence,
    ValidationIssue,
)
from mystery_agents.utils.constants import TEST_DEFAULT_DURATION, TEST_DEFAULT_PLAYERS


def _character(char_id: str, name: str) -> CharacterSpec:
    return CharacterSpec(
        id=char_id,
        name=name,
        gender="female",
        age_range="30-40",
        role="Guest",
        public_description="A guest",
        relation_to_victim="Friend",
    )


def _grid(char_ids: list[str]) -> PresenceGrid:
    return PresenceGrid(
        suspects=[
            SuspectPresence(
                character_id=char_id,
                slots=[
                    PresenceSlot(time="21:00-21:30", room="Hall", doing="Drinks."),
                    PresenceSlot(time="21:30-22:00", room="Hall", doing="Talks."),
                ],
            )
            for char_id in char_ids
        ]
    )


@pytest.fixture
def state() -> GameState:
    """State with two suspects, a selected killer and their presence grid."""
    return GameState(
        meta=MetaInfo(),
        config=GameConfig(
            players=PlayerConfig(total=TEST_DEFAULT_PLAYERS),
            duration_minutes=TEST_DEFAULT_DURATION,
        ),
        characters=[_character("char-1", "Ana"), _character("char-2", "Berta")],
        killer_selection=KillerSelection(
            killer_id="char-1", rationale="Opportunity", truth_narrative="Ana did it."
        ),
        presence=_grid(["char-1", "char-2"]),
    )


def _sheet(char_id: str, slots: int = 2, text: str = "You waited.") -> SuspectSheet:
    return SuspectSheet(
        character_id=char_id,
        slots=[
            SlotText(slot=number, what_they_really_did=text, what_they_will_tell_others=text)
            for number in range(1, slots + 1)
        ],
        subjective_narrative="A quiet evening.",
    )


def _timeline(char_id: str) -> PersonalTimeline:
    event = PersonalEvent(
        time="21:00-21:30",
        location="Hall",
        what_they_really_did="You waited.",
        what_they_will_tell_others="You waited.",
    )
    return PersonalTimeline(
        character_id=char_id, subjective_narrative="A quiet evening.", events=[event]
    )


def _requested(message: str) -> list[str]:
    return message.rsplit("character_ids: ", 1)[1].split("\n", 1)[0].split(", ")


def test_mock_output_gives_every_suspect_a_timeline(state: GameState) -> None:
    """Dry run creates a timeline per suspect; only the killer's story hides something."""
    state.config.dry_run = True

    result = PersonalTimelinesAgent().run(state)

    assert set(result.personal_timelines) == {"char-1", "char-2"}
    killer_event = result.personal_timelines["char-1"].events[1]
    innocent_event = result.personal_timelines["char-2"].events[1]
    assert killer_event.what_they_really_did != killer_event.what_they_will_tell_others
    assert innocent_event.what_they_really_did == innocent_event.what_they_will_tell_others


def test_run_marks_killer_and_stores_timelines(state: GameState) -> None:
    """The prompt identifies the killer, and sheets are built from the grid by character ID."""
    agent = PersonalTimelinesAgent()
    output = SheetsOutput(sheets=[_sheet("char-1"), _sheet("char-2"), _sheet("ghost")])

    with patch.object(agent, "invoke", return_value=output) as mock_invoke:
        result = agent.run(state)

    assert "Ana (ID: char-1) **THE KILLER**" in mock_invoke.call_args[0][1]
    assert set(result.personal_timelines) == {"char-1", "char-2"}
    events = result.personal_timelines["char-2"].events
    assert [(e.time, e.location) for e in events] == [
        ("21:00-21:30", "Hall"),
        ("21:30-22:00", "Hall"),
    ]


def test_run_requires_the_presence_grid(state: GameState) -> None:
    """Sheets are built from the grid, so A7.5 can't run before A7.4."""
    state.presence = None

    with pytest.raises(ValueError, match="presence grid"):
        PersonalTimelinesAgent().run(state)


def test_run_raises_when_a_suspect_is_missing(state: GameState) -> None:
    """A missing suspect raises StructuredOutputError so the graph retries the node."""
    agent = PersonalTimelinesAgent()

    with patch.object(agent, "invoke", return_value=SheetsOutput(sheets=[_sheet("char-1")])):
        with pytest.raises(StructuredOutputError, match="char-2"):
            agent.run(state)


def test_run_asks_for_missing_suspects_only(state: GameState) -> None:
    """A partial answer is completed by asking only for the missing suspects."""
    agent = PersonalTimelinesAgent()
    answers = [
        SheetsOutput(sheets=[_sheet("char-1")]),
        SheetsOutput(sheets=[_sheet("char-2")]),
    ]

    with patch.object(agent, "invoke", side_effect=answers) as mock_invoke:
        result = agent.run(state)

    assert set(result.personal_timelines) == {"char-1", "char-2"}
    assert mock_invoke.call_count == 2
    second_message = mock_invoke.call_args_list[1][0][1]
    assert "ALREADY WRITTEN" in second_message
    assert "### Ana (ID: char-1)" in second_message
    assert _requested(second_message) == ["char-2"]


def test_run_writes_suspects_in_batches_killer_first(state: GameState) -> None:
    """Suspects are requested in small batches, starting with the killer."""
    state.characters = [_character(f"char-{n}", f"Name {n}") for n in range(1, 5)]
    state.presence = _grid([char.id for char in state.characters])
    state.killer_selection = KillerSelection(
        killer_id="char-3", rationale="Opportunity", truth_narrative="Name 3 did it."
    )
    agent = PersonalTimelinesAgent()

    def answer(_: GameState, message: str) -> SheetsOutput:
        return SheetsOutput(sheets=[_sheet(char_id) for char_id in _requested(message)])

    with patch.object(agent, "invoke", side_effect=answer) as mock_invoke:
        result = agent.run(state)

    assert len(result.personal_timelines) == 4
    messages = [call[0][1] for call in mock_invoke.call_args_list]
    assert [_requested(message) for message in messages] == [
        ["char-3", "char-1", "char-2"],
        ["char-4"],
    ]
    assert "ALREADY WRITTEN" in messages[1]


def test_run_asks_again_for_empty_sheets(state: GameState) -> None:
    """A sheet without slot texts is treated as missing and requested again."""
    agent = PersonalTimelinesAgent()
    answers = [
        SheetsOutput(sheets=[_sheet("char-1"), _sheet("char-2", slots=0)]),
        SheetsOutput(sheets=[_sheet("char-2")]),
    ]

    with patch.object(agent, "invoke", side_effect=answers) as mock_invoke:
        result = agent.run(state)

    assert mock_invoke.call_count == 2
    assert len(result.personal_timelines["char-2"].events) == 2


def test_revise_rewrites_only_flagged_suspects(state: GameState) -> None:
    """A review issue rewrites the flagged suspect's timeline and keeps the others as context."""
    original = {"char-1": _timeline("char-1"), "char-2": _timeline("char-2")}
    state.personal_timelines = dict(original)
    agent = PersonalTimelinesAgent()
    issue = ValidationIssue(
        type="contradiction",
        description="Berta sees the hall from the garden",
        related_ids=["char-2"],
        fix_in="timelines",
    )
    revised = SheetsOutput(sheets=[_sheet("char-2", text="You read.")])

    with patch.object(agent, "invoke", return_value=revised) as mock_invoke:
        result = agent.revise(state, [issue])

    request = mock_invoke.call_args[0][1]
    assert "Berta sees the hall from the garden" in request
    assert "ALREADY WRITTEN" in request
    assert _requested(request) == ["char-2"]
    assert result.personal_timelines["char-1"] is original["char-1"]
    assert [e.what_they_really_did for e in result.personal_timelines["char-2"].events] == [
        "You read.",
        "You read.",
    ]


def test_revise_without_known_suspects_changes_nothing(state: GameState) -> None:
    """An issue that names no suspect can't be fixed here, so no call is made."""
    state.personal_timelines = {"char-1": _timeline("char-1")}
    agent = PersonalTimelinesAgent()
    issue = ValidationIssue(type="contradiction", description="x", fix_in="timelines")

    with patch.object(agent, "invoke") as mock_invoke:
        agent.revise(state, [issue])

    mock_invoke.assert_not_called()
