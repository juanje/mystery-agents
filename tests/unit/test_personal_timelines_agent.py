"""Unit tests for PersonalTimelinesAgent (A7.5)."""

from unittest.mock import patch

import pytest

from mystery_agents.agents.a7_5_personal_timelines import A7_5Output, PersonalTimelinesAgent
from mystery_agents.agents.base import StructuredOutputError
from mystery_agents.models.state import (
    CharacterSpec,
    GameConfig,
    GameState,
    KillerSelection,
    MetaInfo,
    PersonalTimeline,
    PlayerConfig,
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


@pytest.fixture
def state() -> GameState:
    """State with two suspects and a selected killer."""
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
    )


def _timeline(char_id: str) -> PersonalTimeline:
    return PersonalTimeline(character_id=char_id, subjective_narrative="A quiet evening.")


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
    """The prompt identifies the killer, and timelines are stored by character ID."""
    agent = PersonalTimelinesAgent()
    output = A7_5Output(timelines=[_timeline("char-1"), _timeline("char-2"), _timeline("ghost")])

    with patch.object(agent, "invoke", return_value=output) as mock_invoke:
        result = agent.run(state)

    assert "Ana (ID: char-1) **THE KILLER**" in mock_invoke.call_args[0][1]
    assert set(result.personal_timelines) == {"char-1", "char-2"}


def test_run_raises_when_a_suspect_is_missing(state: GameState) -> None:
    """A missing suspect raises StructuredOutputError so the graph retries the node."""
    agent = PersonalTimelinesAgent()

    with patch.object(agent, "invoke", return_value=A7_5Output(timelines=[_timeline("char-1")])):
        with pytest.raises(StructuredOutputError, match="char-2"):
            agent.run(state)


def test_run_asks_for_missing_suspects_only(state: GameState) -> None:
    """A partial answer is completed by asking only for the missing suspects."""
    agent = PersonalTimelinesAgent()
    answers = [
        A7_5Output(timelines=[_timeline("char-1")]),
        A7_5Output(timelines=[_timeline("char-2")]),
    ]

    with patch.object(agent, "invoke", side_effect=answers) as mock_invoke:
        result = agent.run(state)

    assert set(result.personal_timelines) == {"char-1", "char-2"}
    assert mock_invoke.call_count == 2
    second_message = mock_invoke.call_args_list[1][0][1]
    assert "ALREADY WRITTEN" in second_message
    assert "### Ana (ID: char-1)" in second_message
    assert "Now return ONLY the timelines for these character_ids: char-2" in second_message
