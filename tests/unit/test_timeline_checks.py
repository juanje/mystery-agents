"""Unit tests for the deterministic personal timeline checks."""

import pytest

from mystery_agents.models.state import (
    CharacterSpec,
    GameConfig,
    GameState,
    KillerSelection,
    MetaInfo,
    Observation,
    PersonalEvent,
    PersonalTimeline,
    PlayerConfig,
)
from mystery_agents.utils.constants import TEST_DEFAULT_DURATION, TEST_DEFAULT_PLAYERS
from mystery_agents.utils.timeline_checks import (
    drop_excess_killer_sightings,
    event_ranges,
    find_killer_sheet_stands_out,
    find_misplaced_observations,
    find_observation_time_errors,
    find_timeline_violations,
    find_too_many_killer_witnesses,
    same_place,
)

NAMES = {"killer": "Diego Ojeda", "ana": "Ana Ruiz", "berta": "Berta Sosa", "carla": "Carla Gil"}


def _event(
    time: str, location: str, hides: bool = False, seen: list[Observation] | None = None
) -> PersonalEvent:
    return PersonalEvent(
        time=time,
        location=location,
        what_they_really_did="You went to the study." if hides else "You stayed.",
        what_they_will_tell_others="You stayed on the terrace." if hides else "You stayed.",
        info_they_observed=seen or [],
    )


def _seen(time: str, who: str, where: str) -> Observation:
    return Observation(time=time, observed_character_id=who, location=where, text=f"You saw {who}.")


@pytest.fixture
def state() -> GameState:
    """The killer slips from the terrace to the study at 21:50-22:20; nobody sees it yet."""
    timelines = {
        "killer": [_event("21:00-21:50", "Terrace"), _event("21:50-22:20", "Study", hides=True)],
        "ana": [_event("21:00-21:50", "Dining room"), _event("21:50-22:20", "Library")],
        "berta": [_event("21:00-21:50", "Dining room"), _event("21:50-22:20", "Cellar")],
        "carla": [_event("21:00-21:50", "Dining room"), _event("21:50-22:20", "Garden")],
    }
    return GameState(
        meta=MetaInfo(),
        config=GameConfig(
            players=PlayerConfig(total=TEST_DEFAULT_PLAYERS),
            duration_minutes=TEST_DEFAULT_DURATION,
            difficulty="medium",
        ),
        characters=[
            CharacterSpec(
                id=char_id,
                name=name,
                gender="female",
                age_range="30-40",
                role="Guest",
                public_description="A guest",
                relation_to_victim="Friend",
            )
            for char_id, name in NAMES.items()
        ],
        killer_selection=KillerSelection(killer_id="killer", rationale="r", truth_narrative="t"),
        personal_timelines={
            char_id: PersonalTimeline(character_id=char_id, events=events, subjective_narrative="")
            for char_id, events in timelines.items()
        },
    )


def test_single_times_last_until_the_next_event() -> None:
    """A moment given as one time lasts until the next one starts."""
    events = [_event("22:25", "Hall"), _event("22:26-22:35", "Study"), _event("?", "Garden")]

    assert event_ranges(events) == [(1345, 1346), (1346, 1355), None]


def test_place_names_match_on_distinctive_words() -> None:
    """Different wordings of a place match; different places don't."""
    assert same_place("Pasillo exterior del piso superior", "el pasillo superior")
    assert same_place("Biblioteca", "biblioteca")
    assert not same_place("Terraza", "Despacho")


def test_consistent_timelines_pass(state: GameState) -> None:
    """Observations at the right time and place raise no issue."""
    state.personal_timelines["ana"].events[1].info_they_observed = [
        _seen("22:00", "berta", "Cellar")
    ]

    assert find_timeline_violations(state) == []


def test_observation_outside_its_moment_is_flagged(state: GameState) -> None:
    """Seeing something at 21:30 can't be listed under a moment that starts at 21:50."""
    state.personal_timelines["ana"].events[1].info_they_observed = [
        _seen("21:30", "berta", "Dining room")
    ]

    issues = find_observation_time_errors(state)

    assert len(issues) == 1
    assert issues[0].related_ids == ["ana"]
    assert issues[0].fix_in == "timelines"


def test_suspect_seen_where_they_were_not_is_flagged(state: GameState) -> None:
    """Ana can't see Berta in the garden while Berta's own timeline has her in the cellar."""
    state.personal_timelines["ana"].events[1].info_they_observed = [
        _seen("22:00", "berta", "Garden")
    ]

    issues = find_misplaced_observations(state)

    assert len(issues) == 1
    assert "Cellar" in issues[0].description
    assert issues[0].related_ids == ["ana"]


def test_two_killer_witnesses_are_fine_for_medium(state: GameState) -> None:
    """Two partial sightings make players compare testimonies."""
    for witness in ("ana", "berta"):
        state.personal_timelines[witness].events[1].info_they_observed = [
            _seen("22:00", "killer", "Study")
        ]

    assert find_too_many_killer_witnesses(state) == []


def test_every_suspect_seeing_the_killer_is_flagged(state: GameState) -> None:
    """Three witnesses on medium solve the case alone; the extra ones are rewritten."""
    for witness in ("ana", "berta", "carla"):
        state.personal_timelines[witness].events[1].info_they_observed = [
            _seen("22:00", "killer", "Study")
        ]

    issues = find_too_many_killer_witnesses(state)

    assert len(issues) == 1
    assert issues[0].type == "over_obvious"
    assert issues[0].related_ids == ["carla"]


def test_seeing_the_killer_while_they_tell_the_truth_does_not_count(state: GameState) -> None:
    """Sightings outside the killer's hidden moments don't break their account."""
    for witness in ("ana", "berta", "carla"):
        state.personal_timelines[witness].events[0].info_they_observed = [
            _seen("21:30", "killer", "Terrace")
        ]

    assert find_too_many_killer_witnesses(state) == []


def test_unaware_killer_alone_with_nothing_to_hide_is_flagged(state: GameState) -> None:
    """If every innocent hides something, the unaware killer's sheet is the odd one out."""
    state.config.killer_knows_identity = False
    for char_id in ("ana", "berta", "carla"):
        state.personal_timelines[char_id].events[1] = _event("21:50-22:20", "Library", hides=True)

    issues = find_killer_sheet_stands_out(state)

    assert len(issues) == 1
    assert issues[0].fix_in == "timelines"
    assert issues[0].related_ids  # at least one innocent becomes truthful


def test_some_truthful_innocents_hide_the_killer_sheet(state: GameState) -> None:
    """With a truthful innocent around, the killer's sheet looks like theirs."""
    state.config.killer_knows_identity = False
    state.personal_timelines["ana"].events[1] = _event("21:50-22:20", "Library", hides=True)

    assert find_killer_sheet_stands_out(state) == []


def test_people_seen_on_their_way_between_places_pass(state: GameState) -> None:
    """Seeing Berta leave the dining room for the cellar fits either of her moments."""
    state.personal_timelines["ana"].events[1].info_they_observed = [
        Observation(
            time="21:52",
            observed_character_id="berta",
            location="Hall",
            text="You saw Berta Sosa walk towards the cellar.",
        )
    ]

    assert find_misplaced_observations(state) == []


def test_excess_killer_sightings_are_removed(state: GameState) -> None:
    """Beyond the limit, extra witnesses lose their killer sightings but keep the rest."""
    other = _seen("22:00", "ana", "Library")
    for witness in ("ana", "berta", "carla"):
        state.personal_timelines[witness].events[1].info_they_observed = [
            _seen("22:00", "killer", "Study")
        ]
    state.personal_timelines["carla"].events[1].info_they_observed.append(other)

    trimmed = drop_excess_killer_sightings(state)

    assert trimmed == ["carla"]
    assert state.personal_timelines["carla"].events[1].info_they_observed == [other]
    assert len(state.personal_timelines["ana"].events[1].info_they_observed) == 1
    assert find_too_many_killer_witnesses(state) == []
