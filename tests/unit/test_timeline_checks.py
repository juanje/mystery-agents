"""Unit tests for the deterministic personal timeline checks."""

import pytest

from mystery_agents.models.state import (
    CharacterSpec,
    DeductionPlan,
    EvidencePiece,
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
    find_absent_from_shared_events,
    find_impossible_planned_sightings,
    find_impossible_sightings,
    find_killer_sheet_stands_out,
    find_misplaced_observations,
    find_missing_planned_observations,
    find_murder_heard_out_of_range,
    find_observation_time_errors,
    find_route_not_followed,
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


def test_sightings_must_be_visible_from_the_observer_room(state: GameState) -> None:
    """From the cellar nobody sees the study; from the library, which overlooks it, they can."""
    from mystery_agents.models.state import HostCharacter, Room, WorldBible

    def room(name: str, connects: list[str]) -> Room:
        return Room(name=name, floor="0", description="x", connects_to=connects, can_see=connects)

    state.world = WorldBible(
        epoch="Modern",
        location_type="Villa",
        location_name="Villa",
        summary="x",
        host=HostCharacter(name="Victim", age=60, role_in_setting="Host"),
        gathering_reason="x",
        rooms=[
            room("Library", ["Study"]),
            room("Study", ["Library"]),
            room("Cellar", []),
            room("Dining room", []),
            room("Garden", []),
            room("Terrace", []),
        ],
    )
    state.personal_timelines["ana"].events[1].info_they_observed = [
        _seen("22:00", "killer", "Study")
    ]
    state.personal_timelines["berta"].events[1].info_they_observed = [
        _seen("22:00", "killer", "Study")
    ]

    issues = find_impossible_sightings(state)

    assert [issue.related_ids for issue in issues] == [["berta"]]


def test_sounds_can_come_from_rooms_out_of_sight(state: GameState) -> None:
    """A scream heard from the cellar is fine even if the study can't be seen."""
    from mystery_agents.models.state import HostCharacter, Room, WorldBible

    state.world = WorldBible(
        epoch="Modern",
        location_type="Villa",
        location_name="Villa",
        summary="x",
        host=HostCharacter(name="Victim", age=60, role_in_setting="Host"),
        gathering_reason="x",
        rooms=[
            Room(name=n, floor="0", description="x", connects_to=[]) for n in ("Study", "Cellar")
        ],
    )
    heard = _seen("22:00", "killer", "Study").model_copy(update={"sense": "sound"})
    state.personal_timelines["berta"].events[1].info_they_observed = [heard]

    assert find_impossible_sightings(state) == []


def test_unaware_killer_looks_from_the_place_their_sheet_shows(state: GameState) -> None:
    """The unaware killer's sheet shows the claimed place, so sightings must fit it."""
    from mystery_agents.models.state import HostCharacter, Room, WorldBible

    state.world = WorldBible(
        epoch="Modern",
        location_type="Villa",
        location_name="Villa",
        summary="x",
        host=HostCharacter(name="Victim", age=60, role_in_setting="Host"),
        gathering_reason="x",
        rooms=[
            Room(name=n, floor="0", description="x", connects_to=[])
            for n in ("Study", "Terrace", "Library", "Cellar", "Dining room", "Garden")
        ],
    )
    state.config.killer_knows_identity = False
    event = state.personal_timelines["killer"].events[1]
    event.claimed_location = "Terrace"
    event.info_they_observed = [_seen("22:00", "ana", "Study")]

    issues = find_impossible_sightings(state)

    assert [issue.related_ids for issue in issues] == [["killer"]]


def test_suspects_must_be_where_shared_events_put_them(state: GameState) -> None:
    """Dinner at 21:30 with Ana: her sheet can't have her elsewhere; the finder must be there too."""
    from mystery_agents.models.state import (
        GlobalEvent,
        GlobalTimeline,
        HostCharacter,
        MurderEvent,
        Room,
        TimeBlock,
        WorldBible,
    )

    state.world = WorldBible(
        epoch="Modern",
        location_type="Villa",
        location_name="Villa",
        summary="x",
        host=HostCharacter(name="Victim", age=60, role_in_setting="Host"),
        gathering_reason="x",
        rooms=[
            Room(name=n, floor="0", description="x", connects_to=[])
            for n in ("Dining room", "Library", "Study")
        ],
    )
    state.timeline_global = GlobalTimeline(
        time_blocks=[
            TimeBlock(
                start="21:00",
                end="22:00",
                events=[
                    GlobalEvent(
                        time_approx="21:30",
                        description="Dinner",
                        character_ids_involved=["ana", "berta"],
                        room_id="Dining room",
                    )
                ],
            )
        ],
        murder_event=MurderEvent(
            time="22:10",
            room="Study",
            what_happens="x",
            heard_elsewhere="",
            heard_in=[],
            scene_when_found="x",
            discovered_at="22:15",
            discovered_by=["killer"],
        ),
    )

    issues = find_absent_from_shared_events(state)

    # Ana and Berta are in the dining room at 21:30; the killer is in the study at 22:15
    assert issues == []

    state.timeline_global.murder_event.discovered_by = ["ana"]
    assert [issue.related_ids for issue in find_absent_from_shared_events(state)] == [["ana"]]


def test_the_murder_is_only_heard_where_it_reaches(state: GameState) -> None:
    """A thud heard in the library is fine; heard from the cellar it isn't."""
    from mystery_agents.models.state import (
        GlobalTimeline,
        HostCharacter,
        MurderEvent,
        Room,
        WorldBible,
    )

    state.world = WorldBible(
        epoch="Modern",
        location_type="Villa",
        location_name="Villa",
        summary="x",
        host=HostCharacter(name="Victim", age=60, role_in_setting="Host"),
        gathering_reason="x",
        rooms=[
            Room(name=n, floor="0", description="x", connects_to=[])
            for n in ("Study", "Library", "Cellar")
        ],
    )
    state.timeline_global = GlobalTimeline(
        time_blocks=[],
        murder_event=MurderEvent(
            time="22:00",
            room="Study",
            what_happens="x",
            heard_elsewhere="A thud",
            heard_in=["Library"],
            scene_when_found="x",
            discovered_at="22:10",
            discovered_by=[],
        ),
    )
    thud = _seen("22:00", "", "Study").model_copy(update={"sense": "sound"})
    state.personal_timelines["ana"].events[1].info_they_observed = [thud]
    state.personal_timelines["berta"].events[1].info_they_observed = [thud]

    issues = find_murder_heard_out_of_range(state)

    assert [issue.related_ids for issue in issues] == [["berta"]]


def _plan(holders: list[str]) -> DeductionPlan:
    def obs(holder: str) -> EvidencePiece:
        return EvidencePiece(
            kind="observation", holder_id=holder, about_id="killer", time="22:00", description="x"
        )

    return DeductionPlan(
        breaking_pieces=[obs(holder) for holder in holders],
        means_piece=EvidencePiece(
            kind="clue", holder_id="", about_id="killer", time="21:00", description="x"
        ),
        red_herrings=[],
    )


def test_killer_must_follow_the_route(state: GameState) -> None:
    """The killer's real place at each route step is the route's room."""
    from mystery_agents.models.state import HostCharacter, Room, RouteStep, WorldBible

    state.world = WorldBible(
        epoch="Modern",
        location_type="Villa",
        location_name="Villa",
        summary="x",
        host=HostCharacter(name="Victim", age=60, role_in_setting="Host"),
        gathering_reason="x",
        rooms=[
            Room(name=n, floor="0", description="x", connects_to=[]) for n in ("Study", "Library")
        ],
    )
    assert state.killer_selection is not None
    state.killer_selection.route = [RouteStep(time="22:00", room="Study", action="x")]
    assert find_route_not_followed(state) == []

    state.killer_selection.route = [RouteStep(time="22:00", room="Library", action="x")]
    assert [issue.related_ids for issue in find_route_not_followed(state)] == [["killer"]]


def test_planned_observations_must_be_on_the_holder_sheet(state: GameState) -> None:
    """A sighting the plan gives Ana must appear on Ana's sheet near the planned time."""
    assert state.killer_selection is not None
    state.killer_selection.deduction_plan = _plan(["ana"])

    assert [issue.related_ids for issue in find_missing_planned_observations(state)] == [["ana"]]

    state.personal_timelines["ana"].events[1].info_they_observed = [
        _seen("22:05", "killer", "Study")
    ]
    assert find_missing_planned_observations(state) == []


def test_planned_witnesses_keep_their_sightings(state: GameState) -> None:
    """If the plan gives three players a sighting, none of them loses it."""
    assert state.killer_selection is not None
    state.killer_selection.deduction_plan = _plan(["ana", "berta", "carla"])
    for witness in ("ana", "berta", "carla"):
        state.personal_timelines[witness].events[1].info_they_observed = [
            _seen("22:00", "killer", "Study")
        ]

    assert drop_excess_killer_sightings(state) == []


def test_plans_with_sightings_through_walls_are_rejected(state: GameState) -> None:
    """A planned sighting of the killer must come from a room with a view of their route."""
    from mystery_agents.models.state import HostCharacter, Room, RouteStep, WorldBible

    state.world = WorldBible(
        epoch="Modern",
        location_type="Villa",
        location_name="Villa",
        summary="x",
        host=HostCharacter(name="Victim", age=60, role_in_setting="Host"),
        gathering_reason="x",
        rooms=[
            Room(name="Study", floor="0", description="x", connects_to=[]),
            Room(name="Garden", floor="0", description="x", connects_to=[], can_see=["Study"]),
            Room(name="Cellar", floor="0", description="x", connects_to=[]),
        ],
    )
    assert state.killer_selection is not None
    state.killer_selection.route = [RouteStep(time="21:55-22:10", room="Study", action="x")]
    plan = _plan(["ana"])
    plan.breaking_pieces[0].from_room = "Garden"
    state.killer_selection.deduction_plan = plan
    assert find_impossible_planned_sightings(state) == []

    plan.breaking_pieces[0].from_room = "Cellar"
    assert len(find_impossible_planned_sightings(state)) == 1
