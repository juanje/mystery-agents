"""Unit tests for the presence grid checks and the sightings it allows."""

import pytest

from mystery_agents.models.state import (
    CharacterSpec,
    DeductionPlan,
    EvidencePiece,
    GameConfig,
    GameState,
    GlobalEvent,
    GlobalTimeline,
    HostCharacter,
    KillerSelection,
    MetaInfo,
    MurderEvent,
    Observation,
    PersonalEvent,
    PersonalTimeline,
    PlayerConfig,
    PresenceGrid,
    PresenceSlot,
    Room,
    RouteStep,
    SuspectPresence,
    TimeBlock,
    WorldBible,
)
from mystery_agents.utils.constants import TEST_DEFAULT_DURATION, TEST_DEFAULT_PLAYERS
from mystery_agents.utils.presence import (
    align_observations,
    apply_presence,
    possible_sightings,
    presence_problems,
)

NAMES = {"killer": "Diego Ojeda", "ana": "Ana Ruiz", "berta": "Berta Sosa"}


def _slot(time: str, room: str, claimed: str = "") -> PresenceSlot:
    return PresenceSlot(time=time, room=room, claimed_room=claimed, doing="x")


@pytest.fixture
def state() -> GameState:
    """
    Dinner at 21:00; the killer slips into the study at 21:30 claiming the library.

    The library overlooks the study; the cellar sees nothing. Ana reads in the
    library, Berta is in the cellar. The body is found at 22:00 in the study.
    """
    rooms = [
        Room(name="Dining room", floor="0", description="x", connects_to=["Library", "Cellar"]),
        Room(
            name="Library",
            floor="0",
            description="x",
            connects_to=["Dining room", "Study"],
            can_see=["Study"],
        ),
        Room(name="Study", floor="0", description="x", connects_to=["Library"]),
        Room(name="Cellar", floor="-1", description="x", connects_to=["Dining room"]),
    ]
    grid = {
        "killer": [
            _slot("21:00-21:30", "Dining room"),
            _slot("21:30-21:35", "Library"),
            _slot("21:35-21:50", "Study", claimed="Library"),
            _slot("21:50-22:00", "Library"),
            _slot("22:00-22:05", "Study"),
        ],
        "ana": [
            _slot("21:00-21:30", "Dining room"),
            _slot("21:30-21:55", "Library"),
            _slot("21:55-22:05", "Study"),
        ],
        "berta": [
            _slot("21:00-21:30", "Dining room"),
            _slot("21:30-21:55", "Cellar"),
            _slot("21:55-21:58", "Dining room"),
            _slot("21:58-21:59", "Library"),
            _slot("21:59-22:05", "Study"),
        ],
    }
    return GameState(
        meta=MetaInfo(),
        config=GameConfig(
            players=PlayerConfig(total=TEST_DEFAULT_PLAYERS),
            duration_minutes=TEST_DEFAULT_DURATION,
            difficulty="medium",
        ),
        world=WorldBible(
            epoch="Modern",
            location_type="Villa",
            location_name="Villa",
            summary="x",
            host=HostCharacter(name="Victim", age=60, role_in_setting="Host"),
            gathering_reason="x",
            rooms=rooms,
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
        timeline_global=GlobalTimeline(
            time_blocks=[
                TimeBlock(
                    start="21:00",
                    end="22:00",
                    events=[
                        GlobalEvent(
                            time_approx="21:10",
                            description="Dinner",
                            character_ids_involved=list(NAMES),
                            room_id="Dining room",
                        )
                    ],
                )
            ],
            murder_event=MurderEvent(
                time="21:40",
                room="Study",
                what_happens="x",
                heard_elsewhere="x",
                heard_in=[],
                scene_when_found="x",
                discovered_at="22:00",
                discovered_by=["ana"],
            ),
        ),
        killer_selection=KillerSelection(
            killer_id="killer",
            rationale="r",
            truth_narrative="t",
            route=[RouteStep(time="21:35-21:50", room="Study", action="x")],
        ),
        presence=PresenceGrid(
            suspects=[
                SuspectPresence(character_id=char_id, slots=slots)
                for char_id, slots in grid.items()
            ]
        ),
    )


def _slots(state: GameState, char_id: str) -> list[PresenceSlot]:
    assert state.presence is not None
    return next(s.slots for s in state.presence.suspects if s.character_id == char_id)


def test_consistent_grid_passes(state: GameState) -> None:
    """The fixture follows the shared events, the route and the floor plan."""
    assert presence_problems(state) == []


def test_moves_between_rooms_that_dont_connect_are_flagged(state: GameState) -> None:
    """From the cellar to the library takes a pass through the dining room."""
    _slots(state, "berta")[2] = _slot("21:55-21:58", "Library")

    problems = presence_problems(state)

    assert any("don't connect" in problem for problem in problems)


def test_people_must_be_where_shared_events_and_the_route_put_them(state: GameState) -> None:
    """Dinner is in the dining room; the killer's route is in the study."""
    _slots(state, "ana")[0] = _slot("21:00-21:30", "Library")
    _slots(state, "killer")[2] = _slot("21:35-21:50", "Library", claimed="Dining room")

    problems = presence_problems(state)

    assert any("Ana Ruiz must be in Dining room" in problem for problem in problems)
    assert any("route has them in Study" in problem for problem in problems)


def test_shared_events_allow_a_few_minutes_of_slack(state: GameState) -> None:
    """Sitting down to dinner at 21:03 meets a 21:00 dinner; at 21:20 it doesn't."""
    assert state.timeline_global is not None
    dinner = state.timeline_global.time_blocks[0].events[0]
    _slots(state, "ana")[0] = _slot("21:03-21:30", "Dining room")
    _slots(state, "ana").insert(0, _slot("21:00-21:03", "Library"))
    dinner.time_approx = "21:00"
    assert not any("Ana Ruiz must be" in p for p in presence_problems(state))

    dinner.time_approx = "21:40"
    assert any("Ana Ruiz must be" in p for p in presence_problems(state))


def test_killer_must_claim_another_room_at_the_murder(state: GameState) -> None:
    """Without a claimed room, the killer's sheet would admit being at the scene."""
    _slots(state, "killer")[2] = _slot("21:35-21:50", "Study")

    assert any("no claimed_room" in problem for problem in presence_problems(state))


def test_planned_sightings_must_be_possible(state: GameState) -> None:
    """A plan that has Berta see the killer from the cellar can't be told."""
    assert state.killer_selection is not None
    piece = EvidencePiece(
        kind="observation",
        holder_id="berta",
        about_id="killer",
        time="21:40",
        from_room="Cellar",
        description="x",
    )
    state.killer_selection.deduction_plan = DeductionPlan(
        breaking_pieces=[piece], means_piece=piece, red_herrings=[]
    )

    assert any("they can't see" in problem for problem in presence_problems(state))


def test_hidden_killer_is_only_visible_to_planned_witnesses(state: GameState) -> None:
    """Ana overlooks the study, but only a planned sighting lets her see the killer there."""
    hidden = [
        s
        for s in possible_sightings(state)
        if s.observed_id == "killer" and s.room == "Study" and s.start < 21 * 60 + 50
    ]
    assert hidden == []

    assert state.killer_selection is not None
    piece = EvidencePiece(
        kind="observation",
        holder_id="ana",
        about_id="killer",
        time="21:40",
        from_room="Library",
        description="x",
    )
    state.killer_selection.deduction_plan = DeductionPlan(
        breaking_pieces=[piece], means_piece=piece, red_herrings=[]
    )
    assert any(
        s.observer_id == "ana" and s.observed_id == "killer" and s.room == "Study"
        for s in possible_sightings(state)
    )


def test_sightings_the_grid_rules_out_are_removed(state: GameState) -> None:
    """Berta can't see Ana from the cellar; Ana can see Berta at dinner, placed by the grid."""

    def seen(time: str, who: str) -> Observation:
        return Observation(time=time, observed_character_id=who, location="Patio", text="x")

    state.personal_timelines = {
        "berta": PersonalTimeline(
            character_id="berta",
            subjective_narrative="",
            events=[
                PersonalEvent(
                    time="21:30-21:55",
                    location="Cellar",
                    what_they_really_did="x",
                    what_they_will_tell_others="x",
                    info_they_observed=[seen("21:40", "ana")],
                )
            ],
        ),
        "ana": PersonalTimeline(
            character_id="ana",
            subjective_narrative="",
            events=[
                PersonalEvent(
                    time="21:00-21:30",
                    location="Dining room",
                    what_they_really_did="x",
                    what_they_will_tell_others="x",
                    info_they_observed=[seen("21:10", "berta")],
                )
            ],
        ),
    }

    assert align_observations(state) == ["berta"]
    assert state.personal_timelines["berta"].events[0].info_they_observed == []
    assert state.personal_timelines["ana"].events[0].info_they_observed[0].location == (
        "Dining room"
    )


def test_sheet_events_take_the_grid_times_and_places(state: GameState) -> None:
    """One event per slot: the grid wins; any other count is left as written."""
    slots = _slots(state, "killer")
    events = [
        PersonalEvent(
            time="?", location="Hall", what_they_really_did="x", what_they_will_tell_others="x"
        )
        for _ in slots
    ]
    timeline = PersonalTimeline(character_id="killer", events=events, subjective_narrative="")

    assert apply_presence(timeline, slots)
    assert [e.location for e in timeline.events] == [s.room for s in slots]
    assert timeline.events[2].claimed_location == "Library"

    timeline.events.pop()
    assert not apply_presence(timeline, slots)


def test_unaware_killer_observes_only_where_their_account_puts_them(state: GameState) -> None:
    """A planned observation inside the killer's hidden slot would be removed from their sheet."""
    assert state.killer_selection is not None
    piece = EvidencePiece(
        kind="observation",
        holder_id="killer",
        about_id="ana",
        time="21:40",
        from_room="Study",
        description="x",
    )
    state.killer_selection.deduction_plan = DeductionPlan(
        breaking_pieces=[], means_piece=piece, red_herrings=[]
    )

    assert any("while they hide" in problem for problem in presence_problems(state))

    piece.time = "21:52"
    piece.from_room = "Library"
    assert not any("while they hide" in problem for problem in presence_problems(state))
