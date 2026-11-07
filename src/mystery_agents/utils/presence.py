"""The presence grid: where every suspect is, and who can see whom from there.

Sheets written in batches each placed people on their own, and the measured games
kept 4 to 8 contradictions of the kind "Javier sees Carmen in the patio, Carmen's
sheet puts her in the dining room". Once positions are fixed for everybody, code
works out which sightings are possible, and the sheet writer only tells them.
"""

from dataclasses import dataclass

from mystery_agents.models.state import (
    EvidencePiece,
    GameState,
    Observation,
    PersonalTimeline,
    PresenceSlot,
    Room,
)
from mystery_agents.utils.timeline_checks import (
    PLANNED_TIME_SLACK_MINUTES,
    TIME_TOLERANCE_MINUTES,
    minutes_in,
    plan_pieces,
)


@dataclass(frozen=True)
class Sighting:
    """A stretch of time when one suspect can see another."""

    observer_id: str
    observed_id: str
    start: int
    end: int
    from_room: str
    room: str


def clock(minute: int) -> str:
    """HH:MM for minutes since midnight."""
    return f"{minute // 60:02d}:{minute % 60:02d}"


def slot_spans(slots: list[PresenceSlot]) -> list[tuple[int, int] | None]:
    """Time range of each slot in minutes; a single time lasts until the next slot."""
    starts = [minutes_in(slot.time) for slot in slots]
    spans: list[tuple[int, int] | None] = []
    for index, times in enumerate(starts):
        if not times:
            spans.append(None)
        elif len(times) >= 2:
            spans.append((times[0], times[1]))
        else:
            following = next((later[0] for later in starts[index + 1 :] if later), times[0])
            spans.append((times[0], following))
    return spans


def _slots(state: GameState) -> dict[str, list[PresenceSlot]]:
    return {s.character_id: s.slots for s in state.presence.suspects} if state.presence else {}


def slot_at(slots: list[PresenceSlot], minute: int) -> PresenceSlot | None:
    """The slot covering a minute; the later one when two meet at that minute."""
    found = None
    for slot, span in zip(slots, slot_spans(slots), strict=True):
        if span and span[0] <= minute <= span[1]:
            found = slot
    return found


def rooms_at(slots: list[PresenceSlot], minute: int) -> list[str]:
    """
    Rooms someone is in around a minute, with the usual few minutes of slack.

    "At 21:30 everyone moves to the dining room" is met by a slot there from 21:32,
    as the sheet checks already accept; demanding the exact minute kept flagging grids
    that told the evening right.
    """
    return [
        slot.room
        for slot, span in zip(slots, slot_spans(slots), strict=True)
        if span and span[0] - TIME_TOLERANCE_MINUTES <= minute <= span[1] + TIME_TOLERANCE_MINUTES
    ]


def _hides(slot: PresenceSlot) -> bool:
    return bool(slot.claimed_room) and slot.claimed_room != slot.room


def _visible_from(rooms: dict[str, Room], room: str) -> set[str]:
    known = rooms.get(room)
    return {room, *known.can_see} if known else {room}


def _observation_pieces(state: GameState) -> list[EvidencePiece]:
    plan = state.killer_selection.deduction_plan if state.killer_selection else None
    return [piece for piece in plan_pieces(plan) if piece.kind == "observation"] if plan else []


def _planned_at(state: GameState, observer_id: str, observed_id: str, minute: int) -> bool:
    return any(
        piece.holder_id == observer_id
        and piece.about_id == observed_id
        and (times := minutes_in(piece.time))
        and abs(times[0] - minute) <= PLANNED_TIME_SLACK_MINUTES
        for piece in _observation_pieces(state)
    )


def _unaware_killer(state: GameState) -> str | None:
    if state.killer_selection and not state.config.killer_knows_identity:
        return state.killer_selection.killer_id
    return None


def _overlaps(state: GameState) -> list[tuple[Sighting, bool]]:
    """
    Every stretch when one suspect has another in view, by the grid and the floor plan.

    Each comes with whether it is the hidden killer seen by someone the deduction plan
    doesn't make a witness. An unaware killer sees nothing from a place their own sheet
    says they weren't, so those stretches are left out.
    """
    rooms = {room.name: room for room in state.world.rooms} if state.world else {}
    grid = _slots(state)
    killer_id = state.killer_selection.killer_id if state.killer_selection else None
    unaware = _unaware_killer(state)
    found = []
    for observer_id, observer_slots in grid.items():
        for observer_slot, observer_span in zip(
            observer_slots, slot_spans(observer_slots), strict=True
        ):
            if not observer_span or (observer_id == unaware and _hides(observer_slot)):
                continue
            visible = _visible_from(rooms, observer_slot.room)
            for observed_id, observed_slots in grid.items():
                if observed_id == observer_id:
                    continue
                for slot, span in zip(observed_slots, slot_spans(observed_slots), strict=True):
                    if not span or slot.room not in visible:
                        continue
                    start, end = max(span[0], observer_span[0]), min(span[1], observer_span[1])
                    if start >= end:
                        continue
                    unplanned = (
                        observed_id == killer_id
                        and _hides(slot)
                        and not any(
                            _planned_at(state, observer_id, observed_id, m)
                            for m in range(start, end + 1)
                        )
                    )
                    sighting = Sighting(
                        observer_id, observed_id, start, end, observer_slot.room, slot.room
                    )
                    found.append((sighting, unplanned))
    return found


def possible_sightings(state: GameState) -> list[Sighting]:
    """
    Every stretch when one suspect can see another and may say so on their sheet.

    The hidden killer is left out except for the deduction plan's witnesses: more
    witnesses than planned solve the case alone.
    """
    return [sighting for sighting, unplanned in _overlaps(state) if not unplanned]


def presence_problems(state: GameState) -> list[str]:
    """
    Find grids that can't be told consistently.

    Checked: every suspect is there and their slots follow each other through
    connected rooms over the whole evening; the shared events, the discovery of the
    body and the killer's route put people where the grid does; planned sightings
    are made from their room and can see the person they are about; the killer
    claims to be elsewhere at the time of the murder.
    """
    if not state.presence or not state.world:
        return ["No presence grid."]
    rooms = {room.name: room for room in state.world.rooms}
    names = {char.id: char.name for char in state.characters}
    grid = _slots(state)
    problems = [f"{name} has no slots." for char_id, name in names.items() if not grid.get(char_id)]

    for char_id, slots in grid.items():
        name = names.get(char_id, char_id)
        spans = slot_spans(slots)
        for index, (slot, span) in enumerate(zip(slots, spans, strict=True)):
            if span is None:
                problems.append(f"{name}'s slot '{slot.time}' has no HH:MM-HH:MM time.")
                continue
            if index == 0:
                continue
            previous, before = slots[index - 1], spans[index - 1]
            if before and abs(span[0] - before[1]) > TIME_TOLERANCE_MINUTES:
                problems.append(
                    f"{name}'s slot {slot.time} doesn't start when {previous.time} ends; "
                    "make the slots consecutive."
                )
            known = rooms.get(previous.room)
            if known and slot.room != previous.room and slot.room not in known.connects_to:
                problems.append(
                    f"{name} goes from {previous.room} to {slot.room} at {slot.time}, but those "
                    "rooms don't connect. Add a slot for the room they pass through."
                )

    timeline = state.timeline_global
    if timeline:
        shared = [
            (event.time_approx, event.room_id, event.character_ids_involved, event.description)
            for block in timeline.time_blocks
            for event in block.events
            if event.room_id in rooms
        ]
        murder = timeline.murder_event
        shared.append(
            (murder.discovered_at, murder.room, murder.discovered_by, "the body is found")
        )
        for time, room, char_ids, what in shared:
            times = minutes_in(time)
            for char_id in char_ids if times else []:
                there = rooms_at(grid.get(char_id, []), times[0])
                if there and room not in there:
                    problems.append(
                        f"{names.get(char_id, char_id)} must be in {room} at {time} ({what}), "
                        f"but the grid has them in {' / '.join(there)}."
                    )
        start = [m for block in timeline.time_blocks[:1] for m in minutes_in(block.start)]
        end = minutes_in(murder.discovered_at)
        for char_id, slots in grid.items():
            timed = [span for span in slot_spans(slots) if span]
            if not timed:
                continue
            if start and timed[0][0] > start[0] + TIME_TOLERANCE_MINUTES:
                problems.append(
                    f"{names.get(char_id, char_id)}'s slots start at {clock(timed[0][0])}; "
                    f"start them at {clock(start[0])}."
                )
            if end and timed[-1][1] < end[0]:
                problems.append(
                    f"{names.get(char_id, char_id)}'s slots end at {clock(timed[-1][1])}; "
                    f"continue them until {clock(end[0])}, when the body is found."
                )

    selection = state.killer_selection
    if selection:
        killer_slots = grid.get(selection.killer_id, [])
        for step in selection.route:
            times = minutes_in(step.time)
            on_route = rooms_at(killer_slots, times[0]) if times else []
            if on_route and step.room in rooms and step.room not in on_route:
                problems.append(
                    f"The killer's route has them in {step.room} at {step.time}, but the grid has "
                    f"them in {' / '.join(on_route)}."
                )
        murder_at = minutes_in(timeline.murder_event.time) if timeline else []
        at_murder = slot_at(killer_slots, murder_at[0]) if murder_at else None
        if at_murder and not _hides(at_murder):
            problems.append(
                f"At the time of the murder the killer's slot {at_murder.time} has no claimed_room: "
                "say where they will claim to have been."
            )

    for sighting, unplanned in _overlaps(state):
        # Leaving it out of their sheet reads as blindness at the table: "you were
        # overlooking the study and saw nobody leave?"
        if unplanned and sighting.end - sighting.start > TIME_TOLERANCE_MINUTES:
            problems.append(
                f"{names.get(sighting.observer_id, sighting.observer_id)} is in "
                f"{sighting.from_room} from {clock(sighting.start)} to {clock(sighting.end)}, "
                f"overlooking the killer in {sighting.room} during their hidden moment. Move them somewhere "
                "without a view of the killer's route then."
            )

    for piece in _observation_pieces(state):
        times = minutes_in(piece.time)
        holder = rooms_at(grid.get(piece.holder_id, []), times[0]) if times else []
        observed = rooms_at(grid.get(piece.about_id, []), times[0]) if times else []
        holder_name = names.get(piece.holder_id, piece.holder_id)
        if holder and piece.from_room and piece.from_room not in holder:
            problems.append(
                f"The plan has {holder_name} seeing something from {piece.from_room} at "
                f"{piece.time}, but the grid has them in {' / '.join(holder)}."
            )
        if piece.holder_id == _unaware_killer(state) and times:
            # Their sheet only shows the account, so they can only see from where it says
            hiding = [
                slot
                for slot, span in zip(
                    grid.get(piece.holder_id, []),
                    slot_spans(grid.get(piece.holder_id, [])),
                    strict=True,
                )
                if span and span[0] <= times[0] <= span[1] and _hides(slot)
            ]
            if hiding:
                problems.append(
                    f"The plan gives the killer an observation at {piece.time}, while they hide "
                    f"where they are ({hiding[0].time}). Move it to a moment they really are where "
                    "they say, or end that slot earlier."
                )
        view = {room for here in holder for room in _visible_from(rooms, here)}
        if holder and observed and not view & set(observed):
            problems.append(
                f"The plan has {holder_name} seeing {names.get(piece.about_id, piece.about_id)} "
                f"at {piece.time}, but from {' / '.join(holder)} they can't see "
                f"{' / '.join(observed)}."
            )
    return problems


def format_presence(state: GameState) -> str:
    """The grid for a prompt, one block per suspect."""
    if not state.presence:
        return "No presence grid"
    names = {char.id: char.name for char in state.characters}
    blocks = []
    for suspect in state.presence.suspects:
        lines = [
            f"{names.get(suspect.character_id, suspect.character_id)} ({suspect.character_id}):"
        ]
        for slot in suspect.slots:
            claim = f" [claims: {slot.claimed_room}]" if _hides(slot) else ""
            lines.append(f"- {slot.time} @ {slot.room}{claim}: {slot.doing}")
        blocks.append("\n".join(lines))
    return "\n\n".join(blocks)


def format_sightings(state: GameState) -> str:
    """Who each suspect can see, when and where, for a prompt."""
    names = {char.id: char.name for char in state.characters}
    by_observer: dict[str, list[str]] = {}
    for s in possible_sightings(state):
        by_observer.setdefault(s.observer_id, []).append(
            f"  - {clock(s.start)}-{clock(s.end)} from {s.from_room}: "
            f"{names.get(s.observed_id, s.observed_id)} in {s.room}"
        )
    if not by_observer:
        return "Nobody can see anybody"
    return "\n".join(
        f"{names.get(observer, observer)} can see:\n" + "\n".join(lines)
        for observer, lines in by_observer.items()
    )


def apply_presence(timeline: PersonalTimeline, slots: list[PresenceSlot]) -> bool:
    """
    Give a sheet's events the grid's times and places, one event per slot.

    The writer is asked for one event per slot; when it returns that, the grid wins
    over whatever it wrote for times and places.

    Returns:
        Whether the events matched the slots and were aligned
    """
    if len(timeline.events) != len(slots):
        return False
    for event, slot in zip(timeline.events, slots, strict=True):
        event.time = slot.time
        event.location = slot.room
        event.claimed_location = slot.claimed_room if _hides(slot) else ""
    return True


def _matching_sighting(
    sightings: list[Sighting], observer_id: str, seen: Observation
) -> Sighting | None:
    times = minutes_in(seen.time)
    if not times:
        return None
    return next(
        (
            s
            for s in sightings
            if s.observer_id == observer_id
            and s.observed_id == seen.observed_character_id
            and s.start - TIME_TOLERANCE_MINUTES <= times[0] <= s.end + TIME_TOLERANCE_MINUTES
        ),
        None,
    )


def align_observations(state: GameState) -> list[str]:
    """
    Keep only the sightings the grid allows, placed where the observed person is.

    A sighting of someone who wasn't visible then is the contradiction players find
    when they compare sheets; with the grid, code knows it for certain. Sounds and
    observations about nobody in particular are left alone.

    Returns:
        IDs of the suspects whose observations were removed
    """
    if not state.presence:
        return []
    sightings = possible_sightings(state)
    suspects = {s.character_id for s in state.presence.suspects}
    trimmed = []
    for char_id, timeline in state.personal_timelines.items():
        removed = False
        for event in timeline.events:
            kept = []
            for seen in event.info_they_observed:
                if seen.sense != "sight" or seen.observed_character_id not in suspects:
                    kept.append(seen)
                    continue
                match = _matching_sighting(sightings, char_id, seen)
                if match:
                    seen.location = match.room
                    kept.append(seen)
                else:
                    removed = True
            event.info_they_observed = kept
        if removed:
            trimmed.append(char_id)
    return trimmed
