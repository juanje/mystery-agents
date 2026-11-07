"""Deterministic checks of the suspects' personal timelines.

The LLM reviews kept missing these in real runs: suspects observing from places they
had already left, people seen where their own timeline doesn't put them, and every
player witnessing the killer, which makes the game obvious.
"""

import re
import unicodedata
from collections.abc import Iterator

from mystery_agents.models.state import (
    DeductionPlan,
    EvidencePiece,
    GameState,
    Observation,
    PersonalEvent,
    Room,
    ValidationIssue,
)

# Minutes of slack around event boundaries: people move between rooms
TIME_TOLERANCE_MINUTES = 5
# Most innocent suspects allowed to see the killer while the killer hides something.
# More than this and players don't need to compare testimonies at all.
MAX_KILLER_WITNESSES = {"easy": 3, "medium": 2, "hard": 2}
# A planned observation may be written a little earlier or later than planned
PLANNED_TIME_SLACK_MINUTES = 10
# Innocent suspects kept hiding something when some must become truthful
MAX_INNOCENTS_HIDING = 3
# Words too generic to tell two places apart ("the corridor of the upper floor")
MIN_PLACE_WORD_LENGTH = 4

_TIME_PATTERN = re.compile(r"(\d{1,2}):(\d{2})")


def minutes_in(text: str) -> list[int]:
    """All HH:MM times in a text, as minutes since midnight."""
    return [int(hours) * 60 + int(mins) for hours, mins in _TIME_PATTERN.findall(text)]


def event_ranges(events: list[PersonalEvent]) -> list[tuple[int, int] | None]:
    """
    Time range of each event, in minutes.

    A single time lasts until the next event starts. Events without a readable
    time get None and are skipped by the checks.
    """
    starts = [minutes_in(event.time) for event in events]
    ranges: list[tuple[int, int] | None] = []
    for index, times in enumerate(starts):
        if not times:
            ranges.append(None)
            continue
        if len(times) >= 2:
            ranges.append((times[0], times[1]))
            continue
        following = next((later[0] for later in starts[index + 1 :] if later), None)
        ranges.append((times[0], following if following is not None else times[0]))
    return ranges


def _covers(span: tuple[int, int], minute: int) -> bool:
    start, end = span
    return start - TIME_TOLERANCE_MINUTES <= minute <= end + TIME_TOLERANCE_MINUTES


def _place_words(place: str) -> set[str]:
    """Distinctive words of a place name, without accents or case."""
    plain = unicodedata.normalize("NFKD", place).encode("ascii", "ignore").decode().casefold()
    return {word for word in re.findall(r"\w+", plain) if len(word) >= MIN_PLACE_WORD_LENGTH}


def same_place(first: str, second: str) -> bool:
    """Whether two place names plausibly refer to the same place."""
    first_words, second_words = _place_words(first), _place_words(second)
    if not first_words or not second_words:
        return True
    return bool(first_words & second_words)


def find_observation_time_errors(state: GameState) -> list[ValidationIssue]:
    """Find observations made at a time the observer wasn't in that moment of their evening."""
    names = {char.id: char.name for char in state.characters}
    issues = []
    for char_id, timeline in state.personal_timelines.items():
        for event, span in zip(timeline.events, event_ranges(timeline.events), strict=True):
            if span is None:
                continue
            for seen in event.info_they_observed:
                times = minutes_in(seen.time)
                if times and not _covers(span, times[0]):
                    issues.append(
                        ValidationIssue(
                            type="timeline_conflict",
                            description=(
                                f"{names.get(char_id, char_id)}'s sheet lists '{seen.text}' at "
                                f"{seen.time} under their moment {event.time} @ {event.location}, "
                                "so they observe it before arriving or after leaving. Move the "
                                "observation to the moment they were really there, or change it."
                            ),
                            related_ids=[char_id],
                            fix_in="timelines",
                        )
                    )
    return issues


def _places_around(timeline_events: list[PersonalEvent], minute: int) -> list[str]:
    """
    Where someone could be seen at a time: the moments covering it and the ones next to them.

    People are often seen on their way between two places ("leaving the dining room
    for the library"), so the neighbouring moments count too.
    """
    spans = event_ranges(timeline_events)
    covering = [index for index, span in enumerate(spans) if span and _covers(span, minute)]
    nearby = {n for index in covering for n in (index - 1, index, index + 1)}
    return [timeline_events[n].location for n in sorted(nearby) if 0 <= n < len(timeline_events)]


def find_misplaced_observations(state: GameState) -> list[ValidationIssue]:
    """Find suspects observed somewhere their own timeline says they weren't."""
    names = {char.id: char.name for char in state.characters}
    issues = []
    for char_id, timeline in state.personal_timelines.items():
        for event in timeline.events:
            for seen in event.info_they_observed:
                observed = state.personal_timelines.get(seen.observed_character_id)
                times = minutes_in(seen.time)
                if not observed or not times or seen.observed_character_id == char_id:
                    continue
                places = _places_around(observed.events, times[0])
                if not places or any(
                    same_place(seen.location, place) or same_place(seen.text, place)
                    for place in places
                ):
                    continue
                observed_name = names.get(seen.observed_character_id, seen.observed_character_id)
                issues.append(
                    ValidationIssue(
                        type="timeline_conflict",
                        description=(
                            f"{names.get(char_id, char_id)} sees {observed_name} at "
                            f"{seen.location} at {seen.time} ('{seen.text}'), but "
                            f"{observed_name}'s own timeline puts them at "
                            f"{' / '.join(dict.fromkeys(places))} around then. Make the "
                            f"observation match where {observed_name} really was."
                        ),
                        related_ids=[char_id],
                        fix_in="timelines",
                    )
                )
    return issues


def _impossible_sightings(
    state: GameState,
) -> Iterator[tuple[str, PersonalEvent, Observation, Room]]:
    """
    Observations of a room the observer couldn't see from where they were.

    Players judge sightings by the plan on their sheets: someone sees their own room
    and the rooms listed as visible from it. Anything else is a sighting they will
    reject when they compare sheets, even if a door connects the two rooms.

    Yields:
        (observer ID, the observer's event, the observation, the room they look from)
    """
    if not state.world or not state.world.rooms:
        return
    rooms = {room.name: room for room in state.world.rooms}
    # An unaware killer's sheet shows their claimed places, so that's where they look from
    unaware_killer = (
        state.killer_selection.killer_id
        if state.killer_selection and not state.config.killer_knows_identity
        else None
    )
    for char_id, timeline in state.personal_timelines.items():
        for event in timeline.events:
            shown_place = (
                event.claimed_location
                if char_id == unaware_killer and event.claimed_location
                else event.location
            )
            here = rooms.get(shown_place)
            if not here:
                continue
            visible = {here.name, *here.can_see}
            for seen in event.info_they_observed:
                if (
                    seen.sense == "sight"
                    and seen.location in rooms
                    and seen.location not in visible
                ):
                    yield char_id, event, seen, here


def find_impossible_sightings(state: GameState) -> list[ValidationIssue]:
    """Find observations of a room the observer couldn't see from where they were."""
    names = {char.id: char.name for char in state.characters}
    return [
        ValidationIssue(
            type="timeline_conflict",
            description=(
                f"{names.get(char_id, char_id)} is in {here.name} at {event.time} "
                f"but observes something in {seen.location} ('{seen.text}'), "
                f"which can't be seen from there. From {here.name} they can see: "
                f"{', '.join(sorted({here.name, *here.can_see}))}."
            ),
            related_ids=[char_id],
            fix_in="timelines",
        )
        for char_id, event, seen, here in _impossible_sightings(state)
    ]


def _plan_pieces(plan: DeductionPlan) -> list[EvidencePiece]:
    """Every piece of a deduction plan: breaking, means and each red herring's two."""
    return [
        *plan.breaking_pieces,
        plan.means_piece,
        *(entry.suspicion for entry in plan.red_herrings),
        *(entry.clearing for entry in plan.red_herrings),
    ]


def _is_planned(state: GameState, holder_id: str, seen: Observation) -> bool:
    """Whether the deduction plan asks for this observation (same holder, subject and time)."""
    plan = state.killer_selection.deduction_plan if state.killer_selection else None
    times = minutes_in(seen.time)
    if not plan or not times:
        return False
    pieces = _plan_pieces(plan)
    return any(
        piece.kind == "observation"
        and piece.holder_id == holder_id
        and piece.about_id == seen.observed_character_id
        and (planned_at := minutes_in(piece.time))
        and abs(planned_at[0] - times[0]) <= PLANNED_TIME_SLACK_MINUTES
        for piece in pieces
    )


def drop_impossible_sightings(state: GameState) -> list[str]:
    """
    Remove sightings the floor plan rules out, unless the deduction plan needs them.

    Rewrites flagged with these kept them in real runs, and the games shipped with
    sightings through walls that players reject when they read the house plan. An
    unplanned sighting adds colour, not proof, so removing it costs the case nothing.
    Planned ones stay, so the review still asks for them to be made possible.

    Returns:
        IDs of the suspects whose sightings were removed
    """
    found = [
        (char_id, event, seen)
        for char_id, event, seen, _ in _impossible_sightings(state)
        if not _is_planned(state, char_id, seen)
    ]
    for _, event, seen in found:
        event.info_they_observed = [o for o in event.info_they_observed if o is not seen]
    return list(dict.fromkeys(char_id for char_id, _, _ in found))


def _place_at(events: list[PersonalEvent], minute: int) -> list[str]:
    """Where someone really is at a time, by their own timeline."""
    return [
        event.location
        for event, span in zip(events, event_ranges(events), strict=True)
        if span and _covers(span, minute)
    ]


def find_absent_from_shared_events(state: GameState) -> list[ValidationIssue]:
    """
    Find suspects whose sheet puts them elsewhere during a shared event they're part of.

    The global timeline's events and the discovery of the body are what every sheet
    is built around; a sheet that misses one contradicts everyone else's.
    """
    if not state.timeline_global or not state.world or not state.world.rooms:
        return []
    names = {char.id: char.name for char in state.characters}
    murder = state.timeline_global.murder_event
    shared = [
        (event.time_approx, event.room_id, event.character_ids_involved, event.description)
        for block in state.timeline_global.time_blocks
        for event in block.events
        if event.room_id
    ]
    shared.append((murder.discovered_at, murder.room, murder.discovered_by, "the body is found"))
    rooms = {room.name for room in state.world.rooms}
    issues = []
    for time, room, char_ids, description in shared:
        times = minutes_in(time)
        if not times or room not in rooms:
            continue
        for char_id in char_ids:
            timeline = state.personal_timelines.get(char_id)
            if not timeline:
                continue
            places = _place_at(timeline.events, times[0])
            if places and room not in places:
                issues.append(
                    ValidationIssue(
                        type="timeline_conflict",
                        description=(
                            f"The shared timeline has {names.get(char_id, char_id)} in {room} at "
                            f"{time} ({description}), but their sheet puts them in "
                            f"{' / '.join(dict.fromkeys(places))}. Put them in {room} then, "
                            "remembering that moment as everyone else does."
                        ),
                        related_ids=[char_id],
                        fix_in="timelines",
                    )
                )
    return issues


def find_murder_heard_out_of_range(state: GameState) -> list[ValidationIssue]:
    """
    Find suspects who hear the murder from a room the murder event says it doesn't reach.

    The murder event fixes where it can be heard; a sheet that hears it from
    elsewhere tells players that suspect was somewhere else.
    """
    if not state.timeline_global or not state.world or not state.world.rooms:
        return []
    murder = state.timeline_global.murder_event
    times = minutes_in(murder.time)
    if not times:
        return []
    audible = {murder.room, *murder.heard_in}
    names = {char.id: char.name for char in state.characters}
    issues = []
    for char_id, timeline in state.personal_timelines.items():
        for event in timeline.events:
            for seen in event.info_they_observed:
                heard_at = minutes_in(seen.time)
                if (
                    seen.sense == "sound"
                    and heard_at
                    and abs(heard_at[0] - times[0]) <= TIME_TOLERANCE_MINUTES
                    and event.location not in audible
                ):
                    issues.append(
                        ValidationIssue(
                            type="timeline_conflict",
                            description=(
                                f"{names.get(char_id, char_id)} hears the murder from "
                                f"{event.location} ('{seen.text}'), but it can only be heard in "
                                f"{', '.join(sorted(audible))}. Remove it or move them there."
                            ),
                            related_ids=[char_id],
                            fix_in="timelines",
                        )
                    )
    return issues


def _killer_sightings(state: GameState) -> dict[str, list[Observation]]:
    """
    Observations of the killer during the moments they hide, by innocent witness.

    Those are the observations that break the killer's account.
    """
    if not state.killer_selection:
        return {}
    killer_id = state.killer_selection.killer_id
    killer_timeline = state.personal_timelines.get(killer_id)
    if not killer_timeline:
        return {}
    hidden_spans = [
        span
        for event, span in zip(
            killer_timeline.events, event_ranges(killer_timeline.events), strict=True
        )
        if span and event.what_they_will_tell_others != event.what_they_really_did
    ]
    sightings: dict[str, list[Observation]] = {}
    for char_id, timeline in state.personal_timelines.items():
        if char_id == killer_id:
            continue
        found = [
            seen
            for event in timeline.events
            for seen in event.info_they_observed
            if seen.observed_character_id == killer_id
            and (times := minutes_in(seen.time))
            and any(_covers(span, times[0]) for span in hidden_spans)
        ]
        if found:
            sightings[char_id] = found
    return sightings


def _planned_witnesses(state: GameState) -> list[str]:
    """Suspects the deduction plan gives an observation that breaks the killer's account."""
    plan = state.killer_selection.deduction_plan if state.killer_selection else None
    if not plan:
        return []
    return list(
        dict.fromkeys(
            piece.holder_id for piece in plan.breaking_pieces if piece.kind == "observation"
        )
    )


def _witness_limit(state: GameState) -> int:
    return max(MAX_KILLER_WITNESSES.get(state.config.difficulty, 2), len(_planned_witnesses(state)))


def _witnesses_planned_first(state: GameState, witnesses: list[str]) -> list[str]:
    """Order witnesses so the planned ones keep their sightings."""
    planned = _planned_witnesses(state)
    return sorted(witnesses, key=lambda char_id: char_id not in planned)


def drop_excess_killer_sightings(state: GameState) -> list[str]:
    """
    Remove the killer sightings of the witnesses beyond the difficulty's limit.

    Asking the model to give fewer suspects these sightings didn't work: rewritten
    timelines kept them. Removing them keeps the case solvable (the first witnesses
    keep theirs) without letting everyone solve it alone.

    Returns:
        IDs of the suspects whose sightings were removed
    """
    sightings = _killer_sightings(state)
    extra = _witnesses_planned_first(state, list(sightings))[_witness_limit(state) :]
    for char_id in extra:
        removed = {id(seen) for seen in sightings[char_id]}
        for event in state.personal_timelines[char_id].events:
            event.info_they_observed = [
                seen for seen in event.info_they_observed if id(seen) not in removed
            ]
    return extra


def find_too_many_killer_witnesses(state: GameState) -> list[ValidationIssue]:
    """
    Find games where too many suspects see the killer during the moments they hide.

    Spread over a couple of players these sightings make the detective compare
    testimonies; given to everyone they solve the case.
    """
    witnesses = _witnesses_planned_first(state, list(_killer_sightings(state)))
    limit = _witness_limit(state)
    if len(witnesses) <= limit or not state.killer_selection:
        return []
    names = {char.id: char.name for char in state.characters}
    killer_name = names.get(state.killer_selection.killer_id, state.killer_selection.killer_id)
    extra = witnesses[limit:]
    return [
        ValidationIssue(
            type="over_obvious",
            description=(
                f"{len(witnesses)} suspects see {killer_name} during the moments they hide: "
                f"{', '.join(names.get(w, w) for w in witnesses)}. At most {limit} should, each "
                "seeing only part of it, or nobody needs to compare testimonies. Rewrite the "
                f"observations of {', '.join(names.get(w, w) for w in extra)} so they concern "
                f"other suspects or other moments of {killer_name}'s evening."
            ),
            related_ids=extra,
            fix_in="timelines",
        )
    ]


def find_route_not_followed(state: GameState) -> list[ValidationIssue]:
    """Find route steps where the killer's own sheet puts them in another room."""
    selection = state.killer_selection
    if not selection or not selection.route or not state.world or not state.world.rooms:
        return []
    timeline = state.personal_timelines.get(selection.killer_id)
    if not timeline:
        return []
    rooms = {room.name for room in state.world.rooms}
    issues = []
    for step in selection.route:
        times = minutes_in(step.time)
        if not times or step.room not in rooms:
            continue
        places = _place_at(timeline.events, times[0])
        if places and step.room not in places:
            issues.append(
                ValidationIssue(
                    type="timeline_conflict",
                    description=(
                        f"The killer's route has them in {step.room} at {step.time} ({step.action}), "
                        f"but their sheet puts them in {' / '.join(dict.fromkeys(places))}. Their real "
                        "location must follow the route; their account can still claim elsewhere."
                    ),
                    related_ids=[selection.killer_id],
                    fix_in="timelines",
                )
            )
    return issues


def find_missing_planned_observations(state: GameState) -> list[ValidationIssue]:
    """
    Find observations the deduction plan assigns that aren't on their holder's sheet.

    Without them the case can't be solved, or a red herring can't be cleared.
    """
    selection = state.killer_selection
    plan = selection.deduction_plan if selection else None
    if not plan:
        return []
    names = {char.id: char.name for char in state.characters}
    pieces = _plan_pieces(plan)
    issues = []
    for piece in pieces:
        timeline = state.personal_timelines.get(piece.holder_id)
        planned_at = minutes_in(piece.time)
        if piece.kind != "observation" or not timeline or not planned_at:
            continue
        found = any(
            seen.observed_character_id == piece.about_id
            and (times := minutes_in(seen.time))
            and abs(times[0] - planned_at[0]) <= PLANNED_TIME_SLACK_MINUTES
            for event in timeline.events
            for seen in event.info_they_observed
        )
        places = _place_at(timeline.events, planned_at[0])
        if found and piece.from_room and places and piece.from_room not in places:
            issues.append(
                ValidationIssue(
                    type="timeline_conflict",
                    description=(
                        f"The deduction plan has {names.get(piece.holder_id, piece.holder_id)} "
                        f"seeing it from {piece.from_room} at {piece.time}, but their sheet puts "
                        f"them in {' / '.join(dict.fromkeys(places))}. Put them in {piece.from_room}."
                    ),
                    related_ids=[piece.holder_id],
                    fix_in="timelines",
                )
            )
        if not found:
            issues.append(
                ValidationIssue(
                    type="logic_gap",
                    description=(
                        f"The deduction plan gives {names.get(piece.holder_id, piece.holder_id)} an "
                        f"observation of {names.get(piece.about_id, piece.about_id)} around "
                        f"{piece.time} ('{piece.description}'), but their sheet doesn't have it. Add it."
                    ),
                    related_ids=[piece.holder_id],
                    fix_in="timelines",
                )
            )
    return issues


def find_impossible_planned_sightings(state: GameState) -> list[str]:
    """
    Find planned observations of the killer that the floor plan rules out.

    The holder is in from_room at that time; the killer is where their route says.
    A plan that asks for a sighting through walls can't be written consistently,
    so it goes back to A7 before anyone builds on it.
    """
    selection = state.killer_selection
    plan = selection.deduction_plan if selection else None
    if not selection or not plan or not state.world or not state.world.rooms:
        return []
    rooms = {room.name: room for room in state.world.rooms}
    route = [
        (span, step.room)
        for step in selection.route
        if (times := minutes_in(step.time))
        for span in [(times[0], times[1] if len(times) > 1 else times[0])]
    ]
    problems = []
    for piece in plan.breaking_pieces:
        times = minutes_in(piece.time)
        here = rooms.get(piece.from_room)
        if piece.kind != "observation" or piece.about_id != selection.killer_id:
            continue
        if not here or not times:
            continue
        killer_rooms = [room for span, room in route if _covers(span, times[0])]
        visible = {here.name, *here.can_see}
        if killer_rooms and not visible & set(killer_rooms):
            problems.append(
                f"The plan has the {piece.holder_id} sighting at {piece.time} made from "
                f"{here.name}, but the killer is in {' / '.join(killer_rooms)} then, which can't be "
                f"seen from there (visible: {', '.join(sorted(visible))}). Pick an observer room with "
                "a view of the killer's route, or another moment of the route."
            )
    return problems


def find_killer_sheet_stands_out(state: GameState) -> list[ValidationIssue]:
    """
    Find games where the unaware killer is the only suspect with nothing to hide.

    That killer's sheet only shows their innocent-looking account, so it has no
    "what you will tell" part. If every innocent sheet has one, the killer's player
    can tell from the shape of their sheet alone.
    """
    if not state.killer_selection or state.config.killer_knows_identity:
        return []
    killer_id = state.killer_selection.killer_id
    hiding = [
        char_id
        for char_id, timeline in state.personal_timelines.items()
        if char_id != killer_id
        and any(e.what_they_will_tell_others != e.what_they_really_did for e in timeline.events)
    ]
    innocents = [char_id for char_id in state.personal_timelines if char_id != killer_id]
    if not innocents or len(hiding) < len(innocents):
        return []

    names = {char.id: char.name for char in state.characters}
    keep = min(MAX_INNOCENTS_HIDING, len(innocents) - 1)
    truthful = hiding[keep:]
    return [
        ValidationIssue(
            type="spoiler",
            description=(
                "Every innocent suspect hides something, so the killer, who doesn't know they are "
                "the killer and only sees their innocent account, is the only sheet with nothing to "
                f"hide. Make {', '.join(names.get(c, c) for c in truthful)} tell the truth about "
                "their whole evening (what they will tell equals what they really did)."
            ),
            related_ids=truthful,
            fix_in="timelines",
        )
    ]


def find_timeline_violations(state: GameState) -> list[ValidationIssue]:
    """Run every deterministic personal timeline check."""
    return [
        *find_observation_time_errors(state),
        *find_misplaced_observations(state),
        *find_impossible_sightings(state),
        *find_absent_from_shared_events(state),
        *find_murder_heard_out_of_range(state),
        *find_route_not_followed(state),
        *find_missing_planned_observations(state),
        *find_too_many_killer_witnesses(state),
        *find_killer_sheet_stands_out(state),
    ]
