"""Deterministic checks of the suspects' personal timelines.

The LLM reviews kept missing these in real runs: suspects observing from places they
had already left, people seen where their own timeline doesn't put them, and every
player witnessing the killer, which makes the game obvious.
"""

import re
import unicodedata

from mystery_agents.models.state import GameState, PersonalEvent, ValidationIssue

# Minutes of slack around event boundaries: people move between rooms
TIME_TOLERANCE_MINUTES = 5
# Most innocent suspects allowed to see the killer while the killer hides something.
# More than this and players don't need to compare testimonies at all.
MAX_KILLER_WITNESSES = {"easy": 3, "medium": 2, "hard": 2}
# Innocent suspects kept hiding something when some must become truthful
MAX_INNOCENTS_HIDING = 3
# Words too generic to tell two places apart ("the corridor of the upper floor")
MIN_PLACE_WORD_LENGTH = 4

_TIME_PATTERN = re.compile(r"(\d{1,2}):(\d{2})")


def _minutes(text: str) -> list[int]:
    """All HH:MM times in a text, as minutes since midnight."""
    return [int(hours) * 60 + int(mins) for hours, mins in _TIME_PATTERN.findall(text)]


def event_ranges(events: list[PersonalEvent]) -> list[tuple[int, int] | None]:
    """
    Time range of each event, in minutes.

    A single time lasts until the next event starts. Events without a readable
    time get None and are skipped by the checks.
    """
    starts = [_minutes(event.time) for event in events]
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


def _format_time(minute: int) -> str:
    return f"{minute // 60:02d}:{minute % 60:02d}"


def find_observation_time_errors(state: GameState) -> list[ValidationIssue]:
    """Find observations made at a time the observer wasn't in that moment of their evening."""
    names = {char.id: char.name for char in state.characters}
    issues = []
    for char_id, timeline in state.personal_timelines.items():
        for event, span in zip(timeline.events, event_ranges(timeline.events), strict=True):
            if span is None:
                continue
            for seen in event.info_they_observed:
                times = _minutes(seen.time)
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


def find_misplaced_observations(state: GameState) -> list[ValidationIssue]:
    """Find suspects observed somewhere their own timeline says they weren't."""
    names = {char.id: char.name for char in state.characters}
    issues = []
    for char_id, timeline in state.personal_timelines.items():
        for event in timeline.events:
            for seen in event.info_they_observed:
                observed = state.personal_timelines.get(seen.observed_character_id)
                times = _minutes(seen.time)
                if not observed or not times or seen.observed_character_id == char_id:
                    continue
                places = [
                    other.location
                    for other, span in zip(
                        observed.events, event_ranges(observed.events), strict=True
                    )
                    if span and _covers(span, times[0])
                ]
                if places and not any(same_place(seen.location, place) for place in places):
                    observed_name = names.get(
                        seen.observed_character_id, seen.observed_character_id
                    )
                    issues.append(
                        ValidationIssue(
                            type="timeline_conflict",
                            description=(
                                f"{names.get(char_id, char_id)} sees {observed_name} at "
                                f"{seen.location} at {seen.time} ('{seen.text}'), but "
                                f"{observed_name}'s own timeline puts them at "
                                f"{' / '.join(places)} then. Make the observation match where "
                                f"{observed_name} really was."
                            ),
                            related_ids=[char_id],
                            fix_in="timelines",
                        )
                    )
    return issues


def find_too_many_killer_witnesses(state: GameState) -> list[ValidationIssue]:
    """
    Find games where too many suspects see the killer during the moments they hide.

    Those observations break the killer's account. Spread over a couple of players
    they make the detective compare testimonies; given to everyone they solve the case.
    """
    if not state.killer_selection:
        return []
    killer_id = state.killer_selection.killer_id
    killer_timeline = state.personal_timelines.get(killer_id)
    if not killer_timeline:
        return []

    hidden_spans = [
        span
        for event, span in zip(
            killer_timeline.events, event_ranges(killer_timeline.events), strict=True
        )
        if span and event.what_they_will_tell_others != event.what_they_really_did
    ]
    witnesses = [
        char_id
        for char_id, timeline in state.personal_timelines.items()
        if char_id != killer_id
        and any(
            seen.observed_character_id == killer_id
            and (times := _minutes(seen.time))
            and any(_covers(span, times[0]) for span in hidden_spans)
            for event in timeline.events
            for seen in event.info_they_observed
        )
    ]
    limit = MAX_KILLER_WITNESSES.get(state.config.difficulty, 2)
    if len(witnesses) <= limit:
        return []

    names = {char.id: char.name for char in state.characters}
    killer_name = names.get(killer_id, killer_id)
    hidden_times = ", ".join(f"{_format_time(a)}-{_format_time(b)}" for a, b in hidden_spans)
    extra = witnesses[limit:]
    return [
        ValidationIssue(
            type="over_obvious",
            description=(
                f"{len(witnesses)} suspects see {killer_name} during the moments they hide "
                f"({hidden_times}): {', '.join(names.get(w, w) for w in witnesses)}. At most "
                f"{limit} should, each seeing only part of it, or nobody needs to compare "
                f"testimonies. Rewrite the observations of {', '.join(names.get(w, w) for w in extra)} "
                f"so they concern other suspects or other moments of {killer_name}'s evening."
            ),
            related_ids=extra,
            fix_in="timelines",
        )
    ]


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
        *find_too_many_killer_witnesses(state),
        *find_killer_sheet_stands_out(state),
    ]
