"""A7.5: Personal Timelines Agent - Alibis, cover stories and observations per suspect."""

import math

from pydantic import BaseModel, ConfigDict, Field

from mystery_agents.models.state import (
    GameState,
    Observation,
    PersonalEvent,
    PersonalTimeline,
    ValidationIssue,
    required_in_schema,
)
from mystery_agents.utils.cache import LLMCache
from mystery_agents.utils.floor_plan import format_floor_plan
from mystery_agents.utils.presence import (
    format_presence,
    hears_the_murder,
    numbered_sightings,
    planned_piece_for,
    presence_slots,
    slot_hides,
    slot_spans,
)
from mystery_agents.utils.prompts import A7_5_SYSTEM_PROMPT
from mystery_agents.utils.state_helpers import (
    format_deduction_plan,
    format_global_timeline,
    format_killer_route,
    format_personal_timelines,
    safe_get_crime_method_description,
    safe_get_crime_scene_description,
    safe_get_crime_time_of_death,
    safe_get_crime_victim_name,
    safe_get_world_location_name,
)
from mystery_agents.utils.timeline_checks import clock, minutes_in

from .base import BaseAgent, StructuredOutputError

# Suspects written per LLM call, to keep each response small and fast
SUSPECTS_PER_CALL = 3
# Calls allowed beyond one per batch, for batches the model returns incomplete
EXTRA_CALLS = 2


# The writer only writes texts: code already knows each slot's time and room from the
# presence grid, and who can see whom, and builds the sheet from them. Writing events
# freely, the model kept its own times and places, and the sheets disagreed with the
# grid and each other.


class SlotObservation(BaseModel):
    """Something the suspect notices, picked from the sightings the grid allows."""

    sighting: str = Field(description="ID of the sighting from this suspect's list (e.g. 'S3').")
    time: str = Field(description="HH:MM within that sighting's time range.")
    text: str = Field(description=Observation.model_fields["text"].description)


class SlotText(BaseModel):
    """The texts of one slot of the suspect's evening."""

    model_config = ConfigDict(json_schema_extra=required_in_schema("observations"))

    slot: int = Field(description="Number of the slot in this suspect's list, from 1.")
    what_they_really_did: str = Field(
        description=PersonalEvent.model_fields["what_they_really_did"].description
    )
    what_they_will_tell_others: str = Field(
        description=PersonalEvent.model_fields["what_they_will_tell_others"].description
    )
    observations: list[SlotObservation] = Field(
        default_factory=list,
        description="Sightings from this suspect's list that fall in this slot and matter: motive, opportunity or alibi.",
    )


class SuspectSheet(BaseModel):
    """One suspect's evening as texts for the grid's slots."""

    model_config = ConfigDict(json_schema_extra=required_in_schema("heard_the_murder"))

    character_id: str = Field(description="ID of the suspect.")
    slots: list[SlotText] = Field(description="One entry for every slot of this suspect, in order.")
    heard_the_murder: str = Field(
        default="",
        description="What they hear of the murder, in second person, when the request says it reaches them; empty otherwise.",
    )
    subjective_narrative: str = Field(
        description=PersonalTimeline.model_fields["subjective_narrative"].description
    )


class SheetsOutput(BaseModel):
    """The texts of the requested suspects' sheets."""

    sheets: list[SuspectSheet] = Field(description="One entry per requested suspect.")


def build_timeline(state: GameState, sheet: SuspectSheet) -> PersonalTimeline | None:
    """
    Build a suspect's sheet from the grid and the texts written for it.

    Times, rooms and claimed rooms come from the grid; each observation goes into the
    slot it happens in, placed where the observed person really is. A slot left without
    text keeps what the grid says they do. A slot where they hide nothing tells the
    truth, so its two texts are the same.

    Returns:
        The timeline, or None when the suspect has no slots in the grid
    """
    slots = presence_slots(state).get(sheet.character_id)
    if not slots:
        return None
    texts = {entry.slot: entry for entry in sheet.slots}
    # The killer hides what they do even where they claim the right room (taking a key,
    # waiting for the body to be found), and an unaware killer's sheet shows only the account
    is_killer = bool(
        state.killer_selection and state.killer_selection.killer_id == sheet.character_id
    )
    events = []
    for number, slot in enumerate(slots, 1):
        entry = texts.get(number)
        did = entry.what_they_really_did if entry else slot.doing
        hides = slot_hides(slot)
        told = entry.what_they_will_tell_others if entry and (hides or is_killer) else did
        events.append(
            PersonalEvent(
                time=slot.time,
                location=slot.room,
                claimed_location=slot.claimed_room if hides else "",
                what_they_really_did=did,
                what_they_will_tell_others=told,
            )
        )

    spans = slot_spans(slots)

    def event_at(minute: int, room: str | None = None) -> PersonalEvent | None:
        """The event covering a minute; at a change of room, the one in the given room."""
        covering = [
            event
            for event, span in zip(events, spans, strict=True)
            if span and span[0] <= minute <= span[1]
        ]
        return next((e for e in covering if e.location == room), covering[0] if covering else None)

    sightings = dict(numbered_sightings(state, sheet.character_id))
    used = set()
    for entry in sheet.slots:
        for seen in entry.observations:
            sighting = sightings.get(seen.sighting.strip().upper())
            if not sighting or seen.sighting in used:
                continue
            used.add(seen.sighting)
            wanted = minutes_in(seen.time)
            minute = min(max(wanted[0] if wanted else sighting.start, sighting.start), sighting.end)
            # A sighting starting as the observer walks in belongs to the room they see it from
            event = event_at(minute, sighting.from_room)
            if event and event.location == sighting.from_room:
                event.info_they_observed.append(
                    Observation(
                        time=clock(minute),
                        observed_character_id=sighting.observed_id,
                        location=sighting.room,
                        text=seen.text,
                    )
                )

    murder = state.timeline_global.murder_event if state.timeline_global else None
    murder_at = minutes_in(murder.time) if murder else []
    then = event_at(murder_at[0]) if murder_at else None
    if murder and then and sheet.heard_the_murder and hears_the_murder(state, sheet.character_id):
        then.info_they_observed.append(
            Observation(
                time=murder.time,
                location=murder.room,
                text=sheet.heard_the_murder,
                sense="sound",
                of_the_murder=True,
            )
        )
    return PersonalTimeline(
        character_id=sheet.character_id,
        events=events,
        subjective_narrative=sheet.subjective_narrative,
    )


class PersonalTimelinesAgent(BaseAgent):
    """
    A7.5: Personal Timelines Agent.

    Gives each suspect what they did, what they will claim and what they saw,
    once the killer and the truth are known. This is a tier 1 agent (logic).
    """

    def __init__(self) -> None:
        """Initialize the personal timelines agent."""
        super().__init__(llm=LLMCache.get_model("tier1"), response_format=SheetsOutput)

    def get_system_prompt(self, state: GameState) -> str:
        """
        Get the system prompt for personal timelines.

        Args:
            state: Current game state

        Returns:
            System prompt string
        """
        return A7_5_SYSTEM_PROMPT

    def run(self, state: GameState) -> GameState:
        """
        Generate a personal timeline for every suspect.

        Args:
            state: Current game state with characters, timeline, killer selection and presence grid

        Returns:
            Updated game state with personal timelines

        Raises:
            ValueError: If the killer or the presence grid is missing
            StructuredOutputError: If a suspect is left without a timeline
        """
        if self._should_use_mock(state):
            return self._mock_output(state)

        if not state.killer_selection:
            raise ValueError("Killer selection is required for personal timelines")
        if not state.presence:
            raise ValueError("A presence grid is required for personal timelines")

        killer_id = state.killer_selection.killer_id
        # The killer goes first: their real movements are what the others observe
        suspect_ids = sorted(
            (char.id for char in state.characters), key=lambda char_id: char_id != killer_id
        )
        state.personal_timelines = self._write(state, self._context(state), suspect_ids, {})
        return state

    def revise(self, state: GameState, issues: list[ValidationIssue]) -> GameState:
        """
        Rewrite the timelines of the suspects a content review found broken.

        The other timelines stay as they are and are shown as context, so the
        rewritten ones stay consistent with them.

        Args:
            state: Current game state with personal timelines
            issues: Review issues with fix_in="timelines"

        Returns:
            Updated game state with the revised personal timelines
        """
        if self._should_use_mock(state) or not state.killer_selection or not state.presence:
            return state

        suspect_ids = {char.id for char in state.characters}
        to_rewrite = [
            char_id
            for char_id in state.personal_timelines
            if any(char_id in issue.related_ids for issue in issues) and char_id in suspect_ids
        ]
        if not to_rewrite:
            return state

        kept = {
            char_id: timeline
            for char_id, timeline in state.personal_timelines.items()
            if char_id not in to_rewrite
        }
        problems = "\n".join(f"- {issue.description}" for issue in issues)
        previous = state.model_copy(
            update={
                "personal_timelines": {
                    char_id: state.personal_timelines[char_id] for char_id in to_rewrite
                }
            }
        )
        context = (
            self._context(state)
            + f"""
PREVIOUS VERSION OF THE TIMELINES YOU ARE REWRITING:
{format_personal_timelines(previous)}

REVIEW OF THAT VERSION - these problems were found in it:
{problems}
Keep what was fine and change only what fixes these problems.
"""
        )
        state.personal_timelines = {**kept, **self._write(state, context, to_rewrite, kept)}
        return state

    def _context(self, state: GameState) -> str:
        """The story every timeline must agree with."""
        killer_id = state.killer_selection.killer_id if state.killer_selection else None
        suspects_info = []
        for char in state.characters:
            killer_mark = " **THE KILLER**" if char.id == killer_id else ""
            secrets = "; ".join(char.personal_secrets) if char.personal_secrets else "None"
            suspects_info.append(
                f"- {char.name} (ID: {char.id}){killer_mark}: {char.role}\n"
                f"  Motive: {char.motive_for_crime or 'None'}\n"
                f"  Secrets: {secrets}"
            )

        return f"""Create the personal timelines (alibis, cover stories and observations) for the suspects requested below:

GAME:
- Target language: {state.config.language}
- Setting: {safe_get_world_location_name(state)}
- Difficulty: {state.config.difficulty}

VICTIM (HOST - NOT A SUSPECT):
- {safe_get_crime_victim_name(state)}

SUSPECTS:
{chr(10).join(suspects_info)}

CRIME:
- Method: {safe_get_crime_method_description(state)}
- Location: {safe_get_crime_scene_description(state)}
- Time of death: {safe_get_crime_time_of_death(state)}

FLOOR PLAN (every location is one of these room names; people move only along connections, and can only see their own room or the rooms listed as visible from it):
{format_floor_plan(state)}

GLOBAL TIMELINE (DO NOT CONTRADICT):
{format_global_timeline(state)}

TRUTH NARRATIVE (what really happened):
{state.killer_selection.truth_narrative if state.killer_selection else ""}

KILLER'S REAL ROUTE (the killer's sheet places them exactly here):
{format_killer_route(state)}

DEDUCTION PLAN (write every observation it assigns on its holder's sheet, at that time):
{format_deduction_plan(state)}

WHERE EVERYONE REALLY IS (fixed; each sheet tells this evening):
{format_presence(state)}
"""

    def _write(
        self,
        state: GameState,
        context: str,
        suspect_ids: list[str],
        written: dict[str, PersonalTimeline],
    ) -> dict[str, PersonalTimeline]:
        """
        Write the texts of each suspect's slots in small batches, and build the sheets in code.

        Args:
            state: Current game state with the presence grid
            context: Story context for the request
            suspect_ids: Suspects to write, in writing order
            written: Timelines already written, shown as context and kept

        Returns:
            The new timelines, keyed by character ID

        Raises:
            StructuredOutputError: If a suspect is left without a sheet
        """
        timelines: dict[str, PersonalTimeline] = {}
        returned: list[str] = []
        max_calls = math.ceil(len(suspect_ids) / SUSPECTS_PER_CALL) + EXTRA_CALLS
        for _ in range(max_calls):
            missing = [char_id for char_id in suspect_ids if char_id not in timelines]
            if not missing:
                break
            batch = missing[:SUSPECTS_PER_CALL]
            request = (
                context
                + self._batch_request(state, {**written, **timelines}, batch)
                + self._slots_request(state, batch)
            )
            result = self.invoke(state, request)
            returned = [sheet.character_id for sheet in result.sheets]
            for sheet in result.sheets:
                # A sheet with no texts leaves that player with nothing to say, so ask again
                if sheet.character_id not in batch or not sheet.slots:
                    continue
                timeline = build_timeline(state, sheet)
                if timeline:
                    timelines[sheet.character_id] = timeline

        missing = [char_id for char_id in suspect_ids if char_id not in timelines]
        if missing:
            # Raising lets the graph retry the node from scratch
            raise StructuredOutputError(
                f"No personal timeline for suspects: {missing} (last returned: {returned})"
            )
        return timelines

    @staticmethod
    def _slots_request(state: GameState, batch: list[str]) -> str:
        """Each requested suspect's slots, what they can see, and whether they hear the murder."""
        names = {char.id: char.name for char in state.characters}
        grid = presence_slots(state)
        murder = state.timeline_global.murder_event if state.timeline_global else None
        blocks = []
        for char_id in batch:
            lines = [f"### {names.get(char_id, char_id)} ({char_id})", "Slots:"]
            for number, slot in enumerate(grid.get(char_id, []), 1):
                claim = (
                    f" - they will claim to be in {slot.claimed_room}" if slot_hides(slot) else ""
                )
                lines.append(f"{number}. {slot.time} @ {slot.room}{claim}: {slot.doing}")
            lines.append("What they can see (pick by ID; nothing else can be seen):")
            sightings = numbered_sightings(state, char_id)
            for sighting_id, sighting in sightings:
                piece = planned_piece_for(state, sighting)
                planned = f" PLANNED, write it: {piece.description}" if piece else ""
                lines.append(
                    f"- {sighting_id}: {clock(sighting.start)}-{clock(sighting.end)} from "
                    f"{sighting.from_room}: {names.get(sighting.observed_id, sighting.observed_id)} "
                    f"in {sighting.room}.{planned}"
                )
            if not sightings:
                lines.append("- nobody")
            heard_from = hears_the_murder(state, char_id)
            lines.append(
                f"They hear the murder from {heard_from}: {murder.heard_elsewhere}"
                if murder and heard_from
                else "The murder doesn't reach them: leave heard_the_murder empty."
            )
            blocks.append("\n".join(lines))
        return (
            "\nWRITE THESE SHEETS (one text per slot; the times and rooms are fixed):\n\n"
            + "\n\n".join(blocks)
            + "\n"
        )

    def _batch_request(
        self, state: GameState, written: dict[str, PersonalTimeline], batch: list[str]
    ) -> str:
        """
        Ask for a few suspects' timelines, showing the ones already written.

        Writing every suspect in one response is slow enough to hit server deadlines,
        and the model sometimes stops after the first one. Small batches with the
        written timelines as context keep observations consistent across suspects.
        """
        written_section = ""
        if written:
            done = state.model_copy(update={"personal_timelines": written})
            written_section = f"""
ALREADY WRITTEN (keep them as they are and stay consistent with them):
{format_personal_timelines(done)}
"""
        return f"""{written_section}
Now return ONLY the timelines for these character_ids: {", ".join(batch)}

"""

    def _mock_output(self, state: GameState) -> GameState:
        """Generate mock personal timelines for dry run mode."""
        killer_id = state.killer_selection.killer_id if state.killer_selection else None
        timelines = {}
        for index, char in enumerate(state.characters):
            other = state.characters[(index + 1) % len(state.characters)]
            is_killer = char.id == killer_id
            timelines[char.id] = PersonalTimeline(
                character_id=char.id,
                events=[
                    PersonalEvent(
                        time="21:00-21:30",
                        location="Drawing room",
                        what_they_really_did="You chatted with the other guests over drinks.",
                        what_they_will_tell_others="You chatted with the other guests over drinks.",
                        info_they_observed=[
                            Observation(
                                time="21:15",
                                observed_character_id=other.id,
                                location="Drawing room",
                                text=f"You saw {other.name} looking nervous.",
                            )
                        ],
                    ),
                    PersonalEvent(
                        time="21:30-21:45",
                        location="Study" if is_killer else "Garden",
                        claimed_location="Garden" if is_killer else "",
                        what_they_really_did=(
                            "You slipped into the study and poured poison into the brandy."
                            if is_killer
                            else "You went out to the garden for some fresh air."
                        ),
                        what_they_will_tell_others="You went out to the garden for some fresh air.",
                        info_they_observed=[
                            Observation(
                                time="21:35",
                                observed_character_id=other.id,
                                location="Study" if other.id == killer_id else "Garden",
                                text=f"You noticed {other.name} was not in the room.",
                            )
                        ],
                    ),
                ],
                subjective_narrative="A pleasant evening of drinks and conversation, with a short walk in the garden.",
            )
        state.personal_timelines = timelines
        return state
