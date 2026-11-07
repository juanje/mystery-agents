"""A7.5: Personal Timelines Agent - Alibis, cover stories and observations per suspect."""

import math

from pydantic import BaseModel, Field

from mystery_agents.models.state import (
    GameState,
    Observation,
    PersonalEvent,
    PersonalTimeline,
    ValidationIssue,
)
from mystery_agents.utils.cache import LLMCache
from mystery_agents.utils.prompts import A7_5_SYSTEM_PROMPT
from mystery_agents.utils.state_helpers import (
    format_global_timeline,
    format_personal_timelines,
    safe_get_crime_method_description,
    safe_get_crime_scene_description,
    safe_get_crime_time_of_death,
    safe_get_crime_victim_name,
    safe_get_world_location_name,
)

from .base import BaseAgent, StructuredOutputError

# Suspects written per LLM call, to keep each response small and fast
SUSPECTS_PER_CALL = 3
# Calls allowed beyond one per batch, for batches the model returns incomplete
EXTRA_CALLS = 2
# Fewer events than this can't cover the evening (the prompt asks for 4 to 6)
MIN_EVENTS_PER_SUSPECT = 3


class A7_5Output(BaseModel):
    """Output format for A7.5 agent."""

    timelines: list[PersonalTimeline] = Field(
        description="One personal timeline per suspect, covering the whole Act 1 timeline."
    )


class PersonalTimelinesAgent(BaseAgent):
    """
    A7.5: Personal Timelines Agent.

    Gives each suspect what they did, what they will claim and what they saw,
    once the killer and the truth are known. This is a tier 1 agent (logic).
    """

    def __init__(self) -> None:
        """Initialize the personal timelines agent."""
        super().__init__(llm=LLMCache.get_model("tier1"), response_format=A7_5Output)

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
            state: Current game state with characters, timeline and killer selection

        Returns:
            Updated game state with personal timelines

        Raises:
            StructuredOutputError: If a suspect is left without a timeline
        """
        if self._should_use_mock(state):
            return self._mock_output(state)

        if not state.killer_selection:
            raise ValueError("Killer selection is required for personal timelines")

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
        if self._should_use_mock(state) or not state.killer_selection:
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
        context = (
            self._context(state)
            + f"""
REVIEW OF THE PREVIOUS VERSION - these problems were found in the timelines you are rewriting:
{problems}
Keep what was fine in those timelines and change only what fixes these problems.
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

GLOBAL TIMELINE (DO NOT CONTRADICT):
{format_global_timeline(state)}

TRUTH NARRATIVE (what really happened):
{state.killer_selection.truth_narrative if state.killer_selection else ""}
"""

    def _write(
        self,
        state: GameState,
        context: str,
        suspect_ids: list[str],
        written: dict[str, PersonalTimeline],
    ) -> dict[str, PersonalTimeline]:
        """
        Write the timelines of the given suspects in small batches.

        Args:
            state: Current game state
            context: Story context for the request
            suspect_ids: Suspects to write, in writing order
            written: Timelines already written, shown as context and kept

        Returns:
            The new timelines, keyed by character ID

        Raises:
            StructuredOutputError: If a suspect is left without a timeline
        """
        timelines: dict[str, PersonalTimeline] = {}
        returned: list[str] = []
        max_calls = math.ceil(len(suspect_ids) / SUSPECTS_PER_CALL) + EXTRA_CALLS
        for _ in range(max_calls):
            missing = [char_id for char_id in suspect_ids if char_id not in timelines]
            if not missing:
                break
            batch = missing[:SUSPECTS_PER_CALL]
            request = context + self._batch_request(state, {**written, **timelines}, batch)
            result = self.invoke(state, request)
            returned = [t.character_id for t in result.timelines]
            # An empty timeline leaves that player with nothing to say, so ask for it again
            timelines.update(
                (t.character_id, t)
                for t in result.timelines
                if t.character_id in batch and len(t.events) >= MIN_EVENTS_PER_SUSPECT
            )

        missing = [char_id for char_id in suspect_ids if char_id not in timelines]
        if missing:
            # Raising lets the graph retry the node from scratch
            raise StructuredOutputError(
                f"No personal timeline for suspects: {missing} (last returned: {returned})"
            )
        return timelines

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
