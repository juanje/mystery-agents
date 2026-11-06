"""A7.5: Personal Timelines Agent - Alibis, cover stories and observations per suspect."""

import math

from pydantic import BaseModel, Field

from mystery_agents.models.state import GameState, PersonalEvent, PersonalTimeline
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
        suspects_info = []
        for char in state.characters:
            killer_mark = " **THE KILLER**" if char.id == killer_id else ""
            secrets = "; ".join(char.personal_secrets) if char.personal_secrets else "None"
            suspects_info.append(
                f"- {char.name} (ID: {char.id}){killer_mark}: {char.role}\n"
                f"  Motive: {char.motive_for_crime or 'None'}\n"
                f"  Secrets: {secrets}"
            )

        user_message = f"""Create the personal timelines (alibis, cover stories and observations) for the suspects requested below:

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
{state.killer_selection.truth_narrative}

Return the response in the exact JSON format specified in the system prompt.
"""

        # The killer goes first: their real movements are what the others observe
        suspect_ids = sorted(
            (char.id for char in state.characters), key=lambda char_id: char_id != killer_id
        )
        timelines: dict[str, PersonalTimeline] = {}
        returned: list[str] = []
        max_calls = math.ceil(len(suspect_ids) / SUSPECTS_PER_CALL) + EXTRA_CALLS
        for _ in range(max_calls):
            missing = [char_id for char_id in suspect_ids if char_id not in timelines]
            if not missing:
                break
            batch = missing[:SUSPECTS_PER_CALL]
            result = self.invoke(state, user_message + self._batch_request(state, timelines, batch))
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

        state.personal_timelines = timelines
        return state

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
                        info_they_observed=[f"You saw {other.name} looking nervous."],
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
                        info_they_observed=[f"You noticed {other.name} was not in the room."],
                    ),
                ],
                subjective_narrative="A pleasant evening of drinks and conversation, with a short walk in the garden.",
            )
        state.personal_timelines = timelines
        return state
