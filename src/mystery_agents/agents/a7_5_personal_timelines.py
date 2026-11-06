"""A7.5: Personal Timelines Agent - Alibis, cover stories and observations per suspect."""

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

# Calls to complete every suspect's timeline before giving up on this attempt
MAX_COMPLETION_CALLS = 3


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

        user_message = f"""Create the personal timelines (alibis, cover stories and observations) for every suspect:

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

REQUIRED OUTPUT:
Return exactly {len(state.characters)} timelines, one for each of these character_ids: {", ".join(char.id for char in state.characters)}

Return the response in the exact JSON format specified in the system prompt.
"""

        suspect_ids = [char.id for char in state.characters]
        timelines: dict[str, PersonalTimeline] = {}
        returned: list[str] = []
        for _ in range(MAX_COMPLETION_CALLS):
            missing = [char_id for char_id in suspect_ids if char_id not in timelines]
            if not missing:
                break
            result = self.invoke(
                state, user_message + self._completion_request(state, timelines, missing)
            )
            returned = [t.character_id for t in result.timelines]
            timelines.update(
                (t.character_id, t) for t in result.timelines if t.character_id in missing
            )

        missing = [char_id for char_id in suspect_ids if char_id not in timelines]
        if missing:
            # Raising lets the graph retry the node from scratch
            raise StructuredOutputError(
                f"No personal timeline for suspects: {missing} (last returned: {returned})"
            )

        state.personal_timelines = timelines
        return state

    def _completion_request(
        self, state: GameState, written: dict[str, PersonalTimeline], missing: list[str]
    ) -> str:
        """
        Ask for the timelines still missing, showing the ones already written.

        The model sometimes returns only the first suspect. Asking for the rest with the
        written ones as context keeps observations consistent across suspects without
        throwing away the work already done.
        """
        if not written:
            return ""
        done = state.model_copy(update={"personal_timelines": written})
        return f"""
ALREADY WRITTEN (keep them as they are and stay consistent with them):
{format_personal_timelines(done)}

Now return ONLY the timelines for these character_ids: {", ".join(missing)}
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
