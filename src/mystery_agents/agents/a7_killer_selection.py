"""A7: Killer Selection Agent - Chooses the culprit and ensures logic is sound."""

from mystery_agents.models.state import GameState, KillerSelection
from mystery_agents.utils.cache import LLMCache
from mystery_agents.utils.prompts import A7_SYSTEM_PROMPT

from .base import BaseAgent


class KillerSelectionAgent(BaseAgent):
    """
    A7: Killer Selection Agent.

    Selects the killer from suspects and ensures the mystery logic is airtight.
    This is a tier 1 agent (logic - most powerful LLM).
    """

    def __init__(self) -> None:
        """Initialize the killer selection agent."""
        super().__init__(llm=LLMCache.get_model("tier1"), response_format=KillerSelection)

    def get_system_prompt(self, state: GameState) -> str:
        """
        Get the system prompt for killer selection.

        Args:
            state: Current game state

        Returns:
            System prompt string
        """
        return A7_SYSTEM_PROMPT

    def run(self, state: GameState) -> GameState:
        """
        Select the killer and ensure logic consistency.

        Args:
            state: Current game state with world, crime, characters, and timeline

        Returns:
            Updated game state with killer selection
        """
        # If dry run, return mock data
        if self._should_use_mock(state):
            return self._mock_output(state)

        # Prepare context for LLM
        suspects_info = []
        for char in state.characters:
            suspects_info.append(
                f"- {char.name} (ID: {char.id}): {char.role}, motive: {char.motive_for_crime}"
            )

        # Format timeline with detailed events so A7 can see what actually happened
        timeline_summary = "No timeline yet"
        if state.timeline_global and state.timeline_global.time_blocks:
            timeline_events = []
            for block in state.timeline_global.time_blocks:
                timeline_events.append(f"\n--- {block.start} to {block.end} ---")
                for event in block.events:
                    char_names = []
                    if event.character_ids_involved:
                        for char_id in event.character_ids_involved:
                            matching_char = next(
                                (c for c in state.characters if c.id == char_id), None
                            )
                            if matching_char:
                                char_names.append(matching_char.name)

                    char_str = f" [{', '.join(char_names)}]" if char_names else ""
                    location_str = f" (Location: {event.room_id})" if event.room_id else ""
                    timeline_events.append(
                        f"  • {event.time_approx}{char_str}{location_str}: {event.description}"
                    )
            timeline_summary = "\n".join(timeline_events)

        # Validate that crime is set (should always be at this point in workflow)
        if not state.crime:
            raise ValueError("Crime specification is required for killer selection")

        user_message = f"""Select the killer from these suspects and ensure the mystery is logically sound:

VICTIM (HOST - NOT A SUSPECT):
- {state.crime.victim.name}: {state.crime.victim.role_in_setting}

SUSPECTS (PLAYERS - CHOOSE ONE AS KILLER):
{chr(10).join(suspects_info)}

CRIME DETAILS:
- Method: {state.crime.murder_method.description}
- Weapon: {state.crime.murder_method.weapon_used}
- Location: {state.crime.crime_scene.description}
- Time of death: {state.crime.time_of_death_approx}

TIMELINE (EXISTING EVENTS - DO NOT CONTRADICT):
{timeline_summary}

DIFFICULTY: {state.config.difficulty}

**CRITICAL INSTRUCTIONS**:
1. Review the timeline carefully and identify which suspect has the BEST opportunity based on EXISTING timeline events
2. Look for moments where a suspect was:
   - Alone or unaccounted for
   - Near the crime location
   - Had a plausible reason to slip away
   - A gap in their whereabouts during or just before time of death
3. Your truth_narrative MUST explain what happened during these EXISTING gaps/opportunities
4. DO NOT invent new events (like "they met privately at X time") unless that event exists in the timeline
5. If you need the killer to have done something, use an existing timeline event where they could have done it

REQUIREMENTS:
1. Create a KillerSelection object with: killer_id, rationale, modified_events, truth_narrative
2. killer_id must match one of the suspect character IDs exactly who HAS an opportunity in the timeline
3. rationale must explain why this character was chosen based on EXISTING timeline opportunities
4. modified_events is an array of strings (can be empty [])
5. truth_narrative must explain the murder using ONLY existing timeline events and gaps
6. All string fields must have values - do not leave any empty

Return the response in the exact JSON format specified in the system prompt.
"""

        # Invoke LLM with structured output
        result = self.invoke(state, user_message)

        # Validate killer is in suspects list
        killer_ids = [c.id for c in state.characters]
        if result.killer_id not in killer_ids:
            # Fallback: choose first character
            result.killer_id = state.characters[0].id if state.characters else "unknown"

        # Update state
        state.killer_selection = result

        return state

    def _mock_output(self, state: GameState) -> GameState:
        """Generate mock data for dry run mode."""
        # Choose first character as killer for mock
        killer_id = state.characters[0].id if state.characters else "mock-killer"

        state.killer_selection = KillerSelection(
            killer_id=killer_id,
            rationale="This character had the strongest motive and best opportunity.",
            modified_events=["Adjusted timeline to place killer near victim at time of death"],
            truth_narrative="""The killer poisoned the victim's evening brandy during Act 1.
They had access to the study and the means to obtain the poison.
The motive was related to the recent change in the victim's will.
All evidence points to this conclusion when properly analyzed.""",
        )

        return state
