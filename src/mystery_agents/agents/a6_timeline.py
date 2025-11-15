"""A6: Timeline Global Agent - Creates the event sequence."""

from mystery_agents.models.state import GameState, GlobalTimeline
from mystery_agents.utils.cache import LLMCache
from mystery_agents.utils.prompts import A6_SYSTEM_PROMPT
from mystery_agents.utils.state_helpers import (
    safe_get_crime_time_of_death,
    safe_get_crime_victim_name,
    safe_get_world_location_name,
)

from .base import BaseAgent


class TimelineAgent(BaseAgent):
    """
    A6: Timeline Global Agent.

    Creates a timeline of events before/during Act 1.
    """

    def __init__(self) -> None:
        """Initialize the timeline agent."""
        super().__init__(llm=LLMCache.get_model("tier2"), response_format=GlobalTimeline)

    def get_system_prompt(self, state: GameState) -> str:
        """
        Get the system prompt for timeline generation.

        Args:
            state: Current game state

        Returns:
            System prompt string
        """
        return A6_SYSTEM_PROMPT

    def run(self, state: GameState) -> GameState:
        """
        Generate the global timeline.

        Args:
            state: Current game state with world, crime, and characters

        Returns:
            Updated game state with timeline
        """
        # If dry run, return mock data
        if self._should_use_mock(state):
            return self._mock_output(state)

        # Prepare context for LLM
        characters_list = []
        for char in state.characters:
            characters_list.append(f"- {char.name} (ID: {char.id}): {char.role}")

        rooms_list = []
        if state.maps:
            for map_spec in state.maps:
                for room in map_spec.rooms:
                    rooms_list.append(f"- {room.name} (ID: {room.id})")

        time_of_death = safe_get_crime_time_of_death(state)
        user_message = f"""Generate a timeline of events for the mystery party game:

SETTING:
- Location: {safe_get_world_location_name(state)}
- Time of death: {time_of_death}

CHARACTERS (SUSPECTS):
{chr(10).join(characters_list) if characters_list else "No characters"}

VICTIM (HOST):
- {safe_get_crime_victim_name(state)}

ROOMS:
{chr(10).join(rooms_list) if rooms_list else "Generate appropriate rooms"}

REQUIREMENTS:
1. Create a "time_blocks" array with TimeBlock objects
2. Each TimeBlock must have: start (HH:MM), end (HH:MM), events (array of GlobalEvent)
3. Each GlobalEvent must have: time_approx (HH:MM), description, character_ids_involved (array), room_id (string or null)
4. Create a "live_action_murder_event" GlobalEvent object (or null)
5. All character IDs must match IDs from the characters list
6. All times must be in HH:MM format (e.g., "20:30")
7. Arrays can be empty [] if not applicable

**CRITICAL FOR GAMEPLAY**: The timeline MUST create plausible opportunity windows for AT LEAST 3-4 different suspects:
- Include moments where suspects are alone, unaccounted for, or could plausibly access the crime scene
- Create gaps in alibis (e.g., "X went to fetch wine", "Y stepped outside", "Z was looking for something")
- Show natural movements and reasons for characters to be near various locations
- Don't lock all characters into ironclad group alibis - leave flexibility

The killer will be selected later, so the timeline should work for MULTIPLE possible killers, not just one.

Return the response in the exact JSON format specified in the system prompt.
"""

        # Invoke LLM with structured output
        result = self.invoke(state, user_message)

        # Update state
        state.timeline_global = result

        return state

    def _mock_output(self, state: GameState) -> GameState:
        """Generate mock data for dry run mode."""
        from mystery_agents.models.state import GlobalEvent, GlobalTimeline, TimeBlock

        time_blocks = [
            TimeBlock(
                start="20:00",
                end="21:00",
                events=[
                    GlobalEvent(
                        time_approx="20:30",
                        description="Guests arrive and mingle",
                        character_ids_involved=[c.id for c in state.characters[:3]],
                        room_id="main_hall",
                    )
                ],
            ),
            TimeBlock(
                start="21:00",
                end="22:00",
                events=[
                    GlobalEvent(
                        time_approx="21:30",
                        description="Dinner is served",
                        character_ids_involved=[c.id for c in state.characters],
                        room_id="dining_room",
                    )
                ],
            ),
            TimeBlock(
                start="22:00",
                end="23:00",
                events=[
                    GlobalEvent(
                        time_approx="22:30",
                        description="Murder occurs",
                        character_ids_involved=[],
                        room_id="study",
                    )
                ],
            ),
        ]

        state.timeline_global = GlobalTimeline(
            time_blocks=time_blocks,
            live_action_murder_event=GlobalEvent(
                time_approx="22:30",
                description="The lights go out and a scream is heard",
                character_ids_involved=[],
                room_id="study",
            ),
        )

        return state
