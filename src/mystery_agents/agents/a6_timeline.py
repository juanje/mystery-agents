"""A6: Timeline Global Agent - Creates the event sequence."""

from mystery_agents.models.state import GameState, GlobalTimeline
from mystery_agents.utils.cache import LLMCache
from mystery_agents.utils.floor_plan import find_room, format_floor_plan
from mystery_agents.utils.prompts import A6_SYSTEM_PROMPT
from mystery_agents.utils.state_helpers import (
    format_retry_feedback,
    safe_get_crime_method_description,
    safe_get_crime_scene_description,
    safe_get_crime_time_of_death,
    safe_get_crime_victim_name,
    safe_get_crime_weapon,
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
            characters_list.append(
                f"- {char.name} (ID: {char.id}): {char.role}, {char.relation_to_victim}"
            )

        time_of_death = safe_get_crime_time_of_death(state)
        # A retry after V2 must know what was wrong with the previous timeline
        report = state.validation
        retry_feedback = (
            format_retry_feedback(
                [f"{i.type}: {i.description}" for i in report.issues], report.suggested_fixes
            )
            if report and not report.is_consistent
            else ""
        )

        user_message = f"""Write the shared facts of Act 1 for this mystery party game:

SETTING:
- Location: {safe_get_world_location_name(state)}
- Gathering reason: {state.world.gathering_reason if state.world else "A special gathering"}
- Duration: {state.config.duration_minutes} minutes

VICTIM (HOST):
- {safe_get_crime_victim_name(state)}

CHARACTERS (SUSPECTS):
{chr(10).join(characters_list) if characters_list else "No characters"}

CRIME (the killer is chosen later):
- Method: {safe_get_crime_method_description(state)}
- Weapon: {safe_get_crime_weapon(state)}
- Scene: {safe_get_crime_scene_description(state)}
- Time of death: {time_of_death}

FLOOR PLAN (every room_id and the murder room are exactly one of these room names):
{format_floor_plan(state)}

{retry_feedback}"""

        # Invoke LLM with structured output
        result = self.invoke(state, user_message)

        # Room names must match the plan exactly; later checks compare them
        for block in result.time_blocks:
            for event in block.events:
                if event.room_id:
                    event.room_id = find_room(state, event.room_id) or event.room_id
        result.murder_event.room = (
            find_room(state, result.murder_event.room) or result.murder_event.room
        )

        # Update state
        state.timeline_global = result

        return state

    def _mock_output(self, state: GameState) -> GameState:
        """Generate mock data for dry run mode."""
        from mystery_agents.models.state import GlobalEvent, GlobalTimeline, MurderEvent, TimeBlock

        time_blocks = [
            TimeBlock(
                start="20:00",
                end="21:00",
                events=[
                    GlobalEvent(
                        time_approx="20:30",
                        description="Guests arrive and mingle",
                        character_ids_involved=[c.id for c in state.characters[:3]],
                        room_id="Hall",
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
                        room_id="Drawing room",
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
                        room_id="Study",
                    )
                ],
            ),
        ]

        state.timeline_global = GlobalTimeline(
            time_blocks=time_blocks,
            murder_event=MurderEvent(
                time="22:30",
                room="Study",
                what_happens="The victim drinks from the glass and collapses over the desk.",
                heard_elsewhere="A thud and a glass breaking",
                heard_in=["Hall"],
                scene_when_found="The victim lies by the desk; the glass is broken on the floor.",
                discovered_at="22:35",
                discovered_by=[c.id for c in state.characters[:1]],
            ),
        )

        return state
