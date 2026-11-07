"""V2: Game Logic Validator - Validates complete game logic consistency."""

from mystery_agents.models.state import GameState, ValidationReport
from mystery_agents.utils.cache import LLMCache
from mystery_agents.utils.floor_plan import format_floor_plan
from mystery_agents.utils.prompts import V2_GAME_LOGIC_VALIDATOR_SYSTEM_PROMPT
from mystery_agents.utils.state_helpers import (
    format_deduction_plan,
    format_global_timeline,
    format_killer_route,
    format_personal_timelines,
    safe_get_crime_method_description,
    safe_get_crime_scene_room_id,
    safe_get_crime_time_of_death,
    safe_get_crime_victim_name,
    safe_get_crime_victim_role,
    safe_get_crime_weapon,
)

from .base import BaseAgent


class GameLogicValidatorAgent(BaseAgent):
    """
    V2: Game Logic Validator Agent.

    Validates the entire game state for logical consistency.
    This is a tier 1 agent (logic - most powerful LLM).
    CRITICAL: This agent controls the retry loop.
    Runs SECOND in the validation sequence (after A7: Killer Selection).
    """

    def __init__(self) -> None:
        """Initialize the game logic validator agent."""
        super().__init__(llm=LLMCache.get_model("tier1"), response_format=ValidationReport)

    def get_system_prompt(self, state: GameState) -> str:
        """
        Get the system prompt for game logic validation.

        Args:
            state: Current game state

        Returns:
            System prompt string
        """
        return V2_GAME_LOGIC_VALIDATOR_SYSTEM_PROMPT

    def run(self, state: GameState) -> GameState:
        """
        Validate the game state for logical consistency.

        Args:
            state: Current game state with all components

        Returns:
            Updated game state with validation report
        """
        # If dry run, return mock validation (always pass)
        if self._should_use_mock(state):
            return self._mock_output(state)

        # Prepare comprehensive context for validation
        victim_name = safe_get_crime_victim_name(state)
        killer_name = "Unknown"
        if state.killer_selection:
            killer_id = state.killer_selection.killer_id
            killer = next((c for c in state.characters if c.id == killer_id), None)
            killer_name = killer.name if killer else "Unknown"

        characters_summary = []
        for char in state.characters:
            characters_summary.append(
                f"- {char.name} (ID: {char.id}): {char.role}, motive: {char.motive_for_crime}"
            )

        # Include full truth narrative for context
        truth_narrative = ""
        if state.killer_selection:
            truth_narrative = f"""
COMPLETE TRUTH NARRATIVE:
{state.killer_selection.truth_narrative}

KILLER RATIONALE:
{state.killer_selection.rationale}

KILLER'S REAL ROUTE:
{format_killer_route(state)}

DEDUCTION PLAN (the sheets must contain the observations it assigns):
{format_deduction_plan(state)}
"""

        user_message = f"""Validate this mystery party game for logical consistency:

GAME CONFIGURATION:
- Difficulty: {state.config.difficulty}
- Players: {len(state.characters)}
- Duration: {state.config.duration_minutes} minutes

VICTIM (HOST):
- {victim_name}: {safe_get_crime_victim_role(state)}

KILLER (SELECTED):
- {killer_name} (ID: {state.killer_selection.killer_id if state.killer_selection else "N/A"})

SUSPECTS:
{chr(10).join(characters_summary) if characters_summary else "No characters"}

CRIME:
- Method: {safe_get_crime_method_description(state)}
- Weapon: {safe_get_crime_weapon(state)}
- Location: {safe_get_crime_scene_room_id(state)}
- Time: {safe_get_crime_time_of_death(state)}

FLOOR PLAN (people move along connections and see only their room or the rooms visible from it):
{format_floor_plan(state)}

COMPLETE TIMELINE:
{format_global_timeline(state)}

PERSONAL TIMELINES (what each suspect really did, claims and observed):
{format_personal_timelines(state)}
{truth_narrative}

Also check that the killer's sheet follows the route, and that the observations the deduction plan assigns
are on their holders' sheets and possible from where the holders were.
"""

        # Invoke LLM with structured output
        result = self.invoke(state, user_message)

        # Update state
        state.validation = result

        return state

    def _mock_output(self, state: GameState) -> GameState:
        """Generate mock validation (always pass) for dry run mode."""
        state.validation = ValidationReport(is_consistent=True, issues=[], suggested_fixes=[])
        return state
