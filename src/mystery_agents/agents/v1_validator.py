"""V1: Validation Agent - Validates game logic consistency."""

from mystery_agents.models.state import GameState, ValidationReport
from mystery_agents.utils.cache import LLMCache
from mystery_agents.utils.prompts import V1_SYSTEM_PROMPT
from mystery_agents.utils.state_helpers import (
    safe_get_crime_method_description,
    safe_get_crime_scene_room_id,
    safe_get_crime_time_of_death,
    safe_get_crime_victim_name,
    safe_get_crime_victim_role,
    safe_get_crime_weapon,
)

from .base import BaseAgent


class ValidationAgent(BaseAgent):
    """
    V1: Validation Agent.

    Validates the entire game state for logical consistency.
    This is a tier 1 agent (logic - most powerful LLM).
    CRITICAL: This agent controls the retry loop.
    """

    def __init__(self) -> None:
        """Initialize the validation agent."""
        super().__init__(llm=LLMCache.get_model("tier1"), response_format=ValidationReport)

    def get_system_prompt(self, state: GameState) -> str:
        """
        Get the system prompt for validation.

        Args:
            state: Current game state

        Returns:
            System prompt string
        """
        return V1_SYSTEM_PROMPT

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

        timeline_events = []
        if state.timeline_global and state.timeline_global.time_blocks:
            for block in state.timeline_global.time_blocks:
                for event in block.events:
                    timeline_events.append(
                        f"[{block.start}-{block.end}] {event.time_approx}: {event.description} (characters: {event.character_ids_involved}, room: {event.room_id or 'N/A'})"
                    )

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

MODIFIED EVENTS (if any):
{chr(10).join(f"- {event}" for event in state.killer_selection.modified_events) if state.killer_selection.modified_events else "- None"}
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

COMPLETE TIMELINE ({len(timeline_events)} events):
{chr(10).join(timeline_events) if timeline_events else "No timeline events"}
{truth_narrative}

REQUIREMENTS:
1. Create a ValidationReport object with: is_consistent (boolean), issues (array), suggested_fixes (array)
2. If is_consistent is true, issues should be empty []
3. If is_consistent is false, provide at least one ValidationIssue in the issues array
4. Each ValidationIssue must have: type, description, related_ids (array, can be empty [])
5. suggested_fixes is an array of strings (can be empty [])
6. All string fields must have values - do not leave any empty

VALIDATION GUIDELINES:
- Check if the killer has motive, means, and a PLAUSIBLE opportunity (doesn't need to be perfect)
- Check if the timeline is REASONABLY consistent (minor gaps are acceptable if explained in truth_narrative)
- Check if the mystery can be solved with available information (some ambiguity is fine for medium difficulty)
- Check if all characters are integrated (they should have roles and motives)
- Only flag CRITICAL issues that make the game unplayable

IMPORTANT:
- Be REASONABLE: Minor inconsistencies that are explained in the truth_narrative should NOT prevent validation
- The truth_narrative explains what REALLY happened - use it to resolve timeline ambiguities
- If the killer has a plausible way to commit the crime (even if not every detail is in the timeline), that's sufficient
- Only mark is_consistent=false for issues that make the game TRULY unplayable or unsolvable

Return the response in the exact JSON format specified in the system prompt.
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
