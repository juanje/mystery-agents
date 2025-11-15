"""A3: Characters Agent - Creates suspect characters."""

from pydantic import BaseModel, Field

from mystery_agents.models.state import CharacterSpec, GameState
from mystery_agents.utils.cache import LLMCache
from mystery_agents.utils.prompts import A3_SYSTEM_PROMPT
from mystery_agents.utils.state_helpers import (
    safe_get_world_epoch,
    safe_get_world_location_name,
    safe_get_world_location_type,
)

from .base import BaseAgent


class A3Output(BaseModel):
    """Output format for A3 agent."""

    characters: list[CharacterSpec] = Field(
        description="List of suspect characters. Must contain exactly the number of players specified."
    )


class CharactersAgent(BaseAgent):
    """
    A3: Characters Agent.

    Creates suspect characters (relationships are created by A4).
    This is a tier 2 agent (content generation - powerful LLM).
    """

    def __init__(self) -> None:
        """Initialize the characters agent."""
        super().__init__(llm=LLMCache.get_model("tier2"), response_format=A3Output)

    def get_system_prompt(self, state: GameState) -> str:
        """
        Get the system prompt for character generation.

        Args:
            state: Current game state

        Returns:
            System prompt string with player count filled in
        """
        num_players = state.config.players.total
        return A3_SYSTEM_PROMPT.format(num_players=num_players)

    def run(self, state: GameState) -> GameState:
        """
        Generate characters only (relationships created by A4).

        Args:
            state: Current game state with world

        Returns:
            Updated game state with characters
        """
        # If dry run, return mock data
        if self._should_use_mock(state):
            return self._mock_output(state)

        # Prepare context for LLM
        gender_constraints = ""
        if state.config.players.male > 0:
            gender_constraints += f"\n- {state.config.players.male} male characters"
        if state.config.players.female > 0:
            gender_constraints += f"\n- {state.config.players.female} female characters"

        user_message = f"""Generate {state.config.players.total} suspect characters for this mystery:

WORLD:
- Setting: {safe_get_world_location_name(state)}
- Epoch: {safe_get_world_epoch(state)}
- Type: {safe_get_world_location_type(state)}
- Country: {state.config.country}
- Summary: {state.world.summary if state.world else "Mystery party setting"}

NOTE: The victim (host's Act 1 character) will be created later. Focus on creating interesting suspects who would naturally be at this location.

CONSTRAINTS:
- Tone: {state.config.tone}
- Total suspects: {state.config.players.total}
{gender_constraints if gender_constraints else "- No specific gender requirements"}

REQUIREMENTS:
1. Create exactly {state.config.players.total} characters in the "characters" array
2. Each character MUST have ALL of these fields:
   - name, gender, age_range, role, public_description
   - personality_traits: array of at least 3-5 personality traits (e.g., ["clever", "suspicious", "charming", "manipulative", "loyal"])
   - relation_to_victim, personal_secrets (at least 2-3 secrets), personal_goals (at least 1-2 goals)
   - act1_objectives: array of 2-3 specific actionable tasks for Act 1 (REQUIRED)
   - motive_for_crime, costume_suggestion
3. All character names MUST be appropriate for {state.config.country}
4. Characters should fit naturally into the world setting ({safe_get_world_location_name(state)}) and epoch ({safe_get_world_epoch(state)})
5. Relationships between characters will be created by a separate agent later

Return the response in the exact JSON format specified in the system prompt.
"""

        # Invoke LLM with structured output
        result = self.invoke(state, user_message)

        # Update state
        state.characters = result.characters

        return state

    def _mock_output(self, state: GameState) -> GameState:
        """Generate mock data for dry run mode."""
        num_players = state.config.players.total

        # Create mock characters
        characters = []
        for i in range(num_players):
            char = CharacterSpec(
                name=f"Character {i + 1}",
                gender="male" if i % 2 == 0 else "female",
                age_range="30-40",
                role=f"Role {i + 1}",
                public_description="A mysterious figure with connections to the victim.",
                personality_traits=["clever", "suspicious", "charming"],
                relation_to_victim=f"Relationship {i + 1}",
                personal_secrets=[f"Secret {i + 1}"],
                personal_goals=[f"Goal {i + 1}"],
                act1_objectives=[f"Objective {i + 1}", f"Objective {i + 2}"],
                motive_for_crime=f"Motive {i + 1}",
                costume_suggestion=f"Costume suggestion {i + 1}",
            )
            characters.append(char)

        state.characters = characters

        return state
