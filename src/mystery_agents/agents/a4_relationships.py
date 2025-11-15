"""A4: Relationships Agent - Creates connections between characters."""

from pydantic import BaseModel, Field

from mystery_agents.config import LLMConfig
from mystery_agents.models.state import GameState, RelationshipSpec
from mystery_agents.utils.prompts import A4_RELATIONSHIPS_SYSTEM_PROMPT
from mystery_agents.utils.state_helpers import (
    safe_get_world_epoch,
    safe_get_world_location_name,
)

from .base import BaseAgent


class A4Output(BaseModel):
    """Output format for A4 agent."""

    relationships: list[RelationshipSpec] = Field(
        description="List of relationships between characters. Each character should have at least 1-2 meaningful relationships."
    )


class RelationshipsAgent(BaseAgent):
    """
    A4: Relationships Agent.

    Creates meaningful relationships between characters after they've been generated.
    This is a tier 2 agent (content generation - powerful LLM).
    """

    def __init__(self) -> None:
        """Initialize the relationships agent."""
        super().__init__(llm=LLMConfig.get_model("tier2"), response_format=A4Output)

    def get_system_prompt(self, state: GameState) -> str:
        """
        Get the system prompt for relationship generation.

        Args:
            state: Current game state

        Returns:
            System prompt string
        """
        return A4_RELATIONSHIPS_SYSTEM_PROMPT

    def run(self, state: GameState) -> GameState:
        """
        Generate relationships between existing characters.

        Args:
            state: Current game state with world and characters

        Returns:
            Updated game state with relationships
        """
        if self._should_use_mock(state):
            return self._mock_output(state)

        # Prepare character summary for context
        character_summaries = []
        for char in state.characters:
            summary = f"- {char.name} (ID: {char.id}): {char.role}, age {char.age_range}"
            summary += f"\n  Public: {char.public_description}"
            summary += f"\n  Secrets: {', '.join(char.personal_secrets[:2]) if char.personal_secrets else 'None'}"
            character_summaries.append(summary)

        characters_info = "\n".join(character_summaries)

        min_relationships = len(state.characters) * 2

        user_message = f"""Create meaningful relationships between these {len(state.characters)} characters:

WORLD CONTEXT:
- Setting: {safe_get_world_location_name(state)}
- Epoch: {safe_get_world_epoch(state)}
- Country: {state.config.country}
- Region: {state.config.region if state.config.region else "N/A"}
- Theme: {state.config.theme}
- Tone: {state.config.tone}

CHARACTERS:
{characters_info}

REQUIREMENTS:
1. **CRITICAL**: Create AT LEAST {min_relationships} relationships (minimum 2-3 per character)
2. **MANDATORY**: Every character MUST appear in at least 2 relationships (either as from_character_id or to_character_id)
3. Create diverse relationship types: family, romantic, professional, rivalry, friendship, other
4. At least 50% of relationships should have tension_level 2 or 3 (high drama)
5. Consider the setting, epoch, and character roles when creating relationships
6. Make relationships that will create interesting dynamics and potential conflicts during the mystery
7. Each relationship needs:
   - from_character_id: ID of first character (must match a character ID above EXACTLY)
   - to_character_id: ID of second character (must match a character ID above EXACTLY)
   - type: "family" | "romantic" | "professional" | "rivalry" | "friendship" | "other"
   - description: Specific, evocative description (e.g., "Former business partners who had a bitter falling out over a failed shipping deal", NOT just "business partners")
   - tension_level: 1 (low), 2 (medium), or 3 (high) - prefer 2 or 3 for better gameplay
8. Relationships should reveal character depth and create natural conflicts
9. Mix symmetric relationships (mutual) with asymmetric ones (one-sided feelings or power dynamics)
10. Ensure no character is isolated - all must be connected to the social web

**IMPORTANT**: The richer the relationship network, the more engaging the mystery party will be. Aim for depth and complexity.

Return the response in the exact JSON format specified in the system prompt.
"""

        result = self.invoke(state, user_message)

        state.relationships = result.relationships

        return state

    def _mock_output(self, state: GameState) -> GameState:
        """Generate mock relationships for dry run mode."""
        if len(state.characters) >= 2:
            state.relationships = [
                RelationshipSpec(
                    from_character_id=state.characters[0].id,
                    to_character_id=state.characters[1].id,
                    type="professional",
                    description="Former business partners with unresolved disputes",
                    tension_level=2,
                )
            ]
            if len(state.characters) >= 3:
                state.relationships.append(
                    RelationshipSpec(
                        from_character_id=state.characters[0].id,
                        to_character_id=state.characters[2].id,
                        type="rivalry",
                        description="Competing for the same professional recognition",
                        tension_level=3,
                    )
                )

        return state
