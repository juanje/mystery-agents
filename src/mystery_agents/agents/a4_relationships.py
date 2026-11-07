"""A4: Relationships Agent - Creates connections between characters."""

from pydantic import BaseModel, Field

from mystery_agents.models.state import GameState, RelationshipSpec
from mystery_agents.utils.cache import LLMCache
from mystery_agents.utils.constants import GAME_TONE_STYLE
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


def merge_duplicate_pairs(relationships: list[RelationshipSpec]) -> list[RelationshipSpec]:
    """
    Keep one relationship per pair of characters, regardless of direction.

    The LLM sometimes returns the same pair twice with different types. Character
    sheets would then list that person twice, so the duplicates are folded into the
    most tense relationship, keeping every description.
    """
    merged: dict[frozenset[str], RelationshipSpec] = {}
    for rel in relationships:
        pair = frozenset((rel.from_character_id, rel.to_character_id))
        existing = merged.get(pair)
        if existing is None:
            merged[pair] = rel
            continue
        main, extra = (
            (existing, rel) if existing.tension_level >= rel.tension_level else (rel, existing)
        )
        merged[pair] = main.model_copy(
            update={"description": f"{main.description} {extra.description}"}
        )
    return list(merged.values())


class RelationshipsAgent(BaseAgent):
    """
    A4: Relationships Agent.

    Creates meaningful relationships between characters after they've been generated.
    This is a tier 2 agent (content generation - powerful LLM).
    """

    def __init__(self) -> None:
        """Initialize the relationships agent."""
        super().__init__(llm=LLMCache.get_model("tier2"), response_format=A4Output)

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
            summary += f"\n  Relation to victim: {char.relation_to_victim}"
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
- Tone: {GAME_TONE_STYLE}
- Victim (host's Act 1 character, created later): {state.config.host_gender}. If a description mentions the victim, use gendered terms that fit.

CHARACTERS:
{characters_info}

Create about {min_relationships} relationships.
"""

        result = self.invoke(state, user_message)

        state.relationships = merge_duplicate_pairs(result.relationships)

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
