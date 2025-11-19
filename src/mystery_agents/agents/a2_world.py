"""A2: World Agent - Creates the game setting with rich cultural details."""

from pydantic import BaseModel, Field

from mystery_agents.models.state import GameState, WorldBible
from mystery_agents.utils.cache import LLMCache
from mystery_agents.utils.constants import GAME_TONE_STYLE, MOCK_WORLD_NAME
from mystery_agents.utils.prompts import A2_WORLD_SYSTEM_PROMPT

from .base import BaseAgent


class A2Output(BaseModel):
    """Output format for A2 agent."""

    world: WorldBible = Field(
        description="The complete world and setting definition, including epoch, location, atmosphere, cultural details, and constraints."
    )


class WorldAgent(BaseAgent):
    """
    A2: World Agent.

    Creates a rich, detailed game world with cultural context based on country, epoch, and theme.
    This is a tier 1 agent (logic/creativity - most powerful LLM).
    """

    def __init__(self) -> None:
        """Initialize the world agent."""
        super().__init__(llm=LLMCache.get_model("tier1"), response_format=A2Output)

    def get_system_prompt(self, state: GameState) -> str:
        """
        Get the system prompt for world generation.

        Args:
            state: Current game state

        Returns:
            System prompt string
        """
        return A2_WORLD_SYSTEM_PROMPT

    def run(self, state: GameState) -> GameState:
        """
        Generate a rich, detailed world specification with cultural context.

        Args:
            state: Current game state with config

        Returns:
            Updated game state with world
        """
        # If dry run, return mock data
        if self._should_use_mock(state):
            return self._mock_output(state)

        # Prepare context for LLM with emphasis on cultural details
        region_context = f" in {state.config.region}" if state.config.region else ""
        location_spec = (
            f"{state.config.region}, {state.config.country}"
            if state.config.region
            else state.config.country
        )
        epoch_desc = (
            state.config.custom_epoch_description
            if state.config.custom_epoch_description
            else state.config.epoch
        )

        user_message = f"""Generate a rich, detailed mystery party game world based on these preferences:

CONFIGURATION:
- Target language: {state.config.language}
- Country: {state.config.country}
{f"- Region: {state.config.region} (CRITICAL: The setting MUST be located in this specific region{region_context}, not in other parts of {state.config.country})" if state.config.region else ""}
- Epoch: {epoch_desc} (CRITICAL: Use this EXACT value for the "epoch" field, and ALL world elements must be historically accurate for this time period)
- Theme: {state.config.theme}
{f"- Custom theme description: {state.config.custom_theme_description}" if state.config.custom_theme_description else ""}
- Tone: {GAME_TONE_STYLE}
- Number of suspects: {state.config.players.total}
- Difficulty: {state.config.difficulty}

HISTORICAL & CULTURAL REQUIREMENTS (ALL MUST BE SATISFIED):
1. **Location Authenticity**: The setting MUST be in {location_spec} - use the specific geography, climate, and landscape of this region
2. **Epoch Accuracy**: ALL elements (technology, transportation, currency, social norms, fashion) MUST be appropriate for {epoch_desc}
3. **Regional Culture**: Cultural details (foods, drinks, customs) MUST be authentic to {location_spec} as it was during {epoch_desc}
4. **Architecture**: Building style and visual elements MUST reflect {location_spec}'s architecture during {epoch_desc}
5. **Fashion & Dress**: Clothing and social dress codes appropriate for {location_spec} during {epoch_desc}
6. **Items & Weapons**: Only include items/weapons that would have existed and been culturally appropriate in {location_spec} during {epoch_desc}
7. **Local Traditions**: Festivals, customs, and social practices specific to {location_spec} during {epoch_desc}
8. **Historical Context**: Consider major historical events or social conditions affecting {location_spec} during {epoch_desc}
9. **NO Anachronisms**: Do not include technology, terms, or concepts that didn't exist in {epoch_desc}
10. **NO Regional Mixing**: If a region is specified, do NOT use cultural elements from other regions of {state.config.country}

REQUIREMENTS:
1. Create a "world" object with epoch, location_type, location_name, summary, gathering_reason, visual_keywords, and constraints
2. The "epoch" field MUST be EXACTLY: {epoch_desc} (copy this value exactly, do NOT modify, expand, or translate it)
3. The world should be rich in detail and feel authentic to {state.config.country} in the {epoch_desc} era
4. The gathering_reason must explain WHY all these people are at this location tonight (e.g., family event, memorial service, celebration, business meeting, ceremony)
5. Include cultural elements that will inform character creation, costume suggestions, and crime details later
6. The location should be suitable for a mystery party with {state.config.players.total} suspects
7. All string fields must have values - do not leave any empty
8. Visual keywords should reflect the cultural and historical atmosphere

Return the response in the exact JSON format specified in the system prompt.
"""

        # Invoke LLM with structured output
        result = self.invoke(state, user_message)

        # Update state
        state.world = result.world

        return state

    def _mock_output(self, state: GameState) -> GameState:
        """Generate mock data for dry run mode."""
        from mystery_agents.models.state import WorldBible

        state.world = WorldBible(
            epoch="Modern",
            location_type="Mansion",
            location_name=MOCK_WORLD_NAME,
            summary="A grand family mansion in the countryside, hosting a reunion.",
            gathering_reason="Annual family reunion to commemorate the patriarch's legacy",
            visual_keywords=["gothic", "elegant", "mysterious", "candlelit"],
            constraints=["Limited access to certain rooms", "No modern technology in some areas"],
        )

        return state
