"""A2.5: Visual Style Agent - Creates consistent visual style for character images."""

from pydantic import BaseModel, Field

from mystery_agents.models.state import GameState, VisualStyle
from mystery_agents.utils.cache import LLMCache
from mystery_agents.utils.prompts import A2_5_VISUAL_STYLE_SYSTEM_PROMPT

from .base import BaseAgent


class A2_5Output(BaseModel):
    """Output format for A2.5 Visual Style agent."""

    visual_style: VisualStyle = Field(
        description="The complete visual style guide for consistent character image generation."
    )


class VisualStyleAgent(BaseAgent):
    """
    A2.5: Visual Style Agent.

    Creates a cohesive visual style guide based on the world setting to ensure
    all character images have a consistent aesthetic.

    This is a tier 2 agent (creativity/cultural awareness).
    """

    def __init__(self) -> None:
        """Initialize the visual style agent."""
        super().__init__(llm=LLMCache.get_model("tier2"), response_format=A2_5Output)

    def get_system_prompt(self, state: GameState) -> str:
        """Return the system prompt for visual style generation."""
        return A2_5_VISUAL_STYLE_SYSTEM_PROMPT

    def run(self, state: GameState) -> GameState:
        """
        Generate visual style guide.

        Args:
            state: Current game state (requires world to be set)

        Returns:
            Updated state with visual_style populated
        """
        if self._should_use_mock(state):
            return self._mock_output(state)

        # Verify world exists
        if not state.world:
            raise ValueError("Cannot generate visual style without world. Run A2 first.")

        # Build context message
        epoch = state.world.epoch
        country = state.config.country if state.config else "Unknown"
        location_name = state.world.location_name
        gathering_reason = state.world.gathering_reason
        visual_keywords = (
            ", ".join(state.world.visual_keywords)
            if state.world.visual_keywords
            else "elegant, mysterious"
        )

        user_message = f"""Create a cohesive visual style guide for character portrait images.

GAME CONTEXT:
- Historical Period: {epoch}
- Country/Culture: {country}
- Location: {location_name}
- Gathering Reason: {gathering_reason}
- Visual Keywords: {visual_keywords}

REQUIREMENTS:
1. All character images should look like they're from the same "photoshoot" or "film production"
2. COMPOSITION: All portraits must be BUST SHOTS (chest and head visible, like Clue/Cluedo game character cards)
3. REALISM: Characters should look like REAL, EVERYDAY PEOPLE, not models or actors. Natural faces with realistic features.
4. The style must be appropriate for {epoch} in {country}
5. Consider historical photography/portrait art styles from {epoch}
6. The color palette should reflect {country}'s cultural aesthetic in {epoch}
7. Lighting should create mystery and drama while being period-appropriate
8. The style should be sophisticated and elegant
9. Explicitly exclude: any text/labels, pure black & white (unless historically required), modern elements
10. All images must be in FULL COLOR unless the period absolutely demands otherwise

Return the visual style guide in the exact JSON format specified in the system prompt."""

        # Invoke LLM with structured output
        result = self.invoke(state, user_message)

        # Update state
        state.visual_style = result.visual_style

        return state

    def _mock_output(self, state: GameState) -> GameState:
        """Generate mock data for dry run mode."""
        state.visual_style = VisualStyle(
            style_description="Modern photorealistic portrait photography",
            art_direction="Sophisticated mystery game aesthetic",
            color_palette=["neutral tones", "warm skin tones", "subtle shadows"],
            color_grading="Natural color grading with slight warmth",
            lighting_setup="Professional portrait lighting with soft key light",
            lighting_mood="Elegant and slightly dramatic",
            background_aesthetic="Subtle, blurred neutral background",
            background_blur="Shallow depth of field, f/2.8",
            technical_specs="8K resolution, professional portrait photography",
            camera_specs="85mm portrait lens, f/2.8, professional DSLR",
            negative_prompts=[
                "text",
                "labels",
                "names",
                "watermarks",
                "black and white",
                "grayscale",
                "modern smartphones",
                "contemporary casual wear",
                "cartoon",
                "anime",
            ],
            period_references=["Contemporary professional portrait photography"],
        )

        return state
