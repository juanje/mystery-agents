"""A2: World Agent - Creates the game setting with rich cultural details."""

from pydantic import BaseModel, Field

from mystery_agents.models.state import GameState, WorldBible
from mystery_agents.utils.cache import LLMCache
from mystery_agents.utils.constants import GAME_TONE_STYLE, MOCK_VICTIM_NAME, MOCK_WORLD_NAME
from mystery_agents.utils.floor_plan import normalize_floor_plan
from mystery_agents.utils.prompts import A2_WORLD_SYSTEM_PROMPT
from mystery_agents.utils.state_helpers import format_retry_feedback

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
        epoch_desc = (
            state.config.custom_epoch_description
            if state.config.custom_epoch_description
            else state.config.epoch
        )

        # A retry after V1 must know what was wrong with the previous world
        report = state.world_validation
        retry_feedback = (
            format_retry_feedback(report.issues, report.suggestions)
            if report and not report.is_coherent
            else ""
        )

        user_message = f"""Create the world for this game:

- Language of every name and text: {state.config.language}
- Country: {state.config.country}
{f"- Region: {state.config.region}. The setting is in this region, with its own culture, not elsewhere in {state.config.country}" if state.config.region else ""}
- Epoch: {epoch_desc}
- Theme: {state.config.theme}
{f"- Theme description: {state.config.custom_theme_description}" if state.config.custom_theme_description else ""}
- Tone: {GAME_TONE_STYLE}
- Suspects: {state.config.players.total}, plus the victim, who is {state.config.host_gender}
- Difficulty: {state.config.difficulty}
{retry_feedback}"""

        # Invoke LLM with structured output
        result = self.invoke(state, user_message)

        # Later agents and V1 compare against the configured epoch, so it is copied, not asked for
        state.world = normalize_floor_plan(result.world.model_copy(update={"epoch": epoch_desc}))

        return state

    def _mock_output(self, state: GameState) -> GameState:
        """Generate mock data for dry run mode."""
        from mystery_agents.models.state import HostCharacter, Room, WorldBible

        state.world = WorldBible(
            epoch="Modern",
            location_type="Mansion",
            location_name=MOCK_WORLD_NAME,
            summary="A grand family mansion in the countryside, hosting a reunion.",
            gathering_reason="Annual family reunion to commemorate the patriarch's legacy",
            visual_keywords=["gothic", "elegant", "mysterious", "candlelit"],
            constraints=["Limited access to certain rooms", "No modern technology in some areas"],
            host=HostCharacter(
                name=MOCK_VICTIM_NAME, age=65, role_in_setting="Owner of the mansion"
            ),
            rooms=[
                Room(
                    name="Drawing room",
                    floor="Ground floor",
                    description="Where the guests gather for drinks.",
                    connects_to=["Hall", "Garden"],
                ),
                Room(
                    name="Hall",
                    floor="Ground floor",
                    description="Entrance hall with the main stairs.",
                    connects_to=["Drawing room", "Study"],
                ),
                Room(
                    name="Study",
                    floor="Ground floor",
                    description="The victim's private study.",
                    connects_to=["Hall", "Garden"],
                ),
                Room(
                    name="Garden",
                    floor="Outside",
                    description="Gardens behind the house.",
                    connects_to=["Drawing room", "Study"],
                    can_see=["Study"],
                ),
            ],
        )

        return state
