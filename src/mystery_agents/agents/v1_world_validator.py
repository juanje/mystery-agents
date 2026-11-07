"""V1: World Validator - Validates world consistency and coherence."""

from pydantic import BaseModel, Field

from mystery_agents.models.state import GameState
from mystery_agents.utils.cache import LLMCache
from mystery_agents.utils.constants import GAME_TONE_STYLE
from mystery_agents.utils.floor_plan import floor_plan_problems, format_floor_plan
from mystery_agents.utils.prompts import V1_WORLD_VALIDATOR_SYSTEM_PROMPT

from .base import BaseAgent


class V1Output(BaseModel):
    """Output format for V1 World Validator agent."""

    is_coherent: bool = Field(
        description="Whether the world is historically, geographically, and culturally coherent"
    )
    issues: list[str] = Field(
        default_factory=list,
        description="List of coherence issues found (empty if is_coherent=True)",
    )
    suggestions: list[str] = Field(
        default_factory=list,
        description="Suggested improvements or corrections (empty if is_coherent=True)",
    )


class WorldValidatorAgent(BaseAgent):
    """
    V1: World Validator Agent.

    Validates the generated world for historical, geographical, and cultural coherence.
    This is a tier 2 agent (analysis - powerful LLM).
    Runs FIRST in the validation sequence (after A2: World Generation).
    """

    def __init__(self) -> None:
        """Initialize the world validator agent."""
        super().__init__(llm=LLMCache.get_model("tier2"), response_format=V1Output)

    def get_system_prompt(self, state: GameState) -> str:
        """
        Get the system prompt for world validation.

        Args:
            state: Current game state

        Returns:
            System prompt string
        """
        return V1_WORLD_VALIDATOR_SYSTEM_PROMPT

    def run(self, state: GameState) -> GameState:
        """
        Validate the generated world for coherence.

        Args:
            state: Current game state with world

        Returns:
            Updated game state with world_validation
        """
        if self._should_use_mock(state):
            return self._mock_output(state)

        if not state.world:
            raise ValueError("World must be generated before validation")

        # Prepare context for validation
        region_info = f" (Region: {state.config.region})" if state.config.region else ""
        epoch_desc = (
            state.config.custom_epoch_description
            if state.config.custom_epoch_description
            else state.config.epoch
        )

        user_message = f"""Check this world against its place and time:

CONFIGURATION:
- Country: {state.config.country}{region_info}
- Epoch: {epoch_desc}
- Theme: {state.config.theme}
- Tone: {GAME_TONE_STYLE}

WORLD:
- Location: {state.world.location_name} ({state.world.location_type})
- Summary: {state.world.summary}
- Visual keywords: {", ".join(state.world.visual_keywords)}
- Constraints: {", ".join(state.world.constraints)}
- Victim (the host's role): {state.world.host.name}, {state.world.host.age}, {state.world.host.role_in_setting}
- Gathering reason: {state.world.gathering_reason}

FLOOR PLAN:
{format_floor_plan(state)}
"""

        result = self.invoke(state, user_message)

        # Store validation result in state
        from mystery_agents.models.state import WorldValidation

        # Whether the plan can stage the evening is measurable, so code checks it
        plan_problems = floor_plan_problems(state.world)
        state.world_validation = WorldValidation(
            is_coherent=result.is_coherent and not plan_problems,
            issues=[*result.issues, *plan_problems],
            suggestions=result.suggestions,
        )

        return state

    def _mock_output(self, state: GameState) -> GameState:
        """Generate mock validation for dry run mode."""
        from mystery_agents.models.state import WorldValidation

        state.world_validation = WorldValidation(
            is_coherent=True,
            issues=[],
            suggestions=[],
        )
        return state
