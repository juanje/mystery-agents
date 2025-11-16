"""V1: World Validator - Validates world consistency and coherence."""

from pydantic import BaseModel, Field

from mystery_agents.models.state import GameState
from mystery_agents.utils.cache import LLMCache
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

        # Build critical validation checks
        critical_checks = []
        if state.config.region:
            critical_checks.append(f"""
**CRITICAL REGION CHECK**:
- Configuration specifies: {state.config.region}
- The world MUST be set in {state.config.region}, NOT in other regions of {state.config.country}
- Cultural elements MUST be specific to {state.config.region}, not borrowed from other regions
- Check if summary/location/culture reference regions OTHER than {state.config.region}
- Flag as is_coherent=false if geographical mismatch detected""")

        critical_checks.append(f"""
**CRITICAL EPOCH CHECK**:
- Configuration specifies: {epoch_desc}
- ALL elements must be historically accurate for this period
- Check for anachronisms: technology, currency, transportation, social concepts that didn't exist yet
- Verify cultural practices, fashion, and customs match the time period
- Flag as is_coherent=false if historical inaccuracies detected""")

        critical_section = "\n".join(critical_checks)

        user_message = f"""Validate the coherence and consistency of this game world:

CONFIGURATION:
- Country: {state.config.country}{region_info}
- Epoch: {state.config.epoch}
{f"- Custom epoch description: {state.config.custom_epoch_description}" if state.config.custom_epoch_description else ""}
- Theme: {state.config.theme}
- Tone: {state.config.tone}

GENERATED WORLD:
- Location Name: {state.world.location_name}
- Location Type: {state.world.location_type}
- Epoch: {state.world.epoch}
- Summary: {state.world.summary}
- Visual Keywords: {", ".join(state.world.visual_keywords)}
- Constraints: {", ".join(state.world.constraints)}
{critical_section}

VALIDATION CRITERIA:

1. **Historical Accuracy** (CRITICAL):
   - Does the location/setting exist or could it have existed in {epoch_desc}?
   - Are ALL references to technology, currency, transportation appropriate for {epoch_desc}?
   - Are social customs, norms, and class structures appropriate for {epoch_desc}?
   - Are there any anachronisms (modern terms, concepts, or items that didn't exist)?

2. **Geographical Consistency** (CRITICAL if region specified):
   - Is the location geographically plausible for {state.config.country}{region_info}?
   - If it's a specific place (train, ship, building), could it realistically exist there in {epoch_desc}?
   - Are geographical references accurate to {state.config.country}{region_info}?
   - Does it avoid mixing cultural/geographical elements from different regions?

3. **Cultural Authenticity** (CRITICAL):
   - Does the setting reflect the cultural identity of {state.config.country}{region_info} during {epoch_desc}?
   - Are architectural styles, foods, drinks, customs appropriate for the location AND time period?
   - Does it respect regional differences within the country (if region specified)?
   - Are festivals, traditions, and social practices historically and culturally accurate?

4. **Internal Consistency**:
   - Do all elements (location_type, location_name, epoch, summary) align with each other?
   - Are there contradictions between different parts of the world description?

IMPORTANT:
- Be thorough but reasonable
- Flag CRITICAL issues (historical impossibilities, geographical errors, major cultural misrepresentations)
- Minor creative liberties are acceptable if they don't break immersion
- If is_coherent=False, provide specific, actionable suggestions for fixes

Return the validation result in JSON format."""

        result = self.invoke(state, user_message)

        # Store validation result in state
        from mystery_agents.models.state import WorldValidation

        state.world_validation = WorldValidation(
            is_coherent=result.is_coherent,
            issues=result.issues,
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
