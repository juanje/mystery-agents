"""Host Character Image Generation Agent (A8.5) - Generates victim and detective images."""

from __future__ import annotations

from pathlib import Path

from langchain_core.language_models import BaseChatModel

from mystery_agents.agents.base import BaseAgent
from mystery_agents.models.state import DetectiveRole, GameState, VictimSpec
from mystery_agents.utils.constants import GAME_ID_LENGTH
from mystery_agents.utils.image_generation import (
    generate_image_with_gemini,
    get_character_image_output_dir,
    portrait_filename,
)
from mystery_agents.utils.prompts import (
    PORTRAIT_COMPOSITION_REQUIREMENTS,
    REALISTIC_APPEARANCE_REQUIREMENTS,
    build_fallback_style_requirements,
    build_visual_style_block,
)
from mystery_agents.utils.state_helpers import (
    safe_get_world_epoch,
    safe_get_world_location_name,
)


class HostImageAgent(BaseAgent):
    """
    Agent that generates character portrait images for host characters (victim and detective).

    Features:
    - Generates images for victim (Act 1) and detective (Act 2)
    - Same technology as CharacterImageAgent (Gemini Image API)
    - Exponential backoff for rate limit errors
    - Mock generation in dry-run mode
    """

    def __init__(self, llm: BaseChatModel | None = None) -> None:
        """
        Initialize the host image generation agent.

        Args:
            llm: The language model (not used for image generation,
                 optional for compatibility with base class)
        """
        # Image generation doesn't use LLM, so we create a dummy one if not provided
        from mystery_agents.utils.cache import LLMCache

        if llm is None:
            llm = LLMCache.get_model("tier3")  # Cheapest tier, won't be used anyway

        super().__init__(llm, response_format=None)

    def get_system_prompt(self, state: GameState) -> str:
        """
        Not used for image generation.

        Args:
            state: Current game state

        Returns:
            Empty string (not applicable for image generation)
        """
        return ""

    def run(self, state: GameState) -> GameState:
        """
        Generate host character images (victim and detective).

        Args:
            state: Current game state with crime (victim) and host_guide (detective)

        Returns:
            Updated state with image_path populated for victim and detective
        """
        # Skip if image generation is not enabled
        if not state.config.generate_images:
            return state

        if self._should_use_mock(state):
            return self._mock_output(state)

        # Check if we have victim and detective
        has_victim = state.crime and state.crime.victim
        has_detective = state.host_guide and state.host_guide.host_act2_detective_role

        if not has_victim and not has_detective:
            return state

        # Create output directory for images
        output_dir = self._output_dir(state)
        output_dir.mkdir(parents=True, exist_ok=True)

        # Generate victim image
        if state.crime and state.crime.victim:
            self._generate_victim_image_sync(state.crime.victim, state, output_dir)

        # Generate detective image
        if state.host_guide and state.host_guide.host_act2_detective_role:
            self._generate_detective_image_sync(
                state.host_guide.host_act2_detective_role, state, output_dir
            )

        # Return updated state
        return state

    def _generate_victim_image_sync(
        self, victim: VictimSpec, state: GameState, output_dir: Path
    ) -> None:
        """
        Generate image for the victim character (synchronous wrapper).

        Args:
            victim: Victim specification
            state: Current game state
            output_dir: Directory to save image
        """
        import asyncio

        async def generate() -> None:
            prompt = self._build_victim_image_prompt(victim, state)
            image_path = output_dir / portrait_filename(victim.id, victim.name)

            success = await generate_image_with_gemini(prompt, image_path)

            if success:
                victim.image_path = str(image_path.absolute())
            else:
                victim.image_path = None

        asyncio.run(generate())

    def _generate_detective_image_sync(
        self, detective: DetectiveRole, state: GameState, output_dir: Path
    ) -> None:
        """
        Generate image for the detective character (synchronous wrapper).

        Args:
            detective: Detective role specification
            state: Current game state
            output_dir: Directory to save image
        """
        import asyncio

        async def generate() -> None:
            prompt = self._build_detective_image_prompt(detective, state)
            image_path = output_dir / self._detective_filename(detective, state)

            success = await generate_image_with_gemini(prompt, image_path)

            if success:
                detective.image_path = str(image_path.absolute())
            else:
                detective.image_path = None

        asyncio.run(generate())

    def _build_victim_image_prompt(self, victim: VictimSpec, state: GameState) -> str:
        """
        Build a detailed image generation prompt for the victim character.

        Args:
            victim: Victim specification
            state: Current game state

        Returns:
            Detailed prompt for image generation
        """
        # Get world context
        epoch = safe_get_world_epoch(state)
        location = safe_get_world_location_name(state)
        country = state.config.country if state.config else "Unknown"

        # Build detailed prompt
        personality = (
            ", ".join(victim.personality_traits)
            if victim.personality_traits
            else "mysterious, commanding"
        )

        prompt = f"""Generate a photorealistic portrait of a {victim.gender} character for a mystery party game.

{PORTRAIT_COMPOSITION_REQUIREMENTS}

{REALISTIC_APPEARANCE_REQUIREMENTS}

CHARACTER DETAILS:
- Name: {victim.name}
- Age: {victim.age}
- Role: {victim.role_in_setting}
- Description: {victim.public_persona}
- Personality: {personality}
- Context: This is the VICTIM of the mystery - a central figure who will be murdered

SETTING CONTEXT:
- Historical Period: {epoch}
- Location: {location}
- Country/Culture: {country}

COSTUME:
{victim.costume_suggestion if victim.costume_suggestion else f"Period-appropriate formal attire for {epoch} in {country}"}
"""

        # Add visual style consistency if available
        if state.visual_style:
            prompt += build_visual_style_block(
                state.visual_style,
                emphasis="This is the VICTIM - a central, authoritative figure with commanding presence",
            )
        else:
            # Fallback if no visual style
            prompt += build_fallback_style_requirements(epoch, country, personality, "victim")

        return prompt

    def _build_detective_image_prompt(self, detective: DetectiveRole, state: GameState) -> str:
        """
        Build a detailed image generation prompt for the detective character.

        Args:
            detective: Detective role specification
            state: Current game state

        Returns:
            Detailed prompt for image generation
        """
        # Get world context
        epoch = safe_get_world_epoch(state)
        location = safe_get_world_location_name(state)
        country = state.config.country if state.config else "Unknown"

        # Build detailed prompt
        personality = (
            ", ".join(detective.personality_traits)
            if detective.personality_traits
            else "analytical, observant, methodical"
        )

        prompt = f"""Generate a photorealistic portrait of a detective character for a mystery party game.

{PORTRAIT_COMPOSITION_REQUIREMENTS}

{REALISTIC_APPEARANCE_REQUIREMENTS}

CHARACTER DETAILS:
- Name: {detective.character_name}
- Role: Detective investigating the murder
- Description: {detective.public_description}
- Personality: {personality}
- Context: This is the DETECTIVE who will solve the mystery in Act 2

SETTING CONTEXT:
- Historical Period: {epoch}
- Location: {location}
- Country/Culture: {country}

COSTUME:
{detective.costume_suggestion if detective.costume_suggestion else f"Classic detective attire for {epoch} in {country}"}
"""

        # Add visual style consistency if available
        if state.visual_style:
            prompt += build_visual_style_block(
                state.visual_style,
                emphasis="This is the DETECTIVE - sharp, intelligent, investigative presence with perceptive gaze",
            )
        else:
            # Fallback if no visual style
            prompt += build_fallback_style_requirements(epoch, country, personality, "detective")

        return prompt

    @staticmethod
    def _output_dir(state: GameState) -> Path:
        """Directory for the host portraits, shared with the suspects' portraits."""
        return get_character_image_output_dir(
            state.meta.id[:GAME_ID_LENGTH], state.config.output_dir
        )

    @staticmethod
    def _detective_filename(detective: DetectiveRole, state: GameState) -> str:
        """The detective has no ID of their own, so the game's names their portrait."""
        return portrait_filename(
            f"detective-{state.meta.id[:GAME_ID_LENGTH]}", detective.character_name
        )

    def _mock_output(self, state: GameState) -> GameState:
        """
        Generate mock image paths for dry run mode.

        Args:
            state: Current game state

        Returns:
            State with mock image paths
        """
        output_dir = self._output_dir(state)
        # Create directory structure even in dry-run mode for consistency
        output_dir.mkdir(parents=True, exist_ok=True)

        # Mock victim image
        if state.crime and state.crime.victim:
            victim = state.crime.victim
            mock_filename = portrait_filename(victim.id, victim.name)
            victim.image_path = str((output_dir / mock_filename).absolute())

        # Mock detective image
        if state.host_guide and state.host_guide.host_act2_detective_role:
            detective = state.host_guide.host_act2_detective_role
            mock_filename = self._detective_filename(detective, state)
            detective.image_path = str((output_dir / mock_filename).absolute())

        return state
