"""Character Image Generation Agent (A3.5) with parallel processing."""

from __future__ import annotations

import asyncio
import logging
from pathlib import Path

from langchain_core.language_models import BaseChatModel

from mystery_agents.agents.base import BaseAgent
from mystery_agents.models.state import CharacterSpec, GameState
from mystery_agents.utils.constants import IMAGE_GENERATION_MAX_CONCURRENT
from mystery_agents.utils.image_generation import (
    generate_image_with_gemini,
    get_character_image_output_dir,
)
from mystery_agents.utils.state_helpers import (
    safe_get_world_epoch,
    safe_get_world_location_name,
)

logger = logging.getLogger(__name__)


class CharacterImageAgent(BaseAgent):
    """
    Agent that generates character portrait images using Gemini Image API.

    Features:
    - Parallel image generation with rate limiting (respects Gemini API limits)
    - Semaphore-based concurrency control
    - Exponential backoff for rate limit errors
    - Mock generation in dry-run mode
    """

    MAX_CONCURRENT_REQUESTS = IMAGE_GENERATION_MAX_CONCURRENT

    def __init__(self, llm: BaseChatModel | None = None) -> None:
        """
        Initialize the character image generation agent.

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
        Generate character images in parallel with rate limiting.

        Args:
            state: Current game state with characters

        Returns:
            Updated state with image_path populated for each character
        """
        # Skip if image generation is not enabled
        if not state.config.generate_images:
            logger.info("[A3.5] ⊘ Image generation disabled, skipping")
            return state

        if self._should_use_mock(state):
            logger.info("[A3.5] 🎭 Character Images: Dry run mode - using mocks")
            return self._mock_output(state)

        if not state.characters:
            logger.warning("[A3.5] ⚠️  No characters found, skipping image generation")
            return state

        logger.info(
            f"[A3.5] 🎨 Generating {len(state.characters)} character images in parallel "
            f"(max {self.MAX_CONCURRENT_REQUESTS} concurrent)"
        )

        # Create output directory for images
        output_dir = self._get_image_output_dir(state)
        output_dir.mkdir(parents=True, exist_ok=True)

        # Generate images in parallel using asyncio
        asyncio.run(self._generate_all_images(state, output_dir))

        # Log success with paths for debugging
        images_with_paths = 0
        for char in state.characters:
            if char.image_path:
                logger.info(f"[A3.5] 📸 {char.name}: {char.image_path}")
                images_with_paths += 1
            else:
                logger.warning(f"[A3.5] ⚠️  {char.name}: No image path set")

        logger.info(
            f"[A3.5] ✅ All character images generated in {output_dir} "
            f"({images_with_paths}/{len(state.characters)} with paths)"
        )

        # Return updated state with modified characters
        # This ensures LangGraph sees the changes
        return state.model_copy(update={"characters": state.characters})

    async def _generate_all_images(self, state: GameState, output_dir: Path) -> None:
        """
        Generate all character images in parallel with concurrency control.

        Args:
            state: Current game state
            output_dir: Directory to save images
        """
        # Create semaphore to limit concurrent requests
        semaphore = asyncio.Semaphore(self.MAX_CONCURRENT_REQUESTS)

        # Create tasks for all characters
        tasks = [
            self._generate_character_image_with_semaphore(character, state, output_dir, semaphore)
            for character in state.characters
        ]

        # Wait for all images to be generated
        await asyncio.gather(*tasks)

    async def _generate_character_image_with_semaphore(
        self,
        character: CharacterSpec,
        state: GameState,
        output_dir: Path,
        semaphore: asyncio.Semaphore,
    ) -> None:
        """
        Generate a single character image with semaphore-based rate limiting.

        Args:
            character: Character specification
            state: Current game state
            output_dir: Directory to save image
            semaphore: Asyncio semaphore for concurrency control
        """
        async with semaphore:
            await self._generate_character_image(character, state, output_dir)

    async def _generate_character_image(
        self, character: CharacterSpec, state: GameState, output_dir: Path
    ) -> None:
        """
        Generate image for a single character with retry logic.

        Args:
            character: Character specification
            state: Current game state
            output_dir: Directory to save image
        """
        prompt = self._build_image_prompt(character, state)
        image_filename = f"{character.id}_{character.name.lower().replace(' ', '_')}.png"
        image_path = output_dir / image_filename

        logger.info(f"[A3.5] 🎨 Generating image for {character.name}")

        # Generate image with retry logic
        success = await generate_image_with_gemini(prompt, image_path)

        if success:
            # Update character with image path (absolute path for robustness)
            character.image_path = str(image_path.absolute())
            logger.info(f"[A3.5] ✅ Generated: {character.name} -> {image_path.name}")
        else:
            logger.error(f"[A3.5] ❌ Failed to generate image for {character.name}")
            # Don't fail the entire generation, just skip this image
            character.image_path = None

    def _build_image_prompt(self, character: CharacterSpec, state: GameState) -> str:
        """
        Build a detailed image generation prompt for a character.

        Args:
            character: Character specification
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
            ", ".join(character.personality_traits)
            if character.personality_traits
            else "mysterious"
        )

        prompt = f"""Generate a photorealistic portrait of a {character.gender} character for a mystery party game.

CHARACTER DETAILS:
- Name: {character.name}
- Age: {character.age_range}
- Role: {character.role}
- Description: {character.public_description}
- Personality: {personality}

SETTING CONTEXT:
- Historical Period: {epoch}
- Location: {location}
- Country/Culture: {country}

COSTUME:
{character.costume_suggestion if character.costume_suggestion else f"Period-appropriate attire for {epoch} in {country}"}
"""

        # Add visual style consistency if available
        if state.visual_style:
            vs = state.visual_style

            prompt += f"""
VISUAL STYLE CONSISTENCY (CRITICAL - Apply to this character):
Style: {vs.style_description}
Art Direction: {vs.art_direction}

Color Palette: {", ".join(vs.color_palette) if vs.color_palette else "natural colors"}
Color Grading: {vs.color_grading}

Lighting: {vs.lighting_setup}
Mood: {vs.lighting_mood}

Background: {vs.background_aesthetic}
Focus: {vs.background_blur}

Technical: {vs.technical_specs}
Camera: {vs.camera_specs}

STRICT EXCLUSIONS (DO NOT INCLUDE):
{chr(10).join(f"- {item}" for item in vs.negative_prompts)}
"""
        else:
            # Fallback if no visual style (shouldn't happen, but safe)
            prompt += f"""
STYLE REQUIREMENTS:
- Photorealistic, professional portrait
- {epoch}-era fashion and styling appropriate for {country}
- Formal mystery party atmosphere
- High-quality, 8K resolution
- Lighting: Dramatic, film noir style
- Framing: Head and shoulders portrait
- Background: Subtle, period-appropriate
- Expression: {personality} demeanor
- FULL COLOR (never black and white)
- NO TEXT, labels, names, or captions
"""

        prompt += (
            "\n\nThe image should feel like a character from a high-quality period mystery film."
        )

        return prompt

    def _get_image_output_dir(self, state: GameState) -> Path:
        """
        Get the output directory for character images.

        Args:
            state: Current game state

        Returns:
            Path to images directory
        """
        # Use game_id from meta
        game_id = state.meta.id[:8] if state.meta else "default"
        return get_character_image_output_dir(game_id)

    def _mock_output(self, state: GameState) -> GameState:
        """
        Generate mock image paths for dry run mode.

        Args:
            state: Current game state

        Returns:
            State with mock image paths
        """
        output_dir = self._get_image_output_dir(state)

        for character in state.characters:
            mock_filename = f"{character.id}_{character.name.lower().replace(' ', '_')}.png"
            character.image_path = str((output_dir / mock_filename).absolute())
            logger.info(f"[A3.5] 🎭 Mock image: {character.name} -> {mock_filename}")

        return state
