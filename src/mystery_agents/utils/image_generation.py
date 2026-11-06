"""Shared utilities for character image generation using Gemini Image API."""

from __future__ import annotations

import asyncio
import base64
import functools
import logging
import os
from io import BytesIO
from pathlib import Path

from langchain_core.messages import HumanMessage
from langchain_google_genai import ChatGoogleGenerativeAI
from PIL import Image as PILImage

from mystery_agents.utils.constants import (
    GAME_DIR_PREFIX,
    IMAGE_GENERATION_MAX_RETRIES,
    IMAGE_GENERATION_MODEL,
    IMAGE_GENERATION_RETRY_DELAY_BASE,
    IMAGE_GENERATION_TEMPERATURE,
    IMAGE_GENERATION_TIMEOUT,
)

logger = logging.getLogger(__name__)


@functools.cache
def _get_image_model(api_key: str) -> ChatGoogleGenerativeAI:
    """Return a shared image generation client (one per API key)."""
    return ChatGoogleGenerativeAI(
        model=IMAGE_GENERATION_MODEL,
        temperature=IMAGE_GENERATION_TEMPERATURE,
        google_api_key=api_key,
        response_modalities=["IMAGE"],
        timeout=IMAGE_GENERATION_TIMEOUT,
    )


async def generate_image_with_gemini(
    prompt: str,
    output_path: Path,
    max_retries: int = IMAGE_GENERATION_MAX_RETRIES,
    retry_delay_base: float = IMAGE_GENERATION_RETRY_DELAY_BASE,
) -> bool:
    """
    Generate an image using Gemini Image API with retry logic.

    Args:
        prompt: Text prompt for image generation
        output_path: Path where to save the generated image
        max_retries: Maximum number of retry attempts
        retry_delay_base: Base delay for exponential backoff (in seconds)

    Returns:
        True if image was generated successfully, False otherwise

    Raises:
        ValueError: If GOOGLE_API_KEY is not set
    """
    for attempt in range(max_retries):
        try:
            await _call_gemini_image_api(prompt, output_path)
            return True
        except Exception as e:
            if attempt < max_retries - 1:
                delay = retry_delay_base * (2**attempt)
                logger.warning(
                    f"Image generation attempt {attempt + 1}/{max_retries} failed for "
                    f"{output_path.name}: {e}. Retrying in {delay:.0f}s"
                )
                await asyncio.sleep(delay)
            else:
                logger.error(
                    f"Image generation failed for {output_path.name} after "
                    f"{max_retries} attempts: {e}"
                )

    return False  # All retries exhausted


async def _call_gemini_image_api(prompt: str, output_path: Path) -> None:
    """
    Call the Gemini image generation model configured in IMAGE_GENERATION_MODEL.

    Uses LangChain's ChatGoogleGenerativeAI with response_modalities=["IMAGE"]
    to generate images from text prompts.

    Args:
        prompt: Image generation prompt
        output_path: Where to save the generated image

    Raises:
        ValueError: If API key is not set or response format is invalid
        Exception: If image generation fails
    """
    # Get API key
    api_key = os.getenv("GOOGLE_API_KEY")
    if not api_key:
        raise ValueError("GOOGLE_API_KEY environment variable not set")

    message = HumanMessage(content=[{"type": "text", "text": prompt}])
    resp = await _get_image_model(api_key).ainvoke([message])

    # Standard content blocks don't depend on the provider's raw layout, which may
    # put text or thought-signature parts before the image
    blocks = resp.content_blocks
    img_b64 = next(
        (
            block.get("base64")
            for block in blocks
            if block["type"] == "image" and block.get("base64")
        ),
        None,
    )
    if not img_b64:
        block_types = [block["type"] for block in blocks]
        raise ValueError(f"No image in model response (content block types: {block_types})")

    img_data = base64.b64decode(img_b64)

    # Open and save image
    image = PILImage.open(BytesIO(img_data))
    image.save(str(output_path), "PNG")


def get_character_image_output_dir(game_id: str, output_dir: str) -> Path:
    """
    Get the standard output directory for character images.

    Args:
        game_id: Game ID (first 8 characters)
        output_dir: Base output directory for generated games

    Returns:
        Path to the images/characters directory for this game
    """
    return Path(output_dir) / f"{GAME_DIR_PREFIX}{game_id}" / "images" / "characters"
