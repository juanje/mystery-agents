"""Shared utilities for character image generation using Gemini Image API."""

from __future__ import annotations

import asyncio
import base64
import os
from io import BytesIO
from pathlib import Path

from langchain_core.messages import HumanMessage
from langchain_google_genai import ChatGoogleGenerativeAI
from PIL import Image as PILImage

from mystery_agents.utils.constants import (
    IMAGE_GENERATION_MAX_RETRIES,
    IMAGE_GENERATION_MODEL,
    IMAGE_GENERATION_RETRY_DELAY_BASE,
    IMAGE_GENERATION_TEMPERATURE,
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
        except Exception:
            if attempt < max_retries - 1:
                delay = retry_delay_base * (2**attempt)
                await asyncio.sleep(delay)
            # Continue to next iteration or return False after last attempt

    return False  # All retries exhausted


async def _call_gemini_image_api(prompt: str, output_path: Path) -> None:
    """
    Call Gemini Image Generation API using Gemini 2.5 Flash Image model.

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

    llm = ChatGoogleGenerativeAI(
        model=IMAGE_GENERATION_MODEL,
        temperature=IMAGE_GENERATION_TEMPERATURE,
        google_api_key=api_key,
        response_modalities=["IMAGE"],
    )

    message = HumanMessage(content=[{"type": "text", "text": prompt}])

    loop = asyncio.get_event_loop()
    resp = await loop.run_in_executor(
        None,
        lambda: llm.invoke([message]),
    )

    # Extract base64 image data from response
    # resp.content is a list, first element should be a dict with image_url
    content_item = resp.content[0]
    if not isinstance(content_item, dict):
        raise ValueError(
            f"Unexpected response format: content[0] is {type(content_item)}, expected dict"
        )

    image_url_data = content_item.get("image_url")
    if not image_url_data or not isinstance(image_url_data, dict):
        raise ValueError(f"No image_url found in response: {content_item}")

    url_str = image_url_data.get("url")
    if not url_str or not isinstance(url_str, str):
        raise ValueError(f"No valid URL string in image_url: {image_url_data}")

    # Extract base64 data (format: data:image/png;base64,<data>)
    img_b64 = url_str.split(",")[-1]
    img_data = base64.b64decode(img_b64)

    # Open and save image
    image = PILImage.open(BytesIO(img_data))
    image.save(str(output_path), "PNG")


def get_character_image_output_dir(game_id: str) -> Path:
    """
    Get the standard output directory for character images.

    Args:
        game_id: Game ID (first 8 characters)

    Returns:
        Path to the images/characters directory for this game
    """
    return Path("output") / f"game_{game_id}" / "images" / "characters"
