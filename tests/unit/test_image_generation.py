"""Tests for image generation utilities."""

import base64
from collections.abc import Callable
from io import BytesIO
from pathlib import Path
from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from PIL import Image as PILImage

from mystery_agents.utils.image_generation import (
    _call_gemini_image_api,
    generate_image_with_gemini,
    get_character_image_output_dir,
)


@pytest.fixture
def sample_image_data() -> bytes:
    """Create sample PNG image data."""
    # Create a small test image
    img = PILImage.new("RGB", (10, 10), color="red")
    buffer = BytesIO()
    img.save(buffer, format="PNG")
    return buffer.getvalue()


@pytest.fixture
def sample_base64_image(sample_image_data: bytes) -> str:
    """Create base64 encoded image data URL."""
    b64_data = base64.b64encode(sample_image_data).decode("utf-8")
    return f"data:image/png;base64,{b64_data}"


@pytest.mark.asyncio
async def test_generate_image_with_gemini_success(
    tmp_path: Path, mock_google_api_key: None, sample_base64_image: str
) -> None:
    """Test successful image generation."""
    output_path = tmp_path / "test_image.png"

    with patch("mystery_agents.utils.image_generation._call_gemini_image_api") as mock_call:
        mock_call.return_value = None  # Success, no exception

        result = await generate_image_with_gemini("Test prompt", output_path)

        assert result is True
        assert mock_call.called
        assert mock_call.call_args[0][0] == "Test prompt"
        assert mock_call.call_args[0][1] == output_path


@pytest.mark.asyncio
async def test_generate_image_with_gemini_retry_on_failure(
    tmp_path: Path, mock_google_api_key: None
) -> None:
    """Test that image generation retries on failure."""
    output_path = tmp_path / "test_image.png"

    call_count = 0

    async def mock_call_with_retries(*args: object, **kwargs: object) -> None:
        nonlocal call_count
        call_count += 1
        if call_count < 3:
            raise Exception("Temporary failure")
        # Success on third attempt

    with patch(
        "mystery_agents.utils.image_generation._call_gemini_image_api",
        side_effect=mock_call_with_retries,
    ):
        with patch("asyncio.sleep", new_callable=AsyncMock):  # Mock sleep to speed up test
            result = await generate_image_with_gemini(
                "Test prompt", output_path, max_retries=5, retry_delay_base=0.01
            )

            assert result is True
            assert call_count == 3


@pytest.mark.asyncio
async def test_generate_image_with_gemini_max_retries_exceeded(
    tmp_path: Path, mock_google_api_key: None
) -> None:
    """Test that image generation returns False after max retries."""
    output_path = tmp_path / "test_image.png"

    with patch(
        "mystery_agents.utils.image_generation._call_gemini_image_api",
        side_effect=Exception("Persistent failure"),
    ):
        with patch("asyncio.sleep", new_callable=AsyncMock):  # Mock sleep to speed up test
            result = await generate_image_with_gemini(
                "Test prompt", output_path, max_retries=3, retry_delay_base=0.01
            )

            assert result is False


@pytest.mark.asyncio
async def test_generate_image_with_gemini_exponential_backoff(
    tmp_path: Path, mock_google_api_key: None
) -> None:
    """Test that retry delays use exponential backoff."""
    output_path = tmp_path / "test_image.png"

    delays: list[float] = []

    async def mock_sleep(delay: float) -> None:
        delays.append(delay)

    with patch(
        "mystery_agents.utils.image_generation._call_gemini_image_api",
        side_effect=Exception("Failure"),
    ):
        with patch("asyncio.sleep", side_effect=mock_sleep):
            await generate_image_with_gemini(
                "Test prompt", output_path, max_retries=4, retry_delay_base=0.1
            )

            # Should have delays: 0.1, 0.2, 0.4 (exponential backoff)
            assert len(delays) == 3
            assert delays[0] == 0.1
            assert delays[1] == 0.2
            assert delays[2] == 0.4


@pytest.mark.asyncio
async def test_call_gemini_image_api_success(
    tmp_path: Path, mock_google_api_key: None, sample_base64_image: str
) -> None:
    """Test successful API call to Gemini."""
    output_path = tmp_path / "test_image.png"

    # Create mock response
    mock_response = MagicMock()
    mock_response.content = [
        {
            "image_url": {
                "url": sample_base64_image,
            },
        },
    ]

    mock_llm = MagicMock()
    mock_llm.invoke.return_value = mock_response

    with patch(
        "mystery_agents.utils.image_generation.ChatGoogleGenerativeAI",
        return_value=mock_llm,
    ):
        # Mock asyncio.get_event_loop().run_in_executor to execute the function directly
        with patch("asyncio.get_event_loop") as mock_loop:
            loop_instance = MagicMock()

            def mock_run_in_executor(executor: Any, func: Callable[[], Any], *args: Any) -> Any:
                # Execute the function immediately (synchronously for testing)
                return func()

            loop_instance.run_in_executor = AsyncMock(side_effect=mock_run_in_executor)
            mock_loop.return_value = loop_instance

            await _call_gemini_image_api("Test prompt", output_path)

            # Image should be saved
            assert output_path.exists()
            assert output_path.stat().st_size > 0

            # Verify LLM was called correctly
            assert mock_llm.invoke.called
            call_args = mock_llm.invoke.call_args
            assert "generation_config" in call_args[1]
            assert call_args[1]["generation_config"]["response_modalities"] == ["IMAGE"]


@pytest.mark.asyncio
async def test_call_gemini_image_api_no_api_key(monkeypatch: pytest.MonkeyPatch) -> None:
    """Test that API call raises error when API key is missing."""
    monkeypatch.delenv("GOOGLE_API_KEY", raising=False)
    with pytest.raises(ValueError, match="GOOGLE_API_KEY"):
        await _call_gemini_image_api("Test prompt", Path("/tmp/test.png"))


@pytest.mark.asyncio
async def test_call_gemini_image_api_invalid_response_format(
    tmp_path: Path, mock_google_api_key: None
) -> None:
    """Test that API call raises error for invalid response format."""
    output_path = tmp_path / "test_image.png"

    # Create mock response with invalid format (not a dict)
    mock_response = MagicMock()
    mock_response.content = ["not a dict"]

    mock_llm = MagicMock()
    mock_llm.invoke.return_value = mock_response

    with patch(
        "mystery_agents.utils.image_generation.ChatGoogleGenerativeAI",
        return_value=mock_llm,
    ):
        with patch("asyncio.get_event_loop") as mock_loop:
            loop_instance = MagicMock()

            def mock_run_in_executor(executor: Any, func: Callable[[], Any], *args: Any) -> Any:
                return func()

            loop_instance.run_in_executor = AsyncMock(side_effect=mock_run_in_executor)
            mock_loop.return_value = loop_instance

            with pytest.raises(ValueError, match="Unexpected response format"):
                await _call_gemini_image_api("Test prompt", output_path)


@pytest.mark.asyncio
async def test_call_gemini_image_api_no_image_url(
    tmp_path: Path, mock_google_api_key: None
) -> None:
    """Test that API call raises error when image_url is missing."""
    output_path = tmp_path / "test_image.png"

    # Create mock response without image_url
    mock_response = MagicMock()
    mock_response.content = [{"not_image_url": "value"}]

    mock_llm = MagicMock()
    mock_llm.invoke.return_value = mock_response

    with patch(
        "mystery_agents.utils.image_generation.ChatGoogleGenerativeAI",
        return_value=mock_llm,
    ):
        with patch("asyncio.get_event_loop") as mock_loop:
            loop_instance = MagicMock()

            def mock_run_in_executor(executor: Any, func: Callable[[], Any], *args: Any) -> Any:
                return func()

            loop_instance.run_in_executor = AsyncMock(side_effect=mock_run_in_executor)
            mock_loop.return_value = loop_instance

            with pytest.raises(ValueError, match="No image_url found"):
                await _call_gemini_image_api("Test prompt", output_path)


@pytest.mark.asyncio
async def test_call_gemini_image_api_invalid_url_format(
    tmp_path: Path, mock_google_api_key: None
) -> None:
    """Test that API call raises error for invalid URL format."""
    output_path = tmp_path / "test_image.png"

    # Create mock response with invalid URL (not a string)
    mock_response = MagicMock()
    mock_response.content = [{"image_url": {"url": 12345}}]  # Not a string

    mock_llm = MagicMock()
    mock_llm.invoke.return_value = mock_response

    with patch(
        "mystery_agents.utils.image_generation.ChatGoogleGenerativeAI",
        return_value=mock_llm,
    ):
        with patch("asyncio.get_event_loop") as mock_loop:
            loop_instance = MagicMock()

            def mock_run_in_executor(executor: Any, func: Callable[[], Any], *args: Any) -> Any:
                return func()

            loop_instance.run_in_executor = AsyncMock(side_effect=mock_run_in_executor)
            mock_loop.return_value = loop_instance

            with pytest.raises(ValueError, match="No valid URL string"):
                await _call_gemini_image_api("Test prompt", output_path)


@pytest.mark.asyncio
async def test_call_gemini_image_api_missing_url(tmp_path: Path, mock_google_api_key: None) -> None:
    """Test that API call raises error when URL is missing."""
    output_path = tmp_path / "test_image.png"

    # Create mock response with image_url but no url key
    mock_response = MagicMock()
    mock_response.content = [{"image_url": {}}]  # Empty dict, no url

    mock_llm = MagicMock()
    mock_llm.invoke.return_value = mock_response

    with patch(
        "mystery_agents.utils.image_generation.ChatGoogleGenerativeAI",
        return_value=mock_llm,
    ):
        with patch("asyncio.get_event_loop") as mock_loop:
            loop_instance = MagicMock()

            def mock_run_in_executor(executor: Any, func: Callable[[], Any], *args: Any) -> Any:
                return func()

            loop_instance.run_in_executor = AsyncMock(side_effect=mock_run_in_executor)
            mock_loop.return_value = loop_instance

            with pytest.raises(ValueError, match="No image_url found"):
                await _call_gemini_image_api("Test prompt", output_path)


def test_get_character_image_output_dir() -> None:
    """Test that output directory path is constructed correctly."""
    game_id = "abc12345"
    result = get_character_image_output_dir(game_id)

    assert isinstance(result, Path)
    assert "output" in str(result)
    assert f"game_{game_id}" in str(result)
    assert "images" in str(result)
    assert "characters" in str(result)
    assert result == Path("output") / f"game_{game_id}" / "images" / "characters"


def test_get_character_image_output_dir_short_id() -> None:
    """Test that output directory works with short game IDs."""
    game_id = "short"
    result = get_character_image_output_dir(game_id)

    assert f"game_{game_id}" in str(result)


def test_get_character_image_output_dir_long_id() -> None:
    """Test that output directory works with long game IDs."""
    game_id = "very_long_game_id_123456789"
    result = get_character_image_output_dir(game_id)

    assert f"game_{game_id}" in str(result)
