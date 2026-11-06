"""Tests for image generation utilities."""

import base64
from io import BytesIO
from pathlib import Path
from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from PIL import Image as PILImage

from mystery_agents.utils.image_generation import (
    _call_gemini_image_api,
    _get_image_model,
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


def _mock_image_model(content: Any) -> Any:
    """Patch the shared image model so ainvoke returns a response with this content."""
    mock_llm = MagicMock()
    mock_llm.ainvoke = AsyncMock(return_value=MagicMock(content=content))
    return patch("mystery_agents.utils.image_generation._get_image_model", return_value=mock_llm)


@pytest.mark.asyncio
async def test_call_gemini_image_api_success(
    tmp_path: Path, mock_google_api_key: None, sample_base64_image: str
) -> None:
    """Test successful API call to Gemini."""
    output_path = tmp_path / "test_image.png"

    with _mock_image_model([{"image_url": {"url": sample_base64_image}}]) as mock_get_model:
        await _call_gemini_image_api("Test prompt", output_path)

    assert output_path.exists()
    assert output_path.stat().st_size > 0
    mock_get_model.return_value.ainvoke.assert_awaited_once()


def test_get_image_model_is_shared(mock_google_api_key: None) -> None:
    """The image client is built once and configured for image output."""
    _get_image_model.cache_clear()
    with patch("mystery_agents.utils.image_generation.ChatGoogleGenerativeAI") as mock_cls:
        first = _get_image_model("key")
        second = _get_image_model("key")
    _get_image_model.cache_clear()

    assert first is second
    mock_cls.assert_called_once()
    # langchain-google-genai v4 takes response_modalities on the constructor
    assert mock_cls.call_args.kwargs["response_modalities"] == ["IMAGE"]


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
    with _mock_image_model(["not a dict"]):
        with pytest.raises(ValueError, match="Unexpected response format"):
            await _call_gemini_image_api("Test prompt", tmp_path / "test_image.png")


@pytest.mark.asyncio
async def test_call_gemini_image_api_no_image_url(
    tmp_path: Path, mock_google_api_key: None
) -> None:
    """Test that API call raises error when image_url is missing."""
    with _mock_image_model([{"not_image_url": "value"}]):
        with pytest.raises(ValueError, match="No image_url found"):
            await _call_gemini_image_api("Test prompt", tmp_path / "test_image.png")


@pytest.mark.asyncio
async def test_call_gemini_image_api_invalid_url_format(
    tmp_path: Path, mock_google_api_key: None
) -> None:
    """Test that API call raises error for invalid URL format."""
    with _mock_image_model([{"image_url": {"url": 12345}}]):
        with pytest.raises(ValueError, match="No valid URL string"):
            await _call_gemini_image_api("Test prompt", tmp_path / "test_image.png")


@pytest.mark.asyncio
async def test_call_gemini_image_api_missing_url(tmp_path: Path, mock_google_api_key: None) -> None:
    """Test that API call raises error when URL is missing."""
    with _mock_image_model([{"image_url": {}}]):
        with pytest.raises(ValueError, match="No image_url found"):
            await _call_gemini_image_api("Test prompt", tmp_path / "test_image.png")


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


@pytest.mark.asyncio
async def test_generate_image_with_gemini_logs_failures(
    tmp_path: Path, caplog: pytest.LogCaptureFixture
) -> None:
    """Failed attempts are logged with the underlying error instead of being swallowed."""
    output_path = tmp_path / "test.png"

    with patch(
        "mystery_agents.utils.image_generation._call_gemini_image_api",
        side_effect=RuntimeError("quota exceeded"),
    ):
        with patch("asyncio.sleep", new_callable=AsyncMock):
            result = await generate_image_with_gemini("prompt", output_path, max_retries=2)

    assert result is False
    messages = [r.getMessage() for r in caplog.records]
    assert any("attempt 1/2 failed" in m and "quota exceeded" in m for m in messages)
    assert any("after 2 attempts: quota exceeded" in m for m in messages)
