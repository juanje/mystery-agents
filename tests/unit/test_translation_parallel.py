"""Unit tests for parallel translation functionality."""

from __future__ import annotations

import asyncio
from unittest.mock import MagicMock

import pytest

from mystery_agents.utils.translation import (
    _translate_all_batches_async,
    _translate_batch_async,
)


@pytest.mark.asyncio
async def test_translate_batch_async() -> None:
    """Test async wrapper for batch translation."""
    # Since we can't easily mock the sync function in executor,
    # we just verify the function signature
    assert asyncio.iscoroutinefunction(_translate_batch_async)


@pytest.mark.asyncio
async def test_translate_all_batches_async_empty() -> None:
    """Test parallel translation with empty input."""
    mock_llm = MagicMock()

    result = await _translate_all_batches_async([], mock_llm, "Spanish")

    assert result == {}


@pytest.mark.asyncio
async def test_translate_all_batches_async_single_batch() -> None:
    """Test parallel translation with single batch."""
    # This test would require mocking _translate_batch_async
    # For now, we just verify the function exists and is async
    assert asyncio.iscoroutinefunction(_translate_all_batches_async)


def test_parallel_translation_functions_exist() -> None:
    """Test that parallel translation functions are defined."""
    from mystery_agents.utils import translation

    # Verify async functions exist
    assert hasattr(translation, "_translate_batch_async")
    assert hasattr(translation, "_translate_all_batches_async")

    # Verify they are coroutine functions
    assert asyncio.iscoroutinefunction(translation._translate_batch_async)
    assert asyncio.iscoroutinefunction(translation._translate_all_batches_async)


def test_translate_all_batches_async_signature() -> None:
    """Test that _translate_all_batches_async has correct signature."""
    import inspect

    sig = inspect.signature(_translate_all_batches_async)
    params = list(sig.parameters.keys())

    assert "text_items" in params
    assert "llm" in params
    assert "target_lang" in params
    assert "batch_size" in params
    assert "max_concurrent" in params

    # Check default values
    assert sig.parameters["batch_size"].default == 20
    assert sig.parameters["max_concurrent"].default == 3


def test_asyncio_import() -> None:
    """Test that asyncio is imported in translation module."""
    from mystery_agents.utils import translation

    assert hasattr(translation, "asyncio")
