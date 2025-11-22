"""Tests for debug middleware."""

from typing import Any, cast
from unittest.mock import MagicMock

import pytest
from langchain.agents.middleware import AgentState
from langchain_core.messages import AIMessage, HumanMessage

from mystery_agents.utils.debug_middleware import _log_model_response_impl as log_model_response


@pytest.fixture
def mock_runtime() -> MagicMock:
    """Create a mock runtime."""
    return MagicMock()


def test_log_model_response_with_ai_message(mock_runtime: MagicMock) -> None:
    """Test that middleware logs AI message content."""
    state = cast(
        AgentState,
        {
            "messages": [
                HumanMessage(content="User question"),
                AIMessage(content="This is a test response from the model"),
            ],
        },
    )

    result = log_model_response(state, mock_runtime)

    # Should return None (middleware doesn't modify state)
    assert result is None


def test_log_model_response_with_empty_content(mock_runtime: MagicMock) -> None:
    """Test that middleware handles empty message content."""
    state = cast(
        AgentState,
        {
            "messages": [
                HumanMessage(content="User question"),
                AIMessage(content=""),
            ],
        },
    )

    result = log_model_response(state, mock_runtime)

    assert result is None


def test_log_model_response_with_long_content(mock_runtime: MagicMock) -> None:
    """Test that middleware truncates long content."""
    long_content = "A" * 2000  # Longer than 1000 chars
    state = cast(
        AgentState,
        {
            "messages": [
                HumanMessage(content="User question"),
                AIMessage(content=long_content),
            ],
        },
    )

    result = log_model_response(state, mock_runtime)

    assert result is None


def test_log_model_response_with_json_content(mock_runtime: MagicMock) -> None:
    """Test that middleware parses JSON content."""
    json_content = '{"result": "success", "data": {"key": "value"}}'
    state = cast(
        AgentState,
        {
            "messages": [
                HumanMessage(content="User question"),
                AIMessage(content=json_content),
            ],
        },
    )

    result = log_model_response(state, mock_runtime)

    assert result is None


def test_log_model_response_with_invalid_json(mock_runtime: MagicMock) -> None:
    """Test that middleware handles invalid JSON gracefully."""
    invalid_json = '{"result": "success", "data": invalid}'
    state = cast(
        AgentState,
        {
            "messages": [
                HumanMessage(content="User question"),
                AIMessage(content=invalid_json),
            ],
        },
    )

    result = log_model_response(state, mock_runtime)

    assert result is None


def test_log_model_response_with_structured_response(mock_runtime: MagicMock) -> None:
    """Test that middleware logs structured response."""
    from pydantic import BaseModel, Field

    class TestModel(BaseModel):
        """Test model."""

        result: str = Field(description="Test result")

    structured = TestModel(result="success")

    state = cast(
        AgentState,
        {
            "messages": [AIMessage(content="Response")],
            "structured_response": structured,
        },
    )

    result = log_model_response(state, mock_runtime)

    assert result is None


def test_log_model_response_without_structured_response(mock_runtime: MagicMock) -> None:
    """Test that middleware handles missing structured response."""
    state = cast(
        AgentState,
        {
            "messages": [AIMessage(content="Response")],
        },
    )

    result = log_model_response(state, mock_runtime)

    assert result is None


def test_log_model_response_with_non_dict_state(mock_runtime: MagicMock) -> None:
    """Test that middleware handles non-dict state."""
    state: Any = "not a dict"

    result = log_model_response(state, mock_runtime)

    assert result is None


def test_log_model_response_with_empty_messages(mock_runtime: MagicMock) -> None:
    """Test that middleware handles empty messages list."""
    state = cast(
        AgentState,
        {
            "messages": [],
        },
    )

    result = log_model_response(state, mock_runtime)

    assert result is None


def test_log_model_response_with_non_ai_message(mock_runtime: MagicMock) -> None:
    """Test that middleware handles non-AI messages."""
    state = cast(
        AgentState,
        {
            "messages": [
                HumanMessage(content="User question"),
                HumanMessage(content="Another user message"),
            ],
        },
    )

    result = log_model_response(state, mock_runtime)

    assert result is None


def test_log_model_response_with_structured_response_no_model_dump(mock_runtime: MagicMock) -> None:
    """Test that middleware handles structured response without model_dump."""

    # Create an object that doesn't have model_dump
    class SimpleObject:
        """Simple object without model_dump."""

        def __init__(self) -> None:
            self.data = "test"

    structured = SimpleObject()

    state = cast(
        AgentState,
        {
            "messages": [AIMessage(content="Response")],
            "structured_response": structured,
        },
    )

    result = log_model_response(state, mock_runtime)

    assert result is None


def test_log_model_response_with_structured_response_error(mock_runtime: MagicMock) -> None:
    """Test that middleware handles errors when accessing structured response."""

    # Create an object that raises an error when accessing model_dump
    class ErrorObject:
        """Object that raises error."""

        @property
        def model_dump(self) -> Any:
            raise RuntimeError("Error accessing model_dump")

    structured = ErrorObject()

    state = cast(
        AgentState,
        {
            "messages": [AIMessage(content="Response")],
            "structured_response": structured,
        },
    )

    # Should not raise, but handle gracefully
    result = log_model_response(state, mock_runtime)

    assert result is None


def test_log_model_response_with_dict_state_keys(mock_runtime: MagicMock) -> None:
    """Test that middleware logs state keys."""
    state = cast(
        AgentState,
        {
            "messages": [AIMessage(content="Response")],
            "key1": "value1",
            "key2": "value2",
        },
    )

    result = log_model_response(state, mock_runtime)

    assert result is None
