"""Unit tests for BaseAgent class."""

from unittest.mock import MagicMock, patch

import pytest
from langchain_core.language_models import BaseChatModel
from langchain_core.messages import AIMessage
from pydantic import BaseModel, Field

from mystery_agents.agents.base import BaseAgent
from mystery_agents.models.state import GameConfig, GameState, MetaInfo, PlayerConfig
from mystery_agents.utils.constants import TEST_DEFAULT_DURATION, TEST_DEFAULT_PLAYERS


class _TestOutputFormat(BaseModel):
    """Test output format for structured responses."""

    result: str = Field(description="Test result")


class _TestAgent(BaseAgent):
    """Test agent implementation."""

    def get_system_prompt(self, state: GameState) -> str:
        """Get test system prompt."""
        return "Test system prompt"

    def _mock_output(self, state: GameState) -> GameState:
        """Generate mock output."""
        return state


@pytest.fixture
def basic_state() -> GameState:
    """Create a basic game state for testing."""
    return GameState(
        meta=MetaInfo(),
        config=GameConfig(
            players=PlayerConfig(total=TEST_DEFAULT_PLAYERS),
            host_gender="male",
            duration_minutes=TEST_DEFAULT_DURATION,
            dry_run=True,
        ),
    )


@pytest.fixture
def mock_llm() -> BaseChatModel:
    """Create a mock LLM."""
    mock = MagicMock(spec=BaseChatModel)
    return mock


def test_base_agent_initialization(mock_llm: BaseChatModel) -> None:
    """Test BaseAgent initialization."""
    agent = _TestAgent(llm=mock_llm)

    assert agent.llm == mock_llm
    assert agent.response_format is None
    # Without a response format the raw model is invoked directly
    assert agent.runnable is mock_llm


def test_base_agent_initialization_with_response_format(mock_llm: BaseChatModel) -> None:
    """Test BaseAgent initialization with response format."""
    agent = _TestAgent(llm=mock_llm, response_format=_TestOutputFormat)

    assert agent.llm == mock_llm
    assert agent.response_format == _TestOutputFormat
    mock_llm.with_structured_output.assert_called_once_with(  # type: ignore[attr-defined]
        _TestOutputFormat, method="json_schema", include_raw=True
    )


def test_should_use_mock_returns_true_for_dry_run(
    mock_llm: BaseChatModel, basic_state: GameState
) -> None:
    """Test _should_use_mock returns True when dry_run is enabled."""
    agent = _TestAgent(llm=mock_llm)

    result = agent._should_use_mock(basic_state)
    assert result is True


def test_should_use_mock_returns_false_when_not_dry_run(mock_llm: BaseChatModel) -> None:
    """Test _should_use_mock returns False when dry_run is disabled."""
    state = GameState(
        meta=MetaInfo(),
        config=GameConfig(
            players=PlayerConfig(total=TEST_DEFAULT_PLAYERS),
            host_gender="male",
            duration_minutes=TEST_DEFAULT_DURATION,
            dry_run=False,
        ),
    )

    agent = _TestAgent(llm=mock_llm)
    result = agent._should_use_mock(state)
    assert result is False


def test_mock_output_returns_state_unchanged(
    mock_llm: BaseChatModel, basic_state: GameState
) -> None:
    """Test _mock_output returns state unchanged by default."""
    agent = _TestAgent(llm=mock_llm)

    result = agent._mock_output(basic_state)
    assert result == basic_state


def _structured_result(
    parsed: BaseModel | None, error: Exception | None = None
) -> dict[str, object]:
    """Build the dict returned by with_structured_output(include_raw=True)."""
    return {"raw": AIMessage(content="raw"), "parsed": parsed, "parsing_error": error}


def test_invoke_with_structured_response(mock_llm: BaseChatModel, basic_state: GameState) -> None:
    """Test invoke with structured response format."""
    agent = _TestAgent(llm=mock_llm, response_format=_TestOutputFormat)
    expected_output = _TestOutputFormat(result="success")
    agent.runnable = MagicMock()
    agent.runnable.invoke.return_value = _structured_result(expected_output)

    result = agent.invoke(basic_state)

    assert result == expected_output
    assert agent.runnable.invoke.call_count == 1


def test_invoke_with_debug_mode_logs_response(mock_llm: BaseChatModel) -> None:
    """Test invoke logs the raw response when debug mode is enabled."""
    state = GameState(
        meta=MetaInfo(),
        config=GameConfig(
            players=PlayerConfig(total=TEST_DEFAULT_PLAYERS),
            host_gender="male",
            duration_minutes=TEST_DEFAULT_DURATION,
            dry_run=False,
            debug_model=True,
        ),
    )
    agent = _TestAgent(llm=mock_llm, response_format=_TestOutputFormat)
    expected_output = _TestOutputFormat(result="debug_success")
    agent.runnable = MagicMock()
    agent.runnable.invoke.return_value = _structured_result(expected_output)

    with patch("mystery_agents.agents.base.log_model_response") as mock_log:
        result = agent.invoke(state)

    assert result == expected_output
    mock_log.assert_called_once()


def test_invoke_without_parsed_output_raises_error(
    mock_llm: BaseChatModel, basic_state: GameState
) -> None:
    """The error says why the response ended, so a cut-off answer is recognisable."""
    agent = _TestAgent(llm=mock_llm, response_format=_TestOutputFormat)
    agent.runnable = MagicMock()
    result = _structured_result(None, ValueError("bad json"))
    result["raw"] = AIMessage(
        content="{...",
        response_metadata={"finish_reason": "MAX_TOKENS"},
        usage_metadata={"input_tokens": 10, "output_tokens": 32768, "total_tokens": 32778},
    )
    agent.runnable.invoke.return_value = result

    with pytest.raises(
        ValueError,
        match=r"_TestOutputFormat \(finish reason MAX_TOKENS, 32768 output tokens\): bad json",
    ):
        agent.invoke(basic_state)


def test_invoke_without_response_format_returns_content(
    mock_llm: BaseChatModel, basic_state: GameState
) -> None:
    """Test invoke without response format returns message content."""
    agent = _TestAgent(llm=mock_llm)  # No response_format
    expected_content = "Test response content"
    agent.runnable = MagicMock()
    agent.runnable.invoke.return_value = AIMessage(content=expected_content)

    result = agent.invoke(basic_state)

    assert result == expected_content


def test_invoke_with_custom_user_message(mock_llm: BaseChatModel, basic_state: GameState) -> None:
    """Test invoke with custom user message."""
    agent = _TestAgent(llm=mock_llm)
    custom_message = "Custom test message"
    agent.runnable = MagicMock()
    agent.runnable.invoke.return_value = AIMessage(content="response")

    agent.invoke(basic_state, user_message=custom_message)

    messages = agent.runnable.invoke.call_args[0][0]
    assert len(messages) == 2  # SystemMessage and HumanMessage
    assert messages[1].content == custom_message
