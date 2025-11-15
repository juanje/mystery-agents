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


@pytest.fixture(autouse=True)
def mock_google_api_key(monkeypatch: pytest.MonkeyPatch) -> None:
    """Mock GOOGLE_API_KEY for all tests."""
    monkeypatch.setenv("GOOGLE_API_KEY", "test-api-key-for-testing")


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
    assert agent.agent is not None


def test_base_agent_initialization_with_response_format(mock_llm: BaseChatModel) -> None:
    """Test BaseAgent initialization with response format."""
    agent = _TestAgent(llm=mock_llm, response_format=_TestOutputFormat)

    assert agent.llm == mock_llm
    assert agent.response_format == _TestOutputFormat


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


def test_invoke_with_structured_response(mock_llm: BaseChatModel, basic_state: GameState) -> None:
    """Test invoke with structured response format."""
    agent = _TestAgent(llm=mock_llm, response_format=_TestOutputFormat)

    expected_output = _TestOutputFormat(result="success")

    with patch("mystery_agents.agents.base.create_agent") as mock_create_agent:
        mock_agent = MagicMock()
        mock_agent.invoke.return_value = {"structured_response": expected_output}
        mock_create_agent.return_value = mock_agent

        agent.agent = mock_agent
        result = agent.invoke(basic_state)

        assert result == expected_output
        assert mock_agent.invoke.call_count == 1


def test_invoke_with_debug_mode(mock_llm: BaseChatModel) -> None:
    """Test invoke with debug mode enabled."""
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

    with patch("mystery_agents.agents.base.create_agent") as mock_create_agent:
        mock_agent = MagicMock()
        mock_agent.invoke.return_value = {"structured_response": expected_output}
        mock_create_agent.return_value = mock_agent

        result = agent.invoke(state)

        assert result == expected_output
        # Should create agent with debug middleware
        assert mock_create_agent.call_count == 1


def test_invoke_without_structured_response_raises_error(
    mock_llm: BaseChatModel, basic_state: GameState
) -> None:
    """Test invoke raises error when structured response is missing."""
    agent = _TestAgent(llm=mock_llm, response_format=_TestOutputFormat)

    with patch("mystery_agents.agents.base.create_agent") as mock_create_agent:
        mock_agent = MagicMock()
        # Return result without structured_response
        mock_agent.invoke.return_value = {"messages": [AIMessage(content="text response")]}
        mock_create_agent.return_value = mock_agent

        agent.agent = mock_agent

        with pytest.raises(ValueError, match="No structured_response in agent result"):
            agent.invoke(basic_state)


def test_invoke_without_response_format_returns_content(
    mock_llm: BaseChatModel, basic_state: GameState
) -> None:
    """Test invoke without response format returns message content."""
    agent = _TestAgent(llm=mock_llm)  # No response_format

    expected_content = "Test response content"

    with patch("mystery_agents.agents.base.create_agent") as mock_create_agent:
        mock_agent = MagicMock()
        mock_agent.invoke.return_value = {
            "messages": [AIMessage(content=expected_content)],
        }
        mock_create_agent.return_value = mock_agent

        agent.agent = mock_agent
        result = agent.invoke(basic_state)

        assert result == expected_content


def test_invoke_with_custom_user_message(mock_llm: BaseChatModel, basic_state: GameState) -> None:
    """Test invoke with custom user message."""
    agent = _TestAgent(llm=mock_llm)
    custom_message = "Custom test message"

    with patch("mystery_agents.agents.base.create_agent") as mock_create_agent:
        mock_agent = MagicMock()
        mock_agent.invoke.return_value = {"messages": [AIMessage(content="response")]}
        mock_create_agent.return_value = mock_agent

        agent.agent = mock_agent
        agent.invoke(basic_state, user_message=custom_message)

        # Verify the message was passed (check call args)
        call_args = mock_agent.invoke.call_args
        messages = call_args[0][0]["messages"]
        assert len(messages) == 2  # SystemMessage and HumanMessage
        assert messages[1].content == custom_message
