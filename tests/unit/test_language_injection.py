"""Tests for language injection in BaseAgent."""

from typing import Any
from unittest.mock import MagicMock

import pytest

from mystery_agents.agents.base import BaseAgent
from mystery_agents.models.state import GameConfig, GameState, MetaInfo
from mystery_agents.utils.constants import LANG_CODE_ENGLISH, TEST_DEFAULT_DURATION


class MockAgent(BaseAgent):
    """Mock agent for testing language injection."""

    def get_system_prompt(self, state: GameState) -> str:
        """Get test system prompt."""
        return "Test system prompt"


@pytest.fixture
def mock_llm() -> MagicMock:
    """Create a mock LLM."""
    llm = MagicMock()
    return llm


@pytest.fixture
def english_state() -> GameState:
    """Create state with English language."""
    return GameState(
        meta=MetaInfo(),
        config=GameConfig(
            dry_run=True,
            language=LANG_CODE_ENGLISH,  # type: ignore[arg-type]
            duration_minutes=TEST_DEFAULT_DURATION,
        ),
    )


@pytest.fixture
def spanish_state() -> GameState:
    """Create state with Spanish language."""
    return GameState(
        meta=MetaInfo(),
        config=GameConfig(
            dry_run=True,
            language="es",
            duration_minutes=TEST_DEFAULT_DURATION,
        ),
    )


class TestLanguageInjection:
    """Test language injection functionality."""

    def test_english_returns_empty_injection(
        self, mock_llm: MagicMock, english_state: GameState
    ) -> None:
        """Test that English state returns no language injection."""
        agent = MockAgent(mock_llm)
        injection = agent._get_language_injection(english_state)
        assert injection == ""

    def test_spanish_returns_injection_with_critical_requirements(
        self, mock_llm: MagicMock, spanish_state: GameState
    ) -> None:
        """Test that Spanish state returns language injection with critical requirements."""
        agent = MockAgent(mock_llm)
        injection = agent._get_language_injection(spanish_state)

        assert "CRITICAL LANGUAGE REQUIREMENTS" in injection
        assert "Spanish" in injection  # Spanish language name
        assert "ALL creative and narrative content" in injection

    def test_injection_includes_json_integrity_warning(
        self, mock_llm: MagicMock, spanish_state: GameState
    ) -> None:
        """Test that injection includes JSON structure integrity warning."""
        agent = MockAgent(mock_llm)
        injection = agent._get_language_injection(spanish_state)

        assert "JSON Structure Integrity" in injection
        assert "Keep ALL JSON keys in English" in injection
        assert "NEVER translate field names" in injection
        assert "CORRECT" in injection
        assert "WRONG" in injection

    def test_injection_includes_cultural_adaptation(
        self, mock_llm: MagicMock, spanish_state: GameState
    ) -> None:
        """Test that injection includes cultural adaptation instructions."""
        agent = MockAgent(mock_llm)
        injection = agent._get_language_injection(spanish_state)

        assert "Cultural Adaptation" in injection
        assert "culturally appropriate expressions" in injection
        assert "idioms" in injection

    def test_injection_includes_context_consistency(
        self, mock_llm: MagicMock, spanish_state: GameState
    ) -> None:
        """Test that injection includes context consistency instructions."""
        agent = MockAgent(mock_llm)
        injection = agent._get_language_injection(spanish_state)

        assert "Context Consistency" in injection
        assert "Input context may already be" in injection
        assert "Maintain narrative consistency" in injection

    def test_injection_overrides_instructions(
        self, mock_llm: MagicMock, spanish_state: GameState
    ) -> None:
        """Test that injection states it overrides previous instructions."""
        agent = MockAgent(mock_llm)
        injection = agent._get_language_injection(spanish_state)

        assert "override any language assumptions" in injection
        assert "HIGHEST PRIORITY" in injection


class TestInvokeWithLanguageInjection:
    """Test that invoke method properly injects language instructions."""

    def test_invoke_appends_injection_to_system_prompt_for_spanish(
        self, mock_llm: MagicMock, spanish_state: GameState
    ) -> None:
        """Test that invoke appends language injection to system prompt for Spanish."""
        # Mock the agent invoke to capture the messages
        captured_messages: list[Any] = []

        def mock_invoke(inputs: Any) -> dict[str, Any]:
            captured_messages.extend(inputs["messages"])
            return {"messages": [MagicMock(content="test response")]}

        agent = MockAgent(mock_llm)
        agent.agent.invoke = mock_invoke  # type: ignore[assignment]

        agent.invoke(spanish_state)

        # Verify we captured messages
        assert len(captured_messages) == 2  # SystemMessage + HumanMessage

        # Check system message content
        system_message = captured_messages[0]
        assert "Test system prompt" in system_message.content
        assert "CRITICAL LANGUAGE REQUIREMENTS" in system_message.content

    def test_invoke_does_not_inject_for_english(
        self, mock_llm: MagicMock, english_state: GameState
    ) -> None:
        """Test that invoke does not inject language instructions for English."""
        # Mock the agent invoke to capture the messages
        captured_messages: list[Any] = []

        def mock_invoke(inputs: Any) -> dict[str, Any]:
            captured_messages.extend(inputs["messages"])
            return {"messages": [MagicMock(content="test response")]}

        agent = MockAgent(mock_llm)
        agent.agent.invoke = mock_invoke  # type: ignore[assignment]

        agent.invoke(english_state)

        # Verify we captured messages
        assert len(captured_messages) == 2

        # Check system message content - should only have original prompt
        system_message = captured_messages[0]
        assert "Test system prompt" in system_message.content
        assert "CRITICAL LANGUAGE REQUIREMENTS" not in system_message.content
        assert system_message.content == "Test system prompt"  # Exact match

    def test_injection_is_appended_not_prepended(
        self, mock_llm: MagicMock, spanish_state: GameState
    ) -> None:
        """Test that language injection is appended (not prepended) to system prompt."""
        captured_messages: list[Any] = []

        def mock_invoke(inputs: Any) -> dict[str, Any]:
            captured_messages.extend(inputs["messages"])
            return {"messages": [MagicMock(content="test response")]}

        agent = MockAgent(mock_llm)
        agent.agent.invoke = mock_invoke  # type: ignore[assignment]

        agent.invoke(spanish_state)

        system_message = captured_messages[0]

        # The original prompt should come first
        prompt_start = system_message.content.find("Test system prompt")
        injection_start = system_message.content.find("CRITICAL LANGUAGE REQUIREMENTS")

        assert prompt_start < injection_start, "Language injection should be appended"


class TestLanguageInjectionWithDifferentLanguages:
    """Test language injection for different target languages."""

    def test_language_name_is_correctly_displayed(self, mock_llm: MagicMock) -> None:
        """Test that the target language name is correctly displayed."""
        state = GameState(
            meta=MetaInfo(),
            config=GameConfig(
                dry_run=True,
                language="es",
                duration_minutes=TEST_DEFAULT_DURATION,
            ),
        )

        agent = MockAgent(mock_llm)
        injection = agent._get_language_injection(state)

        # Spanish should show "Spanish" (from get_language_name)
        assert "Spanish" in injection
