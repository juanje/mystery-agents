"""Base agent class for all game generation agents."""

from abc import ABC, abstractmethod
from typing import Any

from langchain_core.language_models import BaseChatModel
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_core.runnables import Runnable
from pydantic import BaseModel

from mystery_agents.config import structured_output
from mystery_agents.models.state import GameState
from mystery_agents.utils.constants import LANG_CODE_ENGLISH
from mystery_agents.utils.debug_output import describe_response, log_model_response
from mystery_agents.utils.i18n import get_language_name
from mystery_agents.utils.logging_config import AgentLogger
from mystery_agents.utils.prompts import FICTION_CONTEXT


class StructuredOutputError(ValueError):
    """The model response did not match the expected structured output schema."""


class BaseAgent(ABC):
    """
    Base class for all agents in the mystery game generation system.

    Uses provider-agnostic BaseChatModel to support any LLM provider.
    """

    def __init__(self, llm: BaseChatModel, response_format: type[BaseModel] | None = None) -> None:
        """
        Initialize the base agent.

        Args:
            llm: The language model to use (provider-agnostic BaseChatModel)
            response_format: Optional Pydantic model for structured output
        """
        self.llm = llm
        self.response_format = response_format

        self.runnable: Runnable[Any, Any] = (
            structured_output(llm, response_format) if response_format else llm
        )

    @abstractmethod
    def get_system_prompt(self, state: GameState) -> str:
        """
        Get the system prompt for this agent.

        Args:
            state: Current game state

        Returns:
            System prompt string
        """
        pass

    def get_response_format(self, state: GameState) -> type[BaseModel] | None:
        """
        Get the output schema for this call.

        Override to narrow the schema to the current game (e.g., only its room
        names), so the model can't produce values the game doesn't have.

        Args:
            state: Current game state

        Returns:
            The Pydantic model the response must follow
        """
        return self.response_format

    def _mock_output(self, state: GameState) -> GameState:
        """
        Generate mock output for dry run mode.

        Default implementation returns state unchanged.
        Override in subclasses that need mock data generation.

        Args:
            state: Current game state

        Returns:
            Updated game state with mock data (or unchanged if not overridden)
        """
        return state

    def _should_use_mock(self, state: GameState) -> bool:
        """
        Check if mock output should be used (dry run mode).

        Args:
            state: Current game state

        Returns:
            True if dry run mode is enabled
        """
        return state.config.dry_run

    def _get_language_injection(self, state: GameState) -> str:
        """
        Get language-specific instructions to inject into system prompt.

        For non-English languages, returns instructions to generate content
        in the target language while preserving JSON structure.

        Args:
            state: Current game state

        Returns:
            Language injection string (empty for English)
        """
        if state.config.language == LANG_CODE_ENGLISH:
            return ""

        target_lang = get_language_name(state.config.language)

        return f"""

---
CRITICAL LANGUAGE REQUIREMENTS (HIGHEST PRIORITY):

1. Output Language: ALL creative and narrative content (descriptions, dialogues,
   names, backstories, secrets, motives, etc.) MUST be written in fluent,
   natural {target_lang}.

2. JSON Structure Integrity: Keep ALL JSON keys in English.
   NEVER translate field names.
   ✓ CORRECT: {{"description": "Texto en español"}}
   ✗ WRONG:   {{"descripción": "Texto en español"}}

3. Context Consistency: Input context may already be in {target_lang}.
   Maintain narrative consistency and continue in that language.

4. Cultural Adaptation: Use culturally appropriate expressions, idioms,
   and references for {target_lang} speakers.

These requirements override any language assumptions in the instructions above.
"""

    def invoke(self, state: GameState, user_message: str = "") -> Any:
        """
        Invoke the agent with the current state.

        For non-English languages, automatically injects language instructions
        to generate content directly in the target language.

        Args:
            state: Current game state
            user_message: Optional user message (defaults to empty for auto-generation)

        Returns:
            Parsed response_format instance if set, otherwise the raw message content

        Raises:
            StructuredOutputError: If the structured output cannot be parsed
        """
        # Get base system prompt and inject language instructions if needed
        system_prompt = self.get_system_prompt(state)
        language_injection = self._get_language_injection(state)
        full_system_prompt = FICTION_CONTEXT + system_prompt + language_injection

        messages = [
            SystemMessage(content=full_system_prompt),
            HumanMessage(
                content=user_message
                if user_message
                else "Generate the required output based on the system prompt and current state."
            ),
        ]

        response_format = self.get_response_format(state)
        runnable: Runnable[Any, Any] = (
            self.runnable
            if response_format is self.response_format or response_format is None
            else structured_output(self.llm, response_format)
        )
        result = runnable.invoke(messages)

        if not response_format:
            return result.content

        parsed = result["parsed"]
        parsing_error = result["parsing_error"]
        if state.config.debug_model:
            log_model_response(result["raw"], parsed, parsing_error)

        if parsed is None:
            # The graph retries the node silently, so say why before it does
            message = (
                f"The LLM did not return a valid {response_format.__name__} "
                f"({describe_response(result['raw'])}): {parsing_error}"
            )
            AgentLogger(type(self).__name__, state).warning(f"⚠ {message[:500]}")
            raise StructuredOutputError(message)
        return parsed
