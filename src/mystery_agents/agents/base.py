"""Base agent class for all game generation agents."""

from abc import ABC, abstractmethod
from typing import Any

from langchain.agents import create_agent
from langchain_core.language_models import BaseChatModel
from langchain_core.messages import HumanMessage, SystemMessage
from pydantic import BaseModel

from mystery_agents.models.state import GameState
from mystery_agents.utils.constants import LANG_CODE_ENGLISH
from mystery_agents.utils.debug_middleware import log_model_response
from mystery_agents.utils.i18n import get_language_name


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

        # Create agent without middleware (will be added dynamically in invoke if debug is enabled)
        self.agent = create_agent(
            model=llm,
            tools=[],
            middleware=[],
            response_format=response_format,
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
            LLM response (structured if response_format is set, otherwise raw)
            For structured output, access via result["structured_response"]
        """
        # Get base system prompt and inject language instructions if needed
        system_prompt = self.get_system_prompt(state)
        language_injection = self._get_language_injection(state)
        full_system_prompt = system_prompt + language_injection

        messages = [
            SystemMessage(content=full_system_prompt),
            HumanMessage(
                content=user_message
                if user_message
                else "Generate the required output based on the system prompt and current state."
            ),
        ]

        # Use agent with debug middleware if debug_model is enabled
        agent_to_use = self.agent
        if state.config.debug_model and self.response_format:
            # Create agent with debug middleware for this invocation
            agent_to_use = create_agent(
                model=self.llm,
                tools=[],
                middleware=[log_model_response],
                response_format=self.response_format,
            )

        result: dict[str, Any] = agent_to_use.invoke({"messages": messages})  # type: ignore[arg-type]

        if self.response_format:
            if "structured_response" in result:
                return result["structured_response"]
            else:
                error_msg = (
                    "No structured_response in agent result. "
                    "The LLM returned text instead of the required structured format."
                )
                if state.config.debug_model:
                    # Provide detailed error info when debug is enabled
                    print(f"[DEBUG] ✗ {error_msg}")
                    print(f"[DEBUG] Available state keys: {list(result.keys())}")
                    if "messages" in result and result["messages"]:
                        last_msg = result["messages"][-1]
                        print(f"[DEBUG] Last message type: {type(last_msg).__name__}")
                        if hasattr(last_msg, "content"):
                            content = last_msg.content
                            if isinstance(content, str):
                                print(f"[DEBUG] Response Content:\n{'-' * 80}\n{content[:1000]}")
                                if len(content) > 1000:
                                    print(f"\n... (truncated, total length: {len(content)} chars)")
                            else:
                                print(f"[DEBUG] Response Content: {content}")
                raise ValueError(error_msg)

        return result["messages"][-1].content if result.get("messages") else result
