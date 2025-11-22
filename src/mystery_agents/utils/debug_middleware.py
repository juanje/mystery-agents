"""Debugging middleware for LangChain agents."""

import json
from typing import Any

from langchain.agents.middleware import AgentState, after_model
from langchain_core.messages import AIMessage
from langgraph.runtime import Runtime


def _log_model_response_impl(state: AgentState, runtime: Runtime) -> dict[str, Any] | None:
    """
    Internal implementation of log_model_response.

    This is the actual function that does the logging.
    The @after_model decorator wraps this to create the middleware.
    """
    """
    Log model responses for debugging.

    This middleware logs:
    - The raw response content from the model
    - The structured response if available
    - Any errors or issues with the response format
    - Full state information for debugging
    """
    print("\n" + "=" * 80)
    print("[DEBUG MIDDLEWARE] After Model Hook")
    print("=" * 80)

    # Log state keys
    print(
        f"\n[DEBUG] State keys: {list(state.keys()) if isinstance(state, dict) else 'Not a dict'}"
    )

    # Log messages
    messages = state.get("messages", []) if isinstance(state, dict) else []
    print(f"\n[DEBUG] Number of messages: {len(messages)}")

    if messages:
        last_message = messages[-1]
        print(f"\n[DEBUG] Last message type: {type(last_message).__name__}")

        if isinstance(last_message, AIMessage):
            print("[DEBUG] Response Content:")
            print("-" * 80)
            content = last_message.content
            if content:
                # Show first 1000 chars, or full content if shorter
                preview = content[:1000]
                print(preview)
                if len(content) > 1000:
                    print(f"\n... (truncated, total length: {len(content)} chars)")
            else:
                print("(empty content)")
            print("-" * 80)

            # Try to parse as JSON if it looks like JSON
            if content and isinstance(content, str) and content.strip().startswith("{"):
                try:
                    parsed = json.loads(content)
                    print("\n[DEBUG] Content is valid JSON:")
                    print(json.dumps(parsed, indent=2, default=str)[:500])
                except json.JSONDecodeError:
                    print("\n[DEBUG] Content looks like JSON but is invalid")
        else:
            print(f"[DEBUG] Last message: {last_message}")

    # Check for structured response
    if isinstance(state, dict):
        if "structured_response" in state:
            print("\n[DEBUG] ✓ Structured Response Found in state:")
            try:
                structured = state["structured_response"]
                print(f"[DEBUG] Type: {type(structured).__name__}")
                if hasattr(structured, "model_dump"):
                    print("[DEBUG] Structured Data (first 1000 chars):")
                    data_str = json.dumps(structured.model_dump(), indent=2, default=str)
                    print(data_str[:1000])
                    if len(data_str) > 1000:
                        print("... (truncated)")
                else:
                    print(f"[DEBUG] Structured Data: {structured}")
            except Exception as e:
                print(f"[DEBUG] ✗ Error accessing structured response: {e}")
                import traceback

                traceback.print_exc()
        else:
            print("\n[DEBUG] ✗ No structured_response in state")
            print(f"[DEBUG] Available state keys: {list(state.keys())}")

    print("=" * 80 + "\n")

    return None


@after_model
def log_model_response(state: AgentState, runtime: Runtime) -> dict[str, Any] | None:
    """
    Log model responses for debugging (middleware wrapper).

    This is the middleware entry point created by @after_model decorator.
    """
    return _log_model_response_impl(state, runtime)
