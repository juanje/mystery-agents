"""Unit tests for A9 Packaging Agent parallel PDF generation."""

from __future__ import annotations

from unittest.mock import MagicMock

from mystery_agents.agents.a9_packaging import PackagingAgent


def test_generate_all_pdfs_empty_list() -> None:
    """Test that empty PDF task list is handled gracefully."""
    # Create agent without initializing LLM (mock it)
    agent = PackagingAgent.__new__(PackagingAgent)
    agent.llm = MagicMock()

    # Should not raise any exception and return immediately
    agent._generate_all_pdfs([], max_workers=2)


def test_agent_has_parallel_method() -> None:
    """Test that PackagingAgent has the parallel PDF generation method."""
    # Create agent without initializing LLM (mock it)
    agent = PackagingAgent.__new__(PackagingAgent)
    agent.llm = MagicMock()

    # Verify the method exists
    assert hasattr(agent, "_generate_all_pdfs")
    assert callable(agent._generate_all_pdfs)


def test_parallel_generation_method_signature() -> None:
    """Test that _generate_all_pdfs has correct signature."""
    # Use inspect on the unbound method to avoid instantiation overhead
    import inspect

    method = PackagingAgent._generate_all_pdfs
    sig = inspect.signature(method)
    params = list(sig.parameters.keys())

    # First param is 'self' for instance methods
    assert "self" in params
    assert "pdf_tasks" in params
    assert "max_workers" in params

    # Check default value for max_workers
    assert sig.parameters["max_workers"].default == 12


def test_imports_are_present() -> None:
    """Test that required imports are present in the module."""
    import mystery_agents.agents.a9_packaging as module

    # Verify ProcessPoolExecutor is imported
    assert hasattr(module, "ProcessPoolExecutor")

    # Verify time is imported
    assert hasattr(module, "time")

    # Verify worker function exists
    assert hasattr(module, "_generate_pdf_worker")
