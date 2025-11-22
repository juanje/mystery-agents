"""Shared test fixtures and configuration."""

import pytest

# Constant for mock API key - centralized to avoid duplication
MOCK_API_KEY = "test-mock-api-key-for-testing"


@pytest.fixture(autouse=True)
def mock_google_api_key(monkeypatch: pytest.MonkeyPatch) -> None:
    """
    Mock GOOGLE_API_KEY environment variable for all tests.

    This ensures tests don't depend on actual API keys in the environment
    and provides a consistent mock value across all test modules.

    autouse=True ensures this fixture is automatically applied to all tests
    without needing to explicitly request it.
    """
    monkeypatch.setenv("GOOGLE_API_KEY", MOCK_API_KEY)
