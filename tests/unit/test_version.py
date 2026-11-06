"""Tests for the package version."""

import tomllib
from pathlib import Path

import mystery_agents


def test_version_matches_pyproject() -> None:
    """__version__ comes from the package metadata, so it always matches pyproject.toml."""
    pyproject = Path(__file__).parents[2] / "pyproject.toml"
    expected = tomllib.loads(pyproject.read_text())["project"]["version"]

    assert mystery_agents.__version__ == expected
