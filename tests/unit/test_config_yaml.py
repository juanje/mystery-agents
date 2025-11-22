"""Unit tests for YAML configuration loading."""

from pathlib import Path

import pytest

from mystery_agents.agents.a1_config import ConfigLoaderAgent
from mystery_agents.models.state import GameConfig, GameState, MetaInfo


def test_load_from_yaml_valid_config(tmp_path: Path) -> None:
    """Test loading a valid YAML configuration file."""
    yaml_content = """
language: es
country: Spain
region: Andalucía
epoch: modern
theme: family_mansion
players:
  male: 3
  female: 3
host_gender: male
duration_minutes: 90
difficulty: medium
"""

    config_file = tmp_path / "config.yaml"
    config_file.write_text(yaml_content)

    agent = ConfigLoaderAgent()
    state = GameState(
        meta=MetaInfo(),
        config=GameConfig(
            dry_run=True,
            generate_images=False,
            duration_minutes=90,
        ),
    )

    config = agent._load_from_yaml(str(config_file), state)

    assert config.language == "es"
    assert config.country == "Spain"
    assert config.region == "Andalucía"
    assert config.epoch == "modern"
    assert config.theme == "family_mansion"
    assert config.players.male == 3
    assert config.players.female == 3
    assert config.players.total == 6
    assert config.host_gender == "male"
    assert config.duration_minutes == 90
    assert config.difficulty == "medium"
    # CLI flags should be preserved
    assert config.dry_run is True
    assert config.generate_images is False


def test_load_from_yaml_minimal_config(tmp_path: Path) -> None:
    """Test loading a minimal YAML configuration (only required fields)."""
    yaml_content = """
language: en
country: United States
epoch: 1920s
theme: cruise
host_gender: female
"""

    config_file = tmp_path / "config.yaml"
    config_file.write_text(yaml_content)

    agent = ConfigLoaderAgent()
    state = GameState(
        meta=MetaInfo(),
        config=GameConfig(dry_run=True, duration_minutes=90),
    )

    config = agent._load_from_yaml(str(config_file), state)

    assert config.language == "en"
    assert config.country == "United States"
    assert config.region is None
    assert config.epoch == "1920s"
    assert config.theme == "cruise"
    # Should use defaults
    assert config.players.total == 6  # Default 3+3
    assert config.duration_minutes == 90
    assert config.difficulty == "medium"


def test_load_from_yaml_missing_required_field(tmp_path: Path) -> None:
    """Test loading YAML with missing required fields."""
    yaml_content = """
language: es
country: Spain
# Missing: epoch, theme, host_gender
"""

    config_file = tmp_path / "config.yaml"
    config_file.write_text(yaml_content)

    agent = ConfigLoaderAgent()
    state = GameState(
        meta=MetaInfo(),
        config=GameConfig(dry_run=True, duration_minutes=90),
    )

    with pytest.raises(ValueError, match="Missing required fields"):
        agent._load_from_yaml(str(config_file), state)


def test_load_from_yaml_file_not_found() -> None:
    """Test loading from non-existent file."""
    agent = ConfigLoaderAgent()
    state = GameState(
        meta=MetaInfo(),
        config=GameConfig(dry_run=True, duration_minutes=90),
    )

    with pytest.raises(ValueError, match="Configuration file not found"):
        agent._load_from_yaml("/nonexistent/file.yaml", state)


def test_load_from_yaml_invalid_yaml(tmp_path: Path) -> None:
    """Test loading invalid YAML syntax."""
    yaml_content = """
language: es
country: Spain
  invalid: indentation
"""

    config_file = tmp_path / "config.yaml"
    config_file.write_text(yaml_content)

    agent = ConfigLoaderAgent()
    state = GameState(
        meta=MetaInfo(),
        config=GameConfig(dry_run=True, duration_minutes=90),
    )

    with pytest.raises(ValueError, match="Invalid YAML file"):
        agent._load_from_yaml(str(config_file), state)


def test_load_from_yaml_preserves_cli_flags(tmp_path: Path) -> None:
    """Test that CLI flags override YAML values."""
    yaml_content = """
language: es
country: Spain
epoch: modern
theme: family_mansion
host_gender: male
"""

    config_file = tmp_path / "config.yaml"
    config_file.write_text(yaml_content)

    agent = ConfigLoaderAgent()
    state = GameState(
        meta=MetaInfo(),
        config=GameConfig(
            dry_run=True,
            generate_images=True,
            debug_model=True,
            duration_minutes=90,
        ),
    )

    config = agent._load_from_yaml(str(config_file), state)

    # CLI flags should be preserved
    assert config.dry_run is True
    assert config.generate_images is True
    assert config.debug_model is True
