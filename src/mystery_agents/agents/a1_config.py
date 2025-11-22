"""A1: Configuration Loader Agent - Loads and validates YAML configuration."""

from __future__ import annotations

from typing import Any

import click
import yaml

from mystery_agents.models.state import GameConfig, GameState, PlayerConfig


class ConfigLoaderAgent:
    """
    A1: Config Loader Agent.

    Loads game configuration from YAML file and validates it.
    Designed for both CLI and future web interface use.
    """

    def _load_from_yaml(self, yaml_path: str, state: GameState) -> GameConfig:
        """
        Load configuration from YAML file.

        Args:
            yaml_path: Path to YAML configuration file
            state: Current game state (for preserving CLI flags)

        Returns:
            GameConfig loaded from YAML

        Raises:
            ValueError: If YAML file is invalid or missing required fields
        """
        try:
            with open(yaml_path, encoding="utf-8") as f:
                data: dict[str, Any] = yaml.safe_load(f)
        except FileNotFoundError as e:
            raise ValueError(f"Configuration file not found: {yaml_path}") from e
        except yaml.YAMLError as e:
            raise ValueError(f"Invalid YAML file: {e}") from e

        if not isinstance(data, dict):
            raise ValueError("YAML file must contain a dictionary")

        # Validate required fields
        required_fields = {"language", "country", "epoch", "theme", "host_gender"}
        missing_fields = required_fields - set(data.keys())
        if missing_fields:
            raise ValueError(f"Missing required fields in YAML: {', '.join(missing_fields)}")

        # Handle players configuration (can be dict or just counts)
        players_config = data.get("players", {})
        if isinstance(players_config, dict):
            male = players_config.get("male", 3)
            female = players_config.get("female", 3)
            total = male + female
        else:
            raise ValueError("'players' must be a dictionary with 'male' and 'female' keys")

        # Create PlayerConfig
        players = PlayerConfig(total=total, male=male, female=female)

        # Create GameConfig with values from YAML + CLI flags
        config = GameConfig(
            language=data["language"],
            country=data["country"],
            region=data.get("region"),
            epoch=data["epoch"],
            custom_epoch_description=data.get("custom_epoch_description"),
            theme=data["theme"],
            custom_theme_description=data.get("custom_theme_description"),
            players=players,
            host_gender=data["host_gender"],
            duration_minutes=data.get("duration_minutes", 90),
            difficulty=data.get("difficulty", "medium"),
            # CLI flags override YAML
            generate_images=state.config.generate_images,
            dry_run=state.config.dry_run,
            debug_model=state.config.debug_model,
            keep_work_dir=state.config.keep_work_dir,
            config_file=yaml_path,
        )

        return config

    def _display_config_summary(self, config: GameConfig) -> None:
        """
        Display configuration summary.

        Args:
            config: Game configuration to display
        """
        click.echo("=" * 60)
        click.echo("✓ CONFIGURATION COMPLETE")
        click.echo("=" * 60 + "\n")
        click.echo(f"  Language: {config.language}")
        click.echo(f"  Country: {config.country}")
        if config.region:
            click.echo(f"  Region: {config.region}")
        click.echo(f"  Setting: {config.epoch} / {config.theme}")
        click.echo(f"  Duration: {config.duration_minutes} minutes")
        click.echo(f"  Difficulty: {config.difficulty}")
        click.echo()
        click.echo("  PARTY SIZE:")
        click.echo(
            f"    • {config.players.total} PLAYERS (suspects): {config.players.male} male, {config.players.female} female"
        )
        click.echo(f"    • 1 HOST (victim): {config.host_gender}")
        click.echo(f"    • TOTAL: {config.players.total + 1} people at the party")
        if config.generate_images:
            click.echo()
            click.echo("  Images: ✨ ENABLED (character portraits will be generated)")
        click.echo()

    def run(self, state: GameState) -> GameState:
        """
        Load configuration from YAML file.

        Args:
            state: Current game state with config_file path set

        Returns:
            Updated game state with populated config

        Raises:
            ValueError: If config file is missing or invalid
        """
        if not state.config.config_file:
            raise ValueError("Configuration file path is required")

        click.echo("\n=== Loading Configuration ===\n")
        click.echo(f"  File: {state.config.config_file}")

        try:
            config = self._load_from_yaml(state.config.config_file, state)
            state.config = config
            click.echo("\n✓ Configuration loaded successfully!\n")
            self._display_config_summary(config)
            return state
        except ValueError as e:
            click.echo(f"\n❌ Error loading configuration: {e}", err=True)
            raise
