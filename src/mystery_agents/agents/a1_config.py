"""A1: Configuration/Wizard Agent - Collects user preferences."""

from typing import Any, Literal, cast

import click
import yaml

from mystery_agents.models.state import (
    DifficultyLevel,
    Epoch,
    GameConfig,
    GameState,
    PlayerConfig,
    Theme,
)
from mystery_agents.utils.constants import (
    DEFAULT_COUNTRY_EN,
    DEFAULT_COUNTRY_ES,
    LANG_CODE_ENGLISH,
    LANG_CODE_SPANISH,
)


class ConfigWizardAgent:
    """
    A1: Config/Wizard Agent.

    Collects user preferences via CLI and populates GameConfig.
    Supports loading configuration from YAML file.
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
        Run the configuration wizard or load from YAML file.

        Args:
            state: Current game state

        Returns:
            Updated game state with populated config
        """
        # If config file is provided, load from YAML and skip wizard
        if state.config.config_file:
            click.echo("\n=== Loading Configuration from File ===\n")
            click.echo(f"  File: {state.config.config_file}")
            try:
                config = self._load_from_yaml(state.config.config_file, state)
                state.config = config

                click.echo("\n✓ Configuration loaded successfully!\n")
                self._display_config_summary(config)
                return state
            except ValueError as e:
                click.echo(f"\n❌ Error loading configuration: {e}", err=True)
                click.echo("   Falling back to interactive wizard...\n")
                # Continue to interactive wizard below

        click.echo("\n=== Mystery Party Game Generator ===\n")
        click.echo("Let's configure your mystery party game!\n")

        # Language
        language = click.prompt(
            "Language / Idioma",
            type=click.Choice([LANG_CODE_SPANISH, LANG_CODE_ENGLISH], case_sensitive=False),
            default=LANG_CODE_SPANISH,
        )

        # Country (for name generation)
        country_default = (
            DEFAULT_COUNTRY_ES if language == LANG_CODE_SPANISH else DEFAULT_COUNTRY_EN
        )
        country = click.prompt(
            "Country / País (for character names)",
            default=country_default,
            type=str,
        )

        # Region (optional for more specific cultural context)
        region = click.prompt(
            "\nRegion within country (optional, press Enter to skip)",
            type=str,
            default="",
            show_default=False,
        )
        region = region.strip() or None

        # Epoch
        epochs: list[tuple[int, str, Epoch]] = [
            (1, "modern - Contemporary setting", "modern"),
            (2, "1920s - Roaring Twenties", "1920s"),
            (3, "victorian - Victorian era", "victorian"),
            (4, "custom - Your own setting", "custom"),
        ]
        click.echo("\nAvailable epochs:")
        for num, desc, _ in epochs:
            click.echo(f"  {num}. {desc}")
        epoch_choice = click.prompt("Choose epoch", type=int, default=1)
        epoch = cast(Epoch, next((e for n, _, e in epochs if n == epoch_choice), "modern"))

        custom_epoch_description = None
        if epoch == "custom":
            custom_epoch_description = click.prompt("Describe your custom epoch")

        # Theme
        themes: list[tuple[int, str, Theme]] = [
            (1, "family_mansion - Family gathering in a mansion", "family_mansion"),
            (2, "corporate_retreat - Corporate event", "corporate_retreat"),
            (3, "cruise - Luxury cruise ship", "cruise"),
            (4, "train - Orient Express style train", "train"),
            (5, "custom - Your own theme", "custom"),
        ]
        click.echo("\nAvailable themes:")
        for num, desc, _ in themes:
            click.echo(f"  {num}. {desc}")
        theme_choice = click.prompt("Choose theme", type=int, default=1)
        theme = cast(Theme, next((t for n, _, t in themes if n == theme_choice), "family_mansion"))

        custom_theme_description = None
        if theme == "custom":
            custom_theme_description = click.prompt("Describe your custom theme")

        # Player gender distribution
        click.echo("\n" + "=" * 60)
        click.echo("PLAYER CHARACTERS (suspects)")
        click.echo("=" * 60)
        click.echo("ℹ️  These are the PLAYERS (suspects) at your party.")
        click.echo("   The HOST (victim) is SEPARATE and will be added automatically.")
        click.echo("   Example: 6 players + 1 host = 7 total people at the party")
        click.echo("\nSpecify gender distribution for PLAYERS (press Enter for balanced 3/3):")
        male = click.prompt("  Male player characters", type=click.IntRange(0, 10), default=3)
        female = click.prompt("  Female player characters", type=click.IntRange(0, 10), default=3)

        # Calculate total and validate
        total_players = male + female
        if total_players < 4:
            click.echo(
                "⚠️  Warning: Minimum 4 players required. Adjusting to 4 players (2 male, 2 female)."
            )
            male = 2
            female = 2
            total_players = 4
        elif total_players > 10:
            click.echo(
                "⚠️  Warning: Maximum 10 players allowed. Adjusting to 10 players (5 male, 5 female)."
            )
            male = 5
            female = 5
            total_players = 10

        # Host gender
        click.echo("\n" + "=" * 60)
        click.echo("HOST CHARACTER (victim)")
        click.echo("=" * 60)
        click.echo("ℹ️  The HOST plays the victim in Act 1, then becomes detective in Act 2.")
        host_gender_choice = click.prompt(
            "  Host gender",
            type=click.Choice(["male", "female"], case_sensitive=False),
            default="male",
        )
        host_gender: Literal["male", "female"] = cast(Literal["male", "female"], host_gender_choice)

        # Duration
        duration = click.prompt(
            "\nGame duration (minutes)",
            type=click.IntRange(60, 180),
            default=90,
        )

        # Difficulty
        difficulties: list[tuple[int, str, DifficultyLevel]] = [
            (1, "easy - Simple mystery, obvious clues", "easy"),
            (2, "medium - Balanced challenge", "medium"),
            (3, "hard - Complex mystery, subtle clues", "hard"),
        ]
        click.echo("\nDifficulty levels:")
        for num, desc, _ in difficulties:
            click.echo(f"  {num}. {desc}")
        difficulty_choice = click.prompt("Choose difficulty", type=int, default=2)
        difficulty = cast(
            DifficultyLevel,
            next((d for n, _, d in difficulties if n == difficulty_choice), "medium"),
        )

        # Create config (dry_run, debug_model, and generate_images come from CLI flags)
        config = GameConfig(
            language=language,
            country=country,
            region=region,
            epoch=epoch,
            custom_epoch_description=custom_epoch_description,
            theme=theme,
            custom_theme_description=custom_theme_description,
            players=PlayerConfig(total=total_players, male=male, female=female),
            host_gender=host_gender,
            duration_minutes=duration,
            difficulty=difficulty,
            generate_images=state.config.generate_images,
            dry_run=state.config.dry_run,
            debug_model=state.config.debug_model,
        )

        # Update state
        state.config = config

        click.echo("\n")
        self._display_config_summary(config)

        return state
