"""CLI entry point for mystery party game generator."""

import sys
from pathlib import Path

import click

from mystery_agents.graph.workflow import create_workflow
from mystery_agents.models.state import GameConfig, GameState, MetaInfo, PlayerConfig
from mystery_agents.utils.constants import (
    CLUES_DIR,
    DEFAULT_OUTPUT_DIR,
    DEFAULT_RECURSION_LIMIT,
    GAME_ID_LENGTH,
    HOST_DIR,
    PLAYERS_DIR,
)


@click.command()
@click.option(
    "--output-dir",
    default=DEFAULT_OUTPUT_DIR,
    help="Output directory for generated games",
    type=click.Path(path_type=Path),
)
@click.option(
    "--dry-run",
    is_flag=True,
    default=False,
    help="Use mock data instead of calling LLMs (fast testing)",
)
@click.option(
    "--debug",
    is_flag=True,
    default=False,
    help="Enable debug mode to log model responses (useful for troubleshooting)",
)
def generate(output_dir: Path, dry_run: bool, debug: bool) -> None:
    """
    Generate a mystery party game.

    This command launches the mystery party game generator, which will:
    1. Collect your preferences (theme, tone, players, etc.)
    2. Generate a complete mystery with characters, clues, and plot
    3. Create all materials needed to run the game
    4. Package everything into a ready-to-use ZIP file
    """
    click.echo("\n" + "=" * 60)
    click.echo("       MYSTERY PARTY GAME GENERATOR")
    click.echo("=" * 60 + "\n")

    # Create output directory if it doesn't exist
    output_dir.mkdir(parents=True, exist_ok=True)

    initial_state = GameState(
        meta=MetaInfo(),
        config=GameConfig(
            players=PlayerConfig(total=6),
            dry_run=dry_run,
            debug_model=debug,
            duration_minutes=90,
        ),
    )

    # Create and run workflow
    try:
        click.echo("Initializing workflow...\n")
        workflow = create_workflow()

        config = {"recursion_limit": DEFAULT_RECURSION_LIMIT}
        all_states = []
        for output in workflow.stream(initial_state, config=config):
            for node_name, state in output.items():
                all_states.append((node_name, state))

        # Check if we got any states
        if not all_states:
            click.echo("\n❌ Error: Workflow did not produce output", err=True)
            sys.exit(1)

        final_node_name, final_state = all_states[-1]

        if not isinstance(final_state, dict):
            click.echo("\n❌ Error: Unexpected state format", err=True)
            sys.exit(1)

        # Check world validation first (V2)
        world_validation = final_state.get("world_validation")
        if world_validation and not world_validation.is_coherent:
            click.echo(
                "\n❌ Game generation failed: World validation failed after max retries", err=True
            )
            click.echo("\nWorld coherence issues:")
            for issue in world_validation.issues:
                click.echo(f"  - {issue}")
            if world_validation.suggestions:
                click.echo("\nSuggestions for improvement:")
                for suggestion in world_validation.suggestions:
                    click.echo(f"  - {suggestion}")
            sys.exit(1)

        # Check full game validation (V1)
        validation = final_state.get("validation")
        if validation and not validation.is_consistent:
            click.echo("\n❌ Game generation failed validation", err=True)
            click.echo("\nValidation issues:")
            for issue in validation.issues:
                click.echo(f"  - {issue.type}: {issue.description}")
            click.echo("\nSuggested fixes:")
            for fix in validation.suggested_fixes:
                click.echo(f"  - {fix}")
            sys.exit(1)

        # Success!
        click.echo("\n" + "=" * 60)
        click.echo("✓ GAME GENERATED SUCCESSFULLY!")
        click.echo("=" * 60 + "\n")

        packaging = final_state.get("packaging")
        if packaging:
            click.echo(packaging.index_summary)

        meta = final_state.get("meta")
        if not meta:
            click.echo("\n❌ Error: Missing meta information", err=True)
            sys.exit(1)
        game_id = meta.id[:GAME_ID_LENGTH]
        from mystery_agents.utils.constants import GAME_DIR_PREFIX, ZIP_FILE_PREFIX

        zip_path = output_dir / f"{ZIP_FILE_PREFIX}{game_id}.zip"

        click.echo("\n📦 Your game package is ready:")
        click.echo(f"   {zip_path}")
        click.echo("\n📁 Unpacked files are in:")
        click.echo(f"   {output_dir / f'{GAME_DIR_PREFIX}{game_id}'}")

        click.echo("\n" + "=" * 60)
        click.echo("NEXT STEPS:")
        click.echo("=" * 60)
        click.echo("1. Extract the ZIP file")
        from mystery_agents.utils.constants import HOST_GUIDE_FILENAME

        click.echo(f"2. Read the host guide in /{HOST_DIR}/{HOST_GUIDE_FILENAME}")
        click.echo(f"3. Send each player their package from /{PLAYERS_DIR}/")
        click.echo(f"4. Print or prepare the clues from /{CLUES_DIR}/")
        click.echo("5. Host an amazing mystery party!")
        click.echo("\nHave fun! 🎭\n")

    except KeyboardInterrupt:
        click.echo("\n\n⚠ Generation cancelled by user", err=True)
        sys.exit(130)
    except Exception as e:
        click.echo(f"\n\n❌ Error during generation: {e}", err=True)
        import traceback

        traceback.print_exc()
        sys.exit(1)


@click.group()
def cli() -> None:
    """Mystery Party Game Generator - Create complete mystery games with AI."""
    pass


cli.add_command(generate)


if __name__ == "__main__":
    generate()
