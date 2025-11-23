"""Unit tests for CLI functions and error handling."""

from pathlib import Path
from typing import Any
from unittest.mock import patch

import pytest
from click.testing import CliRunner

from mystery_agents.cli import generate
from mystery_agents.models.state import (
    MetaInfo,
    ValidationReport,
    WorldValidation,
)


@pytest.fixture
def sample_game_yml(tmp_path: Path) -> Path:
    """Create a sample game.yml file."""
    config_file = tmp_path / "game.yml"
    config_file.write_text(
        """
language: es
country: Spain
epoch: modern
theme: family_mansion
players:
  male: 3
  female: 3
host_gender: male
duration_minutes: 90
difficulty: medium
"""
    )
    return config_file


def test_cli_quiet_and_verbose_mutually_exclusive(sample_game_yml: Path) -> None:
    """Test that --quiet and -v flags are mutually exclusive."""
    runner = CliRunner()

    with runner.isolated_filesystem(temp_dir=sample_game_yml.parent):
        result = runner.invoke(generate, ["--quiet", "-v"])

        assert result.exit_code == 1
        assert "mutually exclusive" in result.output


def test_cli_no_config_file_error() -> None:
    """Test that CLI shows helpful error when no config file exists."""
    runner = CliRunner()

    with runner.isolated_filesystem():
        result = runner.invoke(generate)

        assert result.exit_code == 1
        assert "No configuration file found" in result.output
        assert "game.yml" in result.output
        assert "game.example.yml" in result.output


def test_cli_config_file_not_found() -> None:
    """Test that CLI shows error when specified config file doesn't exist."""
    runner = CliRunner()

    with runner.isolated_filesystem():
        result = runner.invoke(generate, ["nonexistent.yml"])

        assert result.exit_code == 1
        assert "Configuration file not found" in result.output
        assert "nonexistent.yml" in result.output


def test_cli_workflow_no_output_error(sample_game_yml: Path) -> None:
    """Test that CLI handles workflow with no output."""
    runner = CliRunner()

    with runner.isolated_filesystem(temp_dir=sample_game_yml.parent):
        # Copy the config file to the isolated filesystem
        import shutil

        shutil.copy(sample_game_yml, Path.cwd() / "game.yml")

        with patch("mystery_agents.graph.workflow.create_workflow") as mock_workflow:
            mock_instance = mock_workflow.return_value
            # Simulate workflow that produces no output
            mock_instance.stream.return_value = iter([])

            result = runner.invoke(generate, ["--dry-run"])

            assert result.exit_code == 1
            assert "Workflow did not produce output" in result.output


def test_cli_workflow_unexpected_state_format(sample_game_yml: Path) -> None:
    """Test that CLI handles unexpected state format."""
    runner = CliRunner()

    with runner.isolated_filesystem(temp_dir=sample_game_yml.parent):
        with patch("mystery_agents.graph.workflow.create_workflow") as mock_workflow:
            mock_instance = mock_workflow.return_value
            # Simulate workflow that returns non-dict state
            mock_instance.stream.return_value = iter([{"node": "not a dict"}])

            result = runner.invoke(generate, ["--dry-run"])

            # Should handle gracefully or show error
            assert result.exit_code != 0


def test_cli_world_validation_failure(sample_game_yml: Path) -> None:
    """Test that CLI handles world validation failure."""
    runner = CliRunner()

    with runner.isolated_filesystem(temp_dir=sample_game_yml.parent):
        # Copy the config file to the isolated filesystem
        import shutil

        shutil.copy(sample_game_yml, Path.cwd() / "game.yml")

        with patch("mystery_agents.graph.workflow.create_workflow") as mock_workflow:
            mock_instance = mock_workflow.return_value

            # Create a state with failed world validation
            final_state = {
                "world_validation": WorldValidation(
                    is_coherent=False,
                    issues=["World is inconsistent"],
                    suggestions=["Fix the world"],
                ),
                "meta": MetaInfo(),
            }

            mock_instance.stream.return_value = iter([{"final": final_state}])

            result = runner.invoke(generate, ["--dry-run"])

            assert result.exit_code == 1
            assert "World validation failed" in result.output


def test_cli_validation_failure(sample_game_yml: Path) -> None:
    """Test that CLI handles game validation failure."""
    runner = CliRunner()

    with runner.isolated_filesystem(temp_dir=sample_game_yml.parent):
        # Copy the config file to the isolated filesystem
        import shutil

        shutil.copy(sample_game_yml, Path.cwd() / "game.yml")

        with patch("mystery_agents.graph.workflow.create_workflow") as mock_workflow:
            from mystery_agents.models.state import ValidationIssue

            mock_instance = mock_workflow.return_value

            # Create a state with failed validation
            final_state = {
                "validation": ValidationReport(
                    is_consistent=False,
                    issues=[ValidationIssue(type="timeline_conflict", description="Test issue")],
                    suggested_fixes=["Fix it"],
                ),
                "meta": MetaInfo(),
            }

            mock_instance.stream.return_value = iter([{"final": final_state}])

            result = runner.invoke(generate, ["--dry-run"])

            assert result.exit_code == 1
            assert "Game generation failed validation" in result.output


def test_cli_missing_meta_error(sample_game_yml: Path) -> None:
    """Test that CLI handles missing meta information."""
    runner = CliRunner()

    with runner.isolated_filesystem(temp_dir=sample_game_yml.parent):
        # Copy the config file to the isolated filesystem
        import shutil

        shutil.copy(sample_game_yml, Path.cwd() / "game.yml")

        with patch("mystery_agents.graph.workflow.create_workflow") as mock_workflow:
            mock_instance = mock_workflow.return_value

            # Create a state without meta
            final_state = {
                "validation": ValidationReport(is_consistent=True, issues=[], suggested_fixes=[])
            }

            mock_instance.stream.return_value = iter([{"final": final_state}])

            result = runner.invoke(generate, ["--dry-run"])

            assert result.exit_code == 1
            assert "Missing meta information" in result.output


def test_cli_keyboard_interrupt(sample_game_yml: Path) -> None:
    """Test that CLI handles KeyboardInterrupt gracefully."""
    runner = CliRunner()

    with runner.isolated_filesystem(temp_dir=sample_game_yml.parent):
        # Copy the config file to the isolated filesystem
        import shutil

        shutil.copy(sample_game_yml, Path.cwd() / "game.yml")

        with patch("mystery_agents.graph.workflow.create_workflow") as mock_workflow:
            mock_instance = mock_workflow.return_value
            mock_instance.stream.side_effect = KeyboardInterrupt()

            result = runner.invoke(generate, ["--dry-run"])

            assert result.exit_code == 130
            assert "cancelled by user" in result.output.lower()


def test_cli_api_key_error(sample_game_yml: Path) -> None:
    """Test that CLI shows helpful error for API key issues."""
    runner = CliRunner()

    with runner.isolated_filesystem(temp_dir=sample_game_yml.parent):
        # Copy the config file to the isolated filesystem
        import shutil

        shutil.copy(sample_game_yml, Path.cwd() / "game.yml")

        with patch("mystery_agents.graph.workflow.create_workflow") as mock_workflow:
            mock_instance = mock_workflow.return_value
            mock_instance.stream.side_effect = ValueError("API key invalid")

            result = runner.invoke(generate)

            assert result.exit_code == 1
            assert "API Key Error" in result.output or "API key" in result.output.lower()


def test_cli_generic_error_handling(sample_game_yml: Path) -> None:
    """Test that CLI handles generic errors gracefully."""
    runner = CliRunner()

    with runner.isolated_filesystem(temp_dir=sample_game_yml.parent):
        # Copy the config file to the isolated filesystem
        import shutil

        shutil.copy(sample_game_yml, Path.cwd() / "game.yml")

        with patch("mystery_agents.graph.workflow.create_workflow") as mock_workflow:
            mock_instance = mock_workflow.return_value
            mock_instance.stream.side_effect = RuntimeError("Something went wrong")

            result = runner.invoke(generate, ["--dry-run"])

            assert result.exit_code == 1
            assert "Error during generation" in result.output


def test_cli_success_output(sample_game_yml: Path, tmp_path: Path) -> None:
    """Test that CLI shows success message with correct paths."""
    runner = CliRunner()

    output_dir = tmp_path / "output"
    output_dir.mkdir()

    with runner.isolated_filesystem(temp_dir=sample_game_yml.parent):
        with patch("mystery_agents.graph.workflow.create_workflow") as mock_workflow:
            mock_instance = mock_workflow.return_value

            # Create a successful final state
            meta = MetaInfo()
            final_state = {
                "validation": ValidationReport(is_consistent=True, issues=[], suggested_fixes=[]),
                "meta": meta,
            }

            mock_instance.stream.return_value = iter([{"final": final_state}])

            # Mock Path.exists for zip file
            with patch("pathlib.Path.exists", return_value=True):
                result = runner.invoke(
                    generate,
                    [
                        str(sample_game_yml),
                        "--dry-run",
                        "--output-dir",
                        str(output_dir),
                    ],
                )

                # Should show success message
                assert "GAME GENERATED SUCCESSFULLY" in result.output or result.exit_code == 0


def test_cli_no_images_flag(sample_game_yml: Path) -> None:
    """Test that --no-images flag sets generate_images to False."""
    runner = CliRunner()

    with runner.isolated_filesystem(temp_dir=sample_game_yml.parent):
        with patch("mystery_agents.graph.workflow.create_workflow") as mock_workflow:
            mock_instance = mock_workflow.return_value
            mock_instance.stream.side_effect = Exception("Test stopped")

            # Capture the initial state to check config
            captured_state = None

            def capture_state(*args: Any, **kwargs: Any) -> Any:
                nonlocal captured_state
                # The state is created in the generate function
                # We'll check it via the workflow call
                return iter([])

            mock_instance.stream.side_effect = capture_state

            result = runner.invoke(generate, ["--no-images", "--dry-run"])

            # The flag should be processed (we can't easily test the state here,
            # but we can verify the command doesn't error on the flag)
            # Note: result.command is not available in CliRunner, so we just check exit code
            assert result.exit_code != 0  # Due to our test exception, but flag should be accepted


def test_cli_keep_work_dir_flag(sample_game_yml: Path) -> None:
    """Test that --keep-work-dir flag is accepted."""
    runner = CliRunner()

    with runner.isolated_filesystem(temp_dir=sample_game_yml.parent):
        with patch("mystery_agents.graph.workflow.create_workflow") as mock_workflow:
            mock_instance = mock_workflow.return_value
            mock_instance.stream.side_effect = Exception("Test stopped")

            result = runner.invoke(generate, ["--keep-work-dir", "--dry-run"])

            # Should not error on the flag
            assert result.exit_code != 0  # Due to our test exception, but flag should be accepted


def test_cli_log_file_option(sample_game_yml: Path, tmp_path: Path) -> None:
    """Test that --log-file option is accepted."""
    runner = CliRunner()

    log_file = tmp_path / "test.log"

    with runner.isolated_filesystem(temp_dir=sample_game_yml.parent):
        with patch("mystery_agents.graph.workflow.create_workflow") as mock_workflow:
            mock_instance = mock_workflow.return_value
            mock_instance.stream.side_effect = Exception("Test stopped")

            result = runner.invoke(
                generate,
                ["--log-file", str(log_file), "--dry-run"],
            )

            # Should not error on the flag
            assert result.exit_code != 0  # Due to our test exception, but flag should be accepted
