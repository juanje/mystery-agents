"""Unit tests for logging configuration and AgentLogger."""

from __future__ import annotations

import logging
from pathlib import Path
from unittest.mock import patch

import pytest

from mystery_agents.models.state import GameConfig, GameState, MetaInfo, PlayerConfig
from mystery_agents.utils.logging_config import AgentLogger, setup_logging


@pytest.fixture
def test_state_default() -> GameState:
    """Create a test state with default verbosity (0)."""
    return GameState(
        meta=MetaInfo(),
        config=GameConfig(
            players=PlayerConfig(total=6),
            duration_minutes=90,
            verbosity=0,
            quiet_mode=False,
        ),
    )


@pytest.fixture
def test_state_verbose() -> GameState:
    """Create a test state with INFO verbosity (1)."""
    return GameState(
        meta=MetaInfo(),
        config=GameConfig(
            players=PlayerConfig(total=6),
            duration_minutes=90,
            verbosity=1,
            quiet_mode=False,
        ),
    )


@pytest.fixture
def test_state_debug() -> GameState:
    """Create a test state with DEBUG verbosity (2)."""
    return GameState(
        meta=MetaInfo(),
        config=GameConfig(
            players=PlayerConfig(total=6),
            duration_minutes=90,
            verbosity=2,
            quiet_mode=False,
        ),
    )


@pytest.fixture
def test_state_quiet() -> GameState:
    """Create a test state with quiet mode enabled."""
    return GameState(
        meta=MetaInfo(),
        config=GameConfig(
            players=PlayerConfig(total=6),
            duration_minutes=90,
            verbosity=0,
            quiet_mode=True,
        ),
    )


def test_agent_logger_default_mode_uses_click(test_state_default: GameState) -> None:
    """Test that AgentLogger uses click.echo in default mode (verbosity=0)."""
    log = AgentLogger(__name__, test_state_default)

    with patch("mystery_agents.utils.logging_config.click.echo") as mock_echo:
        log.info("Test message")
        mock_echo.assert_called_once_with("Test message")


def test_agent_logger_verbose_mode_uses_logger(test_state_verbose: GameState) -> None:
    """Test that AgentLogger uses logger.info in verbose mode (verbosity=1)."""
    log = AgentLogger(__name__, test_state_verbose)

    with patch.object(log.logger, "info") as mock_info:
        log.info("Test message")
        mock_info.assert_called_once_with("Test message")


def test_agent_logger_debug_mode(test_state_debug: GameState) -> None:
    """Test that AgentLogger logs debug messages in -vv mode (verbosity=2)."""
    log = AgentLogger(__name__, test_state_debug)

    with patch.object(log.logger, "debug") as mock_debug:
        log.debug("Debug message")
        mock_debug.assert_called_once_with("Debug message")


def test_agent_logger_debug_not_shown_in_default(test_state_default: GameState) -> None:
    """Test that debug messages are not shown in default mode."""
    log = AgentLogger(__name__, test_state_default)

    with patch.object(log.logger, "debug") as mock_debug:
        log.debug("Debug message")
        mock_debug.assert_not_called()


def test_agent_logger_debug_not_shown_in_verbose(test_state_verbose: GameState) -> None:
    """Test that debug messages are not shown in -v mode (only -vv)."""
    log = AgentLogger(__name__, test_state_verbose)

    with patch.object(log.logger, "debug") as mock_debug:
        log.debug("Debug message")
        mock_debug.assert_not_called()


def test_agent_logger_quiet_mode_silences_info(test_state_quiet: GameState) -> None:
    """Test that quiet mode suppresses info messages."""
    log = AgentLogger(__name__, test_state_quiet)

    with patch("mystery_agents.utils.logging_config.click.echo") as mock_echo:
        log.info("Test message")
        mock_echo.assert_not_called()


def test_agent_logger_warning_always_shown(test_state_default: GameState) -> None:
    """Test that warnings are always shown (unless quiet)."""
    log = AgentLogger(__name__, test_state_default)

    with patch.object(log.logger, "warning") as mock_warning:
        log.warning("Warning message")
        mock_warning.assert_called_once_with("Warning message")


def test_agent_logger_warning_silenced_in_quiet(test_state_quiet: GameState) -> None:
    """Test that warnings are silenced in quiet mode."""
    log = AgentLogger(__name__, test_state_quiet)

    with patch.object(log.logger, "warning") as mock_warning:
        log.warning("Warning message")
        mock_warning.assert_not_called()


def test_agent_logger_error_always_shown(test_state_quiet: GameState) -> None:
    """Test that errors are always shown, even in quiet mode."""
    log = AgentLogger(__name__, test_state_quiet)

    with patch.object(log.logger, "error") as mock_error:
        log.error("Error message")
        mock_error.assert_called_once_with("Error message")


def test_setup_logging_creates_console_handler() -> None:
    """Test that setup_logging creates console handler for verbose mode."""
    setup_logging(verbosity=1, quiet=False, log_file=None)

    root_logger = logging.getLogger()
    assert len(root_logger.handlers) > 0
    assert root_logger.level == logging.INFO


def test_setup_logging_no_console_in_quiet() -> None:
    """Test that setup_logging doesn't create console handler in quiet mode."""
    setup_logging(verbosity=0, quiet=True, log_file=None)

    root_logger = logging.getLogger()
    # May have handlers from file, but not console
    assert root_logger.level == logging.WARNING


def test_setup_logging_debug_level() -> None:
    """Test that setup_logging sets DEBUG level for -vv."""
    setup_logging(verbosity=2, quiet=False, log_file=None)

    root_logger = logging.getLogger()
    assert root_logger.level == logging.DEBUG


def test_setup_logging_file_handler(tmp_path: Path) -> None:
    """Test that setup_logging creates file handler when log_file is specified."""
    log_file = tmp_path / "test.log"

    setup_logging(verbosity=1, quiet=False, log_file=str(log_file))

    root_logger = logging.getLogger()
    file_handlers = [h for h in root_logger.handlers if isinstance(h, logging.FileHandler)]
    assert len(file_handlers) > 0

    # Test that logging works
    test_logger = logging.getLogger("test")
    test_logger.info("Test log message")

    # Verify file was written
    assert log_file.exists()
    content = log_file.read_text()
    assert "Test log message" in content


def test_custom_formatter_extracts_agent_name() -> None:
    """Test that custom formatter extracts agent name from module path."""
    from mystery_agents.utils.logging_config import CustomFormatter

    formatter = CustomFormatter()

    # Create a mock log record
    record = logging.LogRecord(
        name="mystery_agents.agents.a2_world",
        level=logging.INFO,
        pathname="",
        lineno=0,
        msg="Test message",
        args=(),
        exc_info=None,
    )

    formatted = formatter.format(record)

    # Should contain [a2_world] not the full path
    assert "[a2_world]" in formatted
    assert "mystery_agents.agents" not in formatted
    assert "Test message" in formatted
    assert "INFO" in formatted
