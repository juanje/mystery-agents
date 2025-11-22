"""Logging configuration and AgentLogger wrapper for mystery-agents."""

from __future__ import annotations

import logging
import sys
from typing import TYPE_CHECKING

import click

if TYPE_CHECKING:
    from mystery_agents.models.state import GameState


class CustomFormatter(logging.Formatter):
    """Custom formatter with [agent_name] context."""

    def format(self, record: logging.LogRecord) -> str:
        """Format log record with agent context."""
        # Extract agent name from logger name (e.g., mystery_agents.agents.a2_world -> a2_world)
        parts = record.name.split(".")
        agent_name = parts[-1] if parts else record.name

        # Create formatted message with [agent_name] context
        timestamp = self.formatTime(record, "%Y-%m-%d %H:%M:%S")
        level = record.levelname
        message = record.getMessage()

        return f"{timestamp} {level} [{agent_name}] {message}"


def setup_logging(verbosity: int, quiet: bool, log_file: str | None = None) -> None:
    """
    Configure logging system based on verbosity level.

    Args:
        verbosity: Logging level (0=default/no logs, 1=INFO, 2=DEBUG)
        quiet: If True, suppress all logging output to console
        log_file: Optional file path to write logs to (always writes INFO+ logs if specified)
    """
    # Get root logger
    root_logger = logging.getLogger()

    # Clear any existing handlers
    root_logger.handlers.clear()

    # Set root logger level based on verbosity and log_file
    # If log_file is specified, we need at least INFO level for the file handler
    if log_file:
        if verbosity >= 2:
            root_logger.setLevel(logging.DEBUG)
        else:
            root_logger.setLevel(logging.INFO)
    elif verbosity == 0:
        # Default mode - no logging to console (only visual progress)
        root_logger.setLevel(logging.WARNING)
    elif verbosity == 1:
        # -v: INFO level
        root_logger.setLevel(logging.INFO)
    else:
        # -vv: DEBUG level
        root_logger.setLevel(logging.DEBUG)

    formatter = CustomFormatter()

    # Console handler (stderr, so it doesn't mix with stdout progress)
    if not quiet and verbosity > 0:
        console_handler = logging.StreamHandler(sys.stderr)
        console_handler.setLevel(logging.DEBUG)
        console_handler.setFormatter(formatter)
        root_logger.addHandler(console_handler)

    # File handler (if specified)
    # Always writes INFO level logs by default, or DEBUG if -vv is used
    if log_file:
        file_handler = logging.FileHandler(log_file, mode="a", encoding="utf-8")
        if verbosity >= 2:
            file_handler.setLevel(logging.DEBUG)
        else:
            file_handler.setLevel(logging.INFO)
        file_handler.setFormatter(formatter)
        root_logger.addHandler(file_handler)

    # Silence noisy third-party loggers
    # weasyprint logs many warnings about unsupported CSS properties
    # Only show them in DEBUG mode (-vv)
    import warnings

    for logger_name in ["weasyprint", "fontTools", "PIL", "weasyprint.css", "weasyprint.html"]:
        third_party_logger = logging.getLogger(logger_name)
        if verbosity < 2:
            third_party_logger.setLevel(logging.ERROR)
        third_party_logger.propagate = True

    # Also silence warnings from these modules
    if verbosity < 2:
        warnings.filterwarnings("ignore", module="weasyprint")
        warnings.filterwarnings("ignore", module="fontTools")
        warnings.filterwarnings("ignore", module="PIL")


class AgentLogger:
    """
    Unified logging interface that adapts output based on verbosity mode.

    This wrapper encapsulates all verbosity logic so agents can use a clean API
    without conditionals. Depending on the mode:
    - Default (verbosity=0): Shows visual progress messages via click.echo
    - Verbose (-v, -vv): Shows structured logs via Python logging
    - Quiet (--quiet): Suppresses all output
    """

    def __init__(self, name: str, state: GameState):
        """
        Initialize AgentLogger.

        Args:
            name: Logger name (typically __name__ from calling module)
            state: Current game state (for accessing config)
        """
        self.name = name
        self.state = state
        self.logger = logging.getLogger(name)

    def info(self, message: str) -> None:
        """
        Log info-level message.

        In default mode: Shows as visual progress with click.echo
        In verbose mode: Shows as structured INFO log
        In quiet mode: Suppressed from console
        With log_file: Always written to file regardless of verbosity

        Args:
            message: Message to log
        """
        # Determine if we should use structured logging
        # Use logger if: verbose mode OR log_file is configured
        use_logger = self.state.config.verbosity > 0 or self.state.config.log_file

        if use_logger:
            # Structured log (goes to console if verbose, and/or to file if configured)
            self.logger.info(message)

        # Console output for default mode (only if not using logger for console)
        if not self.state.config.quiet_mode and self.state.config.verbosity == 0:
            # Default mode: visual progress with click.echo
            click.echo(message)

    def debug(self, message: str) -> None:
        """
        Log debug-level message.

        Only shown in -vv mode (verbosity >= 2).

        Args:
            message: Message to log
        """
        if self.state.config.verbosity >= 2:
            self.logger.debug(message)

    def warning(self, message: str) -> None:
        """
        Log warning message.

        Always shown (unless quiet mode).

        Args:
            message: Message to log
        """
        if not self.state.config.quiet_mode:
            self.logger.warning(message)

    def error(self, message: str) -> None:
        """
        Log error message.

        Always shown (even in quiet mode).

        Args:
            message: Message to log
        """
        self.logger.error(message)
