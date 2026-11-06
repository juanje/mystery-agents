"""Shared helpers for tests."""

import contextlib
import tempfile
from collections.abc import Iterator
from pathlib import Path


@contextlib.contextmanager
def isolated_cwd(temp_dir: Path | None = None) -> Iterator[Path]:
    """
    Run the block inside a fresh, empty working directory.

    Replaces Click's CliRunner.isolated_filesystem(), deprecated in Click 8.5.
    """
    with tempfile.TemporaryDirectory(dir=temp_dir) as tmp, contextlib.chdir(tmp):
        yield Path(tmp)
