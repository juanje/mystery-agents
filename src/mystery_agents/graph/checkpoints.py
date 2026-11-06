"""Persistent checkpoints so an interrupted generation can be resumed."""

import inspect
import sqlite3
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path

from langgraph.checkpoint.serde.jsonplus import JsonPlusSerializer
from langgraph.checkpoint.sqlite import SqliteSaver
from pydantic import BaseModel

from mystery_agents.models import state as state_models

CHECKPOINTS_FILENAME = ".checkpoints.sqlite"


def _state_model_classes() -> list[type[BaseModel]]:
    """Pydantic models that make up GameState, the only types stored in checkpoints."""
    return [
        obj
        for _, obj in inspect.getmembers(state_models, inspect.isclass)
        if issubclass(obj, BaseModel) and obj.__module__ == state_models.__name__
    ]


def make_serializer() -> JsonPlusSerializer:
    """
    Serializer restricted to our state models.

    An explicit allowlist keeps deserialization strict without relying on the
    LANGGRAPH_STRICT_MSGPACK environment variable or the deprecated defaults.
    """
    return JsonPlusSerializer(allowed_msgpack_modules=_state_model_classes())


def checkpoints_path(output_dir: str | Path) -> Path:
    """Location of the checkpoint database inside the output directory."""
    return Path(output_dir) / CHECKPOINTS_FILENAME


@contextmanager
def open_checkpointer(output_dir: str | Path) -> Iterator[SqliteSaver]:
    """
    Open the SQLite checkpointer stored in the output directory.

    Args:
        output_dir: Base output directory for generated games

    Yields:
        Checkpointer ready to pass to create_workflow()
    """
    path = checkpoints_path(output_dir)
    path.parent.mkdir(parents=True, exist_ok=True)
    # LangGraph may write checkpoints from worker threads
    conn = sqlite3.connect(path, check_same_thread=False)
    try:
        yield SqliteSaver(conn, serde=make_serializer())
    finally:
        conn.close()
