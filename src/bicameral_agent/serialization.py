"""Parquet serialization for Episode objects.

Uses a JSON-in-Parquet approach: each episode is stored as a row with an
``episode_id`` column (for filtering) and a ``payload`` column containing
the full JSON representation. This guarantees perfect round-trip fidelity
for nested and variable-length structures.
"""

from __future__ import annotations

import pyarrow as pa
import pyarrow.parquet as pq

from bicameral_agent.schema import Episode


def episode_to_parquet(episode: Episode, path: str) -> None:
    """Write a single Episode to a Parquet file.

    Args:
        episode: The episode to serialize.
        path: File path to write the Parquet file to.
    """
    table = pa.table({
        "episode_id": [episode.episode_id],
        "payload": [episode.model_dump_json()],
    })
    pq.write_table(table, path)


def episode_from_parquet(path: str) -> Episode:
    """Read a single Episode from a Parquet file.

    Args:
        path: File path to read the Parquet file from.

    Returns:
        The deserialized Episode.

    Raises:
        ValueError: If the file contains no rows.
    """
    table = pq.read_table(path)
    if table.num_rows == 0:
        raise ValueError(f"Parquet file at {path} contains no episodes")
    payload = table.column("payload")[0].as_py()
    return Episode.model_validate_json(payload)


def episodes_to_parquet(episodes: list[Episode], path: str) -> None:
    """Write multiple Episodes to a Parquet file (one row per episode).

    Args:
        episodes: List of episodes to serialize.
        path: File path to write the Parquet file to.
    """
    table = pa.table({
        "episode_id": [ep.episode_id for ep in episodes],
        "payload": [ep.model_dump_json() for ep in episodes],
    })
    pq.write_table(table, path)


def episodes_from_parquet(path: str) -> list[Episode]:
    """Read multiple Episodes from a Parquet file.

    Args:
        path: File path to read the Parquet file from.

    Returns:
        List of deserialized Episodes.
    """
    table = pq.read_table(path)
    return [
        Episode.model_validate_json(row.as_py())
        for row in table.column("payload")
    ]
