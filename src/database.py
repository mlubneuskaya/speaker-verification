"""JSON-based speaker database for storing master speaker templates."""

import json
import logging
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np

logger = logging.getLogger(__name__)


class SpeakerDatabase:
    """Simple JSON-backed store for enrolled speaker templates.

    Parameters
    ----------
    db_path : str
        Path to the JSON file used as the database.
    """

    def __init__(self, db_path: str) -> None:
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._data: dict[str, Any] = self._load()

    def _load(self) -> dict[str, Any]:
        """Load the database from disk, returning an empty dict if absent."""
        if not self.db_path.exists():
            logger.info(f"No existing database at {self.db_path}; starting fresh")
            return {}
        try:
            with open(self.db_path, "r") as f:
                data = json.load(f)
            logger.info(f"Loaded {len(data)} speaker(s) from {self.db_path}")
            return data
        except (json.JSONDecodeError, OSError) as exc:
            logger.error(f"Failed to read database: {exc}")
            raise

    def _save(self) -> None:
        """Persist the in-memory database to disk."""
        try:
            with open(self.db_path, "w") as f:
                json.dump(self._data, f, indent=2)
            logger.debug(f"Database saved to {self.db_path}")
        except OSError as exc:
            logger.error(f"Failed to write database: {exc}")
            raise

    def add_speaker(
        self,
        user_id: str,
        name: str,
        embedding: np.ndarray,
        enrolled_videos: list[str] | None = None,
    ) -> None:
        """Enroll a new speaker or overwrite an existing record.

        Parameters
        ----------
        user_id : str
            Unique identifier for the speaker.
        name : str
            Human-readable display name.
        embedding : np.ndarray
            L2-normalized master speaker template of shape ``(emb_dim,)``.
        enrolled_videos : list[str] or None
            Video IDs (directory names) used to build the template. Stored so
            benchmarks can exclude these clips from test pairs.
        """
        self._data[user_id] = {
            "name": name,
            "embedding": embedding.tolist(),
            "metadata": {
                "enrollment_date": datetime.now().isoformat(),
                "enrolled_videos": enrolled_videos or [],
            },
        }
        self._save()
        logger.info(f"Enrolled speaker '{name}' (id={user_id})")

    def get_speaker(self, user_id: str) -> dict[str, Any]:
        """Retrieve a speaker record by ID.

        Parameters
        ----------
        user_id : str
            Speaker identifier.

        Returns
        -------
        dict[str, Any]
            Record with keys ``name``, ``embedding``, ``metadata``.

        Raises
        ------
        KeyError
            If ``user_id`` is not in the database.
        """
        if user_id not in self._data:
            raise KeyError(f"Speaker '{user_id}' not found in database")
        record = self._data[user_id].copy()
        record["embedding"] = np.array(record["embedding"], dtype=np.float32)
        return record

    def list_speakers(self) -> list[str]:
        """Return all enrolled speaker IDs.

        Returns
        -------
        list[str]
            List of user IDs.
        """
        return list(self._data.keys())

    def get_all_embeddings(self) -> dict[str, np.ndarray]:
        """Return a mapping of user_id -> embedding for every enrolled speaker.

        Returns
        -------
        dict[str, np.ndarray]
            Each value is an L2-normalized embedding vector.
        """
        return {
            uid: np.array(record["embedding"], dtype=np.float32)
            for uid, record in self._data.items()
        }

    def delete_speaker(self, user_id: str) -> None:
        """Remove a speaker from the database.

        Parameters
        ----------
        user_id : str
            Speaker identifier to remove.
        """
        if user_id not in self._data:
            raise KeyError(f"Speaker '{user_id}' not found")
        name = self._data[user_id]["name"]
        del self._data[user_id]
        self._save()
        logger.info(f"Deleted speaker '{name}' (id={user_id})")

    def get_enrollment_video_map(self) -> dict[str, list[str]]:
        """Return a mapping of speaker_id -> list of enrolled video IDs.

        Used by the benchmark to exclude enrollment clips from test pairs.

        Returns
        -------
        dict[str, list[str]]
            Keys are speaker IDs; values are video ID lists (may be empty for
            speakers enrolled via file upload rather than VoxCeleb).
        """
        return {
            uid: record.get("metadata", {}).get("enrolled_videos", [])
            for uid, record in self._data.items()
        }

    def __len__(self) -> int:
        return len(self._data)
