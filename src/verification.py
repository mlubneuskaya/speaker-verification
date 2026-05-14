"""Speaker verification via cosine similarity against enrolled templates."""

import logging

import numpy as np
import torch

from src.database import SpeakerDatabase
from src.embeddings import EmbeddingExtractor
from src.preprocessing import preprocess_audio

logger = logging.getLogger(__name__)


def cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    """Compute cosine similarity between two L2-normalized vectors.

    Since both inputs are assumed to be unit-norm, this reduces to the dot
    product and is therefore O(d) with no extra normalization overhead.

    Parameters
    ----------
    a : np.ndarray
        First L2-normalized embedding.
    b : np.ndarray
        Second L2-normalized embedding.

    Returns
    -------
    float
        Cosine similarity in ``[-1, 1]``.
    """
    return float(np.dot(a, b))


def verify_speaker(
    extractor: EmbeddingExtractor,
    db: SpeakerDatabase,
    audio_path: str,
    threshold: float = 0.25,
    target_sr: int = 16000,
    energy_threshold: float = 0.005,
    frame_length_ms: float = 20.0,
) -> tuple[str | None, float, bool]:
    """Identify the best-matching enrolled speaker for a probe recording.

    Parameters
    ----------
    extractor : EmbeddingExtractor
        Initialized ECAPA embedding extractor.
    db : SpeakerDatabase
        Initialized speaker database containing enrolled templates.
    audio_path : str
        Path to the probe audio file.
    threshold : float
        Cosine similarity threshold above which a match is declared.
    target_sr : int
        Target sample rate for preprocessing.
    energy_threshold : float
        VAD energy threshold.
    frame_length_ms : float
        VAD frame length in milliseconds.

    Returns
    -------
    tuple[str | None, float, bool]
        ``(best_match_id, best_score, is_match)`` where ``best_match_id`` is
        ``None`` if the database is empty.
    """
    all_embeddings = db.get_all_embeddings()

    if not all_embeddings:
        logger.warning("Database is empty; cannot verify against any speaker")
        return None, 0.0, False

    try:
        waveform = preprocess_audio(
            audio_path,
            target_sr=target_sr,
            energy_threshold=energy_threshold,
            frame_length_ms=frame_length_ms,
        )
        probe_emb = extractor.extract(waveform)
    except Exception as exc:
        logger.error(f"Failed to process probe audio {audio_path}: {exc}")
        raise

    best_id: str | None = None
    best_score: float = -2.0

    for uid, template in all_embeddings.items():
        score = cosine_similarity(probe_emb, template)
        logger.debug(f"Score vs '{uid}': {score:.4f}")
        if score > best_score:
            best_score = score
            best_id = uid

    is_match = best_score >= threshold
    logger.info(
        f"Verification result: id={best_id}, score={best_score:.4f}, "
        f"match={'YES' if is_match else 'NO'}"
    )
    return best_id, best_score, is_match


def verify_against_embedding(
    probe_emb: np.ndarray,
    template: np.ndarray,
    threshold: float = 0.25,
) -> tuple[float, bool]:
    """Compare a probe embedding directly against a stored template.

    Parameters
    ----------
    probe_emb : np.ndarray
        L2-normalized probe embedding.
    template : np.ndarray
        L2-normalized enrolled template.
    threshold : float
        Decision threshold.

    Returns
    -------
    tuple[float, bool]
        ``(score, is_match)``.
    """
    score = cosine_similarity(probe_emb, template)
    return score, score >= threshold
