"""Speaker enrollment: derive a master template from multiple audio samples."""

import logging
import time
from pathlib import Path

import numpy as np

from src.database import SpeakerDatabase
from src.embeddings import EmbeddingExtractor, l2_normalize
from src.preprocessing import preprocess_audio

logger = logging.getLogger(__name__)


def enroll_speaker(
    extractor: EmbeddingExtractor,
    db: SpeakerDatabase,
    user_id: str,
    name: str,
    audio_paths: list[str],
    target_sr: int = 16000,
    energy_threshold: float = 0.005,
    frame_length_ms: float = 20.0,
) -> np.ndarray:
    """Enroll a speaker by computing a master template from multiple recordings.

    Three (or more) audio samples are preprocessed, embedded, averaged, and
    L2-normalized to produce a single canonical speaker vector which is stored
    in the database.

    Parameters
    ----------
    extractor : EmbeddingExtractor
        Initialized ECAPA embedding extractor.
    db : SpeakerDatabase
        Initialized speaker database.
    user_id : str
        Unique identifier for the speaker.
    name : str
        Human-readable display name.
    audio_paths : list[str]
        Paths to enrollment audio recordings (minimum 3).
    target_sr : int
        Target sample rate for preprocessing.
    energy_threshold : float
        VAD energy threshold.
    frame_length_ms : float
        VAD frame length in milliseconds.

    Returns
    -------
    np.ndarray
        The L2-normalized master speaker template of shape ``(emb_dim,)``.

    Raises
    ------
    ValueError
        If fewer than 3 audio samples are provided.
    """
    if len(audio_paths) < 3:
        raise ValueError(
            f"Enrollment requires at least 3 samples; got {len(audio_paths)}"
        )

    t_total_start = time.perf_counter()
    embeddings: list[np.ndarray] = []

    for idx, path in enumerate(audio_paths):
        t_sample_start = time.perf_counter()
        logger.info(
            f"Processing enrollment sample {idx + 1}/{len(audio_paths)}: {path}"
        )
        try:
            waveform = preprocess_audio(
                path,
                target_sr=target_sr,
                energy_threshold=energy_threshold,
                frame_length_ms=frame_length_ms,
            )
            emb = extractor.extract(waveform)
            embeddings.append(emb)
        except Exception as exc:
            logger.error(f"Failed to process sample {path}: {exc}")
            raise
        elapsed = time.perf_counter() - t_sample_start
        logger.info(f"  Sample {idx + 1} processed in {elapsed:.2f}s")

    mean_embedding = np.mean(
        np.stack(embeddings, axis=0), axis=0
    )  # TODO a better pooling?
    master_template = l2_normalize(mean_embedding)

    # Store the video directory names so benchmarks can exclude these clips.
    enrolled_videos = list({Path(p).parent.name for p in audio_paths})

    db.add_speaker(
        user_id=user_id,
        name=name,
        embedding=master_template,
        enrolled_videos=enrolled_videos,
    )
    total_elapsed = time.perf_counter() - t_total_start
    logger.info(
        f"Master template created for '{name}' from {len(embeddings)} samples "
        f"in {total_elapsed:.2f}s ({total_elapsed / len(embeddings):.2f}s/sample)"
    )
    return master_template, total_elapsed
