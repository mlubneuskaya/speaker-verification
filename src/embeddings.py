"""ECAPA-TDNN embedding extraction with L2 normalization."""

import logging
from typing import Callable

import numpy as np
import torch
from speechbrain.inference.speaker import SpeakerRecognition

logger = logging.getLogger(__name__)


def l2_normalize(embedding: np.ndarray) -> np.ndarray:
    """L2-normalize a 1-D embedding vector.

    Parameters
    ----------
    embedding : np.ndarray
        Raw embedding vector.

    Returns
    -------
    np.ndarray
        Unit-norm embedding vector.
    """
    norm = np.linalg.norm(embedding)
    if norm < 1e-10:
        logger.warning("Embedding norm is near zero; returning unnormalized vector")
        return embedding
    return embedding / norm


class EmbeddingExtractor:
    """Wraps SpeechBrain ECAPA-TDNN for single-file embedding extraction.

    Parameters
    ----------
    model_source : str
        HuggingFace model ID or local path.
    savedir : str
        Local directory for caching the pretrained model.
    """

    def __init__(self, model_source: str, savedir: str) -> None:
        logger.info(f"Loading speaker model from {model_source}")
        try:
            self._model = SpeakerRecognition.from_hparams(
                source=model_source,
                savedir=savedir,
            )
            logger.info("Speaker model loaded successfully")
        except Exception as exc:
            logger.error(f"Failed to load speaker model: {exc}")
            raise

    def extract(self, waveform: torch.Tensor) -> np.ndarray:
        """Extract an L2-normalized embedding from a preprocessed waveform.

        Parameters
        ----------
        waveform : torch.Tensor
            Mono waveform of shape ``(1, time)`` at 16 kHz.

        Returns
        -------
        np.ndarray
            L2-normalized embedding vector of shape ``(emb_dim,)``.
        """
        try:
            wav_lens = torch.ones(1)
            with torch.no_grad():
                emb = self._model.encode_batch(waveform, wav_lens)
            emb_np = emb.squeeze().cpu().numpy()
            return l2_normalize(emb_np)
        except Exception as exc:
            logger.error(f"Embedding extraction failed: {exc}")
            raise

    def extract_from_path(
        self,
        path: str,
        preprocess_fn: Callable = None,
    ) -> np.ndarray:
        """Load audio from disk and extract its embedding.

        Parameters
        ----------
        path : str
            Path to the audio file.
        preprocess_fn : callable or None
            Optional preprocessing function ``(path) -> torch.Tensor``.
            If None, uses ``src.preprocessing.preprocess_audio``.

        Returns
        -------
        np.ndarray
            L2-normalized embedding vector.
        """
        if preprocess_fn is None:
            from src.preprocessing import preprocess_audio

            preprocess_fn = preprocess_audio

        try:
            waveform = preprocess_fn(path)
        except Exception as exc:
            logger.error(f"Preprocessing failed for {path}: {exc}")
            raise

        return self.extract(waveform)
