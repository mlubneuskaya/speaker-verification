"""Audio preprocessing: loading, resampling, mono conversion, and VAD."""

import logging
import subprocess
import tempfile
from pathlib import Path

import numpy as np
import soundfile as sf
import torch
import torchaudio

from src.utils import find_ffmpeg

logger = logging.getLogger(__name__)


def load_audio(path: str) -> tuple[torch.Tensor, int]:
    """Load an audio file from disk.

    Parameters
    ----------
    path : str
        Path to the audio file (WAV, FLAC, etc.).

    Returns
    -------
    tuple[torch.Tensor, int]
        Waveform tensor of shape ``(channels, time)`` and sample rate.
    """
    p = Path(path)
    try:
        if p.suffix.lower() == ".m4a":
            waveform, sample_rate = _load_m4a(str(p))
        else:
            waveform, sample_rate = _load_via_soundfile(str(p))
        logger.debug(
            f"Loaded audio: {path} | sr={sample_rate} | shape={waveform.shape}"
        )
        return waveform, sample_rate
    except Exception as exc:
        logger.error(f"Failed to load audio from {path}: {exc}")
        raise


def _load_via_soundfile(path: str) -> tuple[torch.Tensor, int]:
    """Load a WAV/FLAC/OGG file using soundfile, bypassing torchaudio backends.

    Parameters
    ----------
    path : str
        Path to the audio file.

    Returns
    -------
    tuple[torch.Tensor, int]
        Waveform of shape ``(channels, time)`` and sample rate.
    """
    data, sample_rate = sf.read(path, dtype="float32", always_2d=True)
    waveform = torch.from_numpy(data.T)  # (channels, time)
    return waveform, sample_rate


def _load_m4a(path: str) -> tuple[torch.Tensor, int]:
    """Decode an M4A file to a float32 tensor via ffmpeg subprocess.

    Avoids torchaudio's torchcodec/ffmpeg backend, which may have broken
    native libraries depending on the installation.

    Parameters
    ----------
    path : str
        Path to the M4A file.

    Returns
    -------
    tuple[torch.Tensor, int]
        Waveform of shape ``(channels, time)`` and sample rate.
    """
    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
        tmp_path = tmp.name

    try:
        subprocess.run(
            [find_ffmpeg(), "-y", "-i", path, "-vn", tmp_path],
            check=True,
            capture_output=True,
        )
        data, sample_rate = sf.read(tmp_path, dtype="float32")
    finally:
        Path(tmp_path).unlink(missing_ok=True)

    waveform = torch.from_numpy(data)
    if waveform.dim() == 1:
        waveform = waveform.unsqueeze(0)
    else:
        waveform = waveform.T  # (channels, time)
    return waveform, sample_rate


def to_mono_16k(
    waveform: torch.Tensor,
    orig_sr: int,
    target_sr: int = 16000,
) -> torch.Tensor:
    """Convert waveform to mono and resample to target sample rate.

    Parameters
    ----------
    waveform : torch.Tensor
        Input waveform of shape ``(channels, time)``.
    orig_sr : int
        Original sample rate of the waveform.
    target_sr : int
        Target sample rate. Defaults to 16000 Hz.

    Returns
    -------
    torch.Tensor
        Mono waveform of shape ``(1, time)`` at ``target_sr``.
    """
    if waveform.shape[0] > 1:
        waveform = waveform.mean(dim=0, keepdim=True)

    if orig_sr != target_sr:
        resampler = torchaudio.transforms.Resample(
            orig_freq=orig_sr, new_freq=target_sr
        )
        waveform = resampler(waveform)
        logger.debug(f"Resampled {orig_sr} -> {target_sr} Hz")

    return waveform


def apply_vad(
    waveform: torch.Tensor,
    sample_rate: int,
    energy_threshold: float = 0.005,
    frame_length_ms: float = 20.0,
) -> torch.Tensor:
    """Remove leading and trailing silence using energy-based VAD.

    Parameters
    ----------
    waveform : torch.Tensor
        Mono waveform of shape ``(1, time)``.
    sample_rate : int
        Sample rate of the waveform.
    energy_threshold : float
        Minimum frame RMS energy to be considered voiced.
    frame_length_ms : float
        Frame length in milliseconds for energy computation.

    Returns
    -------
    torch.Tensor
        Trimmed waveform of shape ``(1, time)``.
    """
    frame_len = int(sample_rate * frame_length_ms / 1000)
    signal = waveform.squeeze(0)

    n_complete = len(signal) // frame_len
    if n_complete == 0:
        return waveform

    frames = signal[: n_complete * frame_len].reshape(n_complete, frame_len)
    energy = frames.pow(2).mean(dim=1).sqrt()
    voiced = energy > energy_threshold

    if not voiced.any():
        logger.warning("VAD: no voiced frames detected, returning original waveform")
        return waveform

    first = int(voiced.nonzero(as_tuple=True)[0][0].item()) * frame_len
    last = int((voiced.nonzero(as_tuple=True)[0][-1].item() + 1)) * frame_len
    last = min(last, signal.shape[0])

    trimmed = waveform[:, first:last]
    logger.debug(f"VAD trimmed {waveform.shape[1]} -> {trimmed.shape[1]} samples")
    return trimmed


def preprocess_audio(
    path: str,
    target_sr: int = 16000,
    energy_threshold: float = 0.005,
    frame_length_ms: float = 20.0,
) -> torch.Tensor:
    """Full preprocessing pipeline: load -> mono+resample -> VAD.

    Parameters
    ----------
    path : str
        Path to the input audio file.
    target_sr : int
        Target sample rate in Hz.
    energy_threshold : float
        VAD energy threshold.
    frame_length_ms : float
        VAD frame length in milliseconds.

    Returns
    -------
    torch.Tensor
        Preprocessed mono waveform of shape ``(1, time)`` at ``target_sr``.
    """
    waveform, sr = load_audio(path)
    waveform = to_mono_16k(waveform, sr, target_sr)
    waveform = apply_vad(waveform, target_sr, energy_threshold, frame_length_ms)
    return waveform
