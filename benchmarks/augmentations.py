"""Audio augmentation functions for the 7 benchmark tasks."""

import logging
import os
import random
import subprocess
import tempfile
from functools import lru_cache
from pathlib import Path
from typing import Literal

import numpy as np
import soundata
import soundfile as sf
import torch
import torchaudio

from src.utils import find_ffmpeg

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Task 2: Amplitude scaling
# ---------------------------------------------------------------------------


def scale_amplitude(waveform: torch.Tensor, factor: float) -> torch.Tensor:
    """Multiply waveform amplitude by a scalar factor.

    Parameters
    ----------
    waveform : torch.Tensor
        Input waveform of shape ``(1, time)``.
    factor : float
        Amplitude scaling factor.

    Returns
    -------
    torch.Tensor
        Scaled waveform, clipped to ``[-1, 1]``.
    """
    scaled = waveform * factor
    return torch.clamp(scaled, -1.0, 1.0)


# ---------------------------------------------------------------------------
# Task 3: Resampling
# ---------------------------------------------------------------------------


def naive_subsample(
    waveform: torch.Tensor,
    step: int,
    orig_sr: int,
    target_sr: int = 16000,
) -> torch.Tensor:
    """Decimate by keeping every ``step``-th sample, then upsample back.

    No anti-aliasing filter is applied, intentionally introducing aliasing
    artefacts to simulate naive subsampling.

    Parameters
    ----------
    waveform : torch.Tensor
        Input waveform of shape ``(1, time)``.
    step : int
        Decimation factor (keep every ``step``-th sample).
    orig_sr : int
        Sample rate of the input waveform before decimation.
    target_sr : int
        Sample rate to restore after upsampling.

    Returns
    -------
    torch.Tensor
        Waveform resampled back to ``target_sr``.
    """
    decimated = waveform[:, ::step]
    decimated_sr = orig_sr // step
    resampler = torchaudio.transforms.Resample(
        orig_freq=decimated_sr, new_freq=target_sr
    )
    return resampler(decimated)


def interpolated_resample(
    waveform: torch.Tensor,
    orig_sr: int,
    factor: int,
    target_sr: int = 16000,
) -> torch.Tensor:
    """Downsample with anti-aliasing then upsample back to ``target_sr``.

    Parameters
    ----------
    waveform : torch.Tensor
        Input waveform of shape ``(1, time)``.
    orig_sr : int
        Original sample rate of the waveform.
    factor : int
        Downsampling factor.
    target_sr : int
        Target sample rate to restore after downsampling.

    Returns
    -------
    torch.Tensor
        Waveform resampled back to ``target_sr``.
    """
    downsampled_sr = orig_sr // factor
    down = torchaudio.transforms.Resample(orig_freq=orig_sr, new_freq=downsampled_sr)
    up = torchaudio.transforms.Resample(orig_freq=downsampled_sr, new_freq=target_sr)
    return up(down(waveform))


# ---------------------------------------------------------------------------
# Task 4: Gaussian noise
# ---------------------------------------------------------------------------


def add_gaussian_noise(waveform: torch.Tensor, snr_db: float) -> torch.Tensor:
    """Add zero-mean Gaussian noise at a specified SNR level.

    Parameters
    ----------
    waveform : torch.Tensor
        Input waveform of shape ``(1, time)``.
    snr_db : float
        Signal-to-noise ratio in decibels.

    Returns
    -------
    torch.Tensor
        Noisy waveform, clipped to ``[-1, 1]``.
    """
    signal_power = waveform.pow(2).mean()
    snr_linear = 10.0 ** (snr_db / 10.0)
    noise_power = signal_power / snr_linear
    noise = torch.randn_like(waveform) * noise_power.sqrt()
    return torch.clamp(waveform + noise, -1.0, 1.0)


# ---------------------------------------------------------------------------
# Task 5: Environmental noise (UrbanSound8K via soundata)
# ---------------------------------------------------------------------------


@lru_cache(maxsize=1)
def _get_urbansound8k(data_home: str):
    """Return a cached soundata UrbanSound8K dataset object.

    Parameters
    ----------
    data_home : str
        Parent directory of the ``UrbanSound8K`` folder.
    """
    return soundata.initialize("urbansound8k", data_home=data_home)


def _load_random_noise_clip(
    noise_dir: str,
    target_length: int,
    sample_rate: int,
    rng: random.Random,
) -> torch.Tensor:
    """Load a random UrbanSound8K clip via soundata and fit to ``target_length``.

    Parameters
    ----------
    noise_dir : str
        Path to the ``UrbanSound8K`` dataset root directory.
    target_length : int
        Required number of samples.
    sample_rate : int
        Target sample rate for the noise clip.
    rng : random.Random
        Random number generator instance.

    Returns
    -------
    torch.Tensor
        Noise waveform of shape ``(1, target_length)``.
    """
    data_home = str(Path(noise_dir).parent)
    dataset = _get_urbansound8k(data_home)

    clip_ids = dataset.clip_ids
    clip_id = rng.choice(clip_ids)
    clip = dataset.clip(clip_id)

    audio_data, noise_sr = clip.audio  # numpy array, shape (n_samples,) or (ch, n_samples)
    if audio_data.ndim == 1:
        noise_wav = torch.from_numpy(audio_data).unsqueeze(0).float()
    else:
        noise_wav = torch.from_numpy(audio_data).float()
        if noise_wav.shape[0] > 1:
            noise_wav = noise_wav.mean(dim=0, keepdim=True)

    if noise_sr != sample_rate:
        noise_wav = torchaudio.transforms.Resample(noise_sr, sample_rate)(noise_wav)

    while noise_wav.shape[1] < target_length:
        noise_wav = torch.cat([noise_wav, noise_wav], dim=1)
    return noise_wav[:, :target_length]


def add_environmental_noise(
    waveform: torch.Tensor,
    noise_dir: str,
    snr_db: float,
    sample_rate: int = 16000,
    seed: int = 0,
) -> torch.Tensor:
    """Mix a waveform with environmental background noise at a target SNR.

    Parameters
    ----------
    waveform : torch.Tensor
        Input waveform of shape ``(1, time)``.
    noise_dir : str
        Directory containing background noise WAV files.
    snr_db : float
        Target signal-to-noise ratio in decibels.
    sample_rate : int
        Sample rate of the waveform and noise.
    seed : int
        Seed for selecting a random noise clip.

    Returns
    -------
    torch.Tensor
        Mixed waveform, clipped to ``[-1, 1]``.
    """
    rng = random.Random(seed)
    noise = _load_random_noise_clip(noise_dir, waveform.shape[1], sample_rate, rng)
    noise = noise.to(waveform.device)
    signal_power = waveform.pow(2).mean()
    noise_power = noise.pow(2).mean()

    if noise_power < 1e-10:
        logger.warning("Noise clip has near-zero power; returning original waveform")
        return waveform

    snr_linear = 10.0 ** (snr_db / 10.0)
    scale = (signal_power / (snr_linear * noise_power)).sqrt()
    mixed = waveform + scale * noise
    return torch.clamp(mixed, -1.0, 1.0)


# ---------------------------------------------------------------------------
# Task 6: Lossy codec compression
# ---------------------------------------------------------------------------

_CODEC_MAP = {
    "mp3": ("libmp3lame", "mp3"),
    "aac": ("aac", "m4a"),
    "opus": ("libopus", "opus"),
}


def apply_codec_compression(
    waveform: torch.Tensor,
    sample_rate: int,
    codec: Literal["mp3", "aac", "opus"],
    bitrate_kbps: int,
) -> torch.Tensor:
    """Encode a waveform with a lossy codec then decode it back to PCM.

    Requires ``ffmpeg`` to be installed and on the system ``PATH``.

    Parameters
    ----------
    waveform : torch.Tensor
        Input waveform of shape ``(1, time)``.
    sample_rate : int
        Sample rate of the waveform.
    codec : {"mp3", "aac", "opus"}
        Codec to use for compression.
    bitrate_kbps : int
        Target encoding bitrate in kbps.

    Returns
    -------
    torch.Tensor
        Decoded waveform of shape ``(1, time)``.
    """
    if codec not in _CODEC_MAP:
        raise ValueError(f"Unsupported codec '{codec}'. Choose from {list(_CODEC_MAP)}")

    encoder, ext = _CODEC_MAP[codec]

    with tempfile.TemporaryDirectory() as tmpdir:
        input_path = os.path.join(tmpdir, "input.wav")
        compressed_path = os.path.join(tmpdir, f"compressed.{ext}")
        output_path = os.path.join(tmpdir, "output.wav")

        original_device = waveform.device
        audio_np = waveform.squeeze(0).cpu().numpy()
        sf.write(input_path, audio_np, sample_rate)

        try:
            subprocess.run(
                [
                    find_ffmpeg(),
                    "-y",
                    "-i",
                    input_path,
                    "-c:a",
                    encoder,
                    "-b:a",
                    f"{bitrate_kbps}k",
                    compressed_path,
                ],
                check=True,
                capture_output=True,
            )
            subprocess.run(
                [
                    find_ffmpeg(),
                    "-y",
                    "-i",
                    compressed_path,
                    "-ar",
                    str(sample_rate),
                    "-ac",
                    "1",
                    output_path,
                ],
                check=True,
                capture_output=True,
            )
        except subprocess.CalledProcessError as exc:
            logger.error(f"ffmpeg failed for codec={codec}: {exc.stderr.decode()}")
            raise

        data, _ = sf.read(output_path, dtype="float32")
        decoded = torch.tensor(data, dtype=torch.float32).unsqueeze(0)
        # Crop to original length to remove encoder delay/padding
        decoded = decoded[:, : waveform.shape[1]]
        return decoded.to(original_device)


# ---------------------------------------------------------------------------
# Task 7: Reverberation
# ---------------------------------------------------------------------------


def apply_reverberation(
    waveform: torch.Tensor,
    rir_dir: str,
    sample_rate: int = 16000,
    seed: int = 0,
) -> torch.Tensor:
    """Convolve a waveform with a random Room Impulse Response (RIR).

    Parameters
    ----------
    waveform : torch.Tensor
        Input waveform of shape ``(1, time)``.
    rir_dir : str
        Directory containing RIR WAV files (e.g. from OpenSLR SLR28).
    sample_rate : int
        Expected sample rate. RIR files will be resampled if needed.
    seed : int
        Seed for selecting a random RIR.

    Returns
    -------
    torch.Tensor
        Reverberant waveform of shape ``(1, time)``, same length as input.
    """
    rng = random.Random(seed)
    rir_files = list(Path(rir_dir).rglob("*.wav"))
    if not rir_files:
        raise FileNotFoundError(f"No RIR WAV files found in {rir_dir}")

    rir_path = rng.choice(rir_files)
    try:
        data, rir_sr = sf.read(str(rir_path), dtype="float32", always_2d=True)
        rir = torch.from_numpy(data.T)  # (channels, time)
    except Exception as exc:
        logger.error(f"Failed to load RIR from {rir_path}: {exc}")
        raise

    if rir.shape[0] > 1:
        rir = rir.mean(dim=0, keepdim=True)
    if rir_sr != sample_rate:
        rir = torchaudio.transforms.Resample(rir_sr, sample_rate)(rir)

    rir = rir.to(waveform.device)
    rir = rir / (rir.abs().max() + 1e-10)

    reverberant = torchaudio.functional.fftconvolve(waveform, rir)

    # Align by removing pre-delay: start from the direct-path impulse peak
    direct_path_idx = int(rir.abs().argmax(dim=-1).item())
    reverberant = reverberant[:, direct_path_idx : direct_path_idx + waveform.shape[1]]

    peak = reverberant.abs().max()
    if peak > 1e-10:
        reverberant = reverberant / peak * waveform.abs().max()

    return reverberant
