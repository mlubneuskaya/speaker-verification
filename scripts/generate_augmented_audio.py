"""Generate perturbed audio variants for one test file.

The variants mirror the benchmark tasks used in the report:

* task2: amplitude scaling,
* task3: naive and interpolated downsampling,
* task4: additive Gaussian noise,
* task5: environmental background noise,
* task6: lossy codec compression,
* task7: reverberation.

By default the input is processed exactly like benchmark samples
(``preprocess_audio``: mono, 16 kHz, VAD) before perturbations are applied.
"""

from __future__ import annotations

import argparse
import logging
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Literal

import soundfile as sf
import torch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from benchmarks.augmentations import (
    add_environmental_noise,
    add_gaussian_noise,
    apply_codec_compression,
    apply_reverberation,
    interpolated_resample,
    naive_subsample,
    scale_amplitude,
)
from src.config import load_config
from src.preprocessing import load_audio, preprocess_audio, to_mono_16k
from src.utils import find_ffmpeg

logger = logging.getLogger(__name__)

Codec = Literal["mp3", "aac", "opus"]

CODEC_OUTPUT = {
    "mp3": ("libmp3lame", "mp3"),
    "aac": ("aac", "m4a"),
    "opus": ("libopus", "opus"),
}


def _safe_tag(value: float | int | str) -> str:
    return str(value).replace(".", "p").replace("-", "m")


def _write_wav(path: Path, waveform: torch.Tensor, sample_rate: int) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    audio = waveform.detach().cpu().squeeze(0).numpy()
    sf.write(path, audio, sample_rate)
    logger.info("Wrote %s", path)


def _encode_codec_file(
    waveform: torch.Tensor,
    sample_rate: int,
    output_path: Path,
    codec: Codec,
    bitrate_kbps: int,
) -> None:
    encoder, _ = CODEC_OUTPUT[codec]
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with tempfile.TemporaryDirectory() as tmpdir:
        input_path = Path(tmpdir) / "input.wav"
        sf.write(input_path, waveform.detach().cpu().squeeze(0).numpy(), sample_rate)
        subprocess.run(
            [
                find_ffmpeg(),
                "-y",
                "-i",
                str(input_path),
                "-c:a",
                encoder,
                "-b:a",
                f"{bitrate_kbps}k",
                str(output_path),
            ],
            check=True,
            capture_output=True,
        )
    logger.info("Wrote %s", output_path)


def _load_input(
    input_path: Path,
    sample_rate: int,
    use_vad: bool,
) -> torch.Tensor:
    if use_vad:
        return preprocess_audio(str(input_path), target_sr=sample_rate)

    waveform, orig_sr = load_audio(str(input_path))
    return to_mono_16k(waveform, orig_sr, target_sr=sample_rate)


def generate_variants(
    input_path: Path,
    output_dir: Path,
    cfg: dict,
    use_vad: bool,
    save_codec_containers: bool,
    seed: int,
) -> list[Path]:
    sample_rate = int(cfg["audio"]["sample_rate"])
    bm_cfg = cfg["benchmarks"]
    paths = cfg["paths"]

    waveform = _load_input(input_path, sample_rate, use_vad=use_vad)
    output_dir.mkdir(parents=True, exist_ok=True)

    written: list[Path] = []

    baseline_path = output_dir / "task1_baseline_preprocessed.wav"
    _write_wav(baseline_path, waveform, sample_rate)
    written.append(baseline_path)

    for factor in bm_cfg["task2"]["amplitude_factors"]:
        out = output_dir / f"task2_amplitude_{_safe_tag(factor)}.wav"
        _write_wav(out, scale_amplitude(waveform, float(factor)), sample_rate)
        written.append(out)

    for step in bm_cfg["task3"]["naive_steps"]:
        out = output_dir / f"task3_naive_step{step}.wav"
        _write_wav(out, naive_subsample(waveform, int(step), sample_rate), sample_rate)
        written.append(out)

    for factor in bm_cfg["task3"]["interp_factors"]:
        out = output_dir / f"task3_interp_factor{factor}.wav"
        augmented = interpolated_resample(waveform, sample_rate, int(factor), sample_rate)
        _write_wav(out, augmented, sample_rate)
        written.append(out)

    for snr in bm_cfg["task4"]["snr_levels_db"]:
        out = output_dir / f"task4_gaussian_snr{_safe_tag(snr)}dB.wav"
        augmented = add_gaussian_noise(waveform, float(snr))
        _write_wav(out, augmented, sample_rate)
        written.append(out)

    noise_dir = Path(paths["urban_sound"])
    if noise_dir.exists():
        for snr in bm_cfg["task5"]["snr_levels_db"]:
            out = output_dir / f"task5_env_snr{_safe_tag(snr)}dB.wav"
            augmented = add_environmental_noise(
                waveform,
                str(noise_dir),
                float(snr),
                sample_rate=sample_rate,
                seed=seed,
            )
            _write_wav(out, augmented, sample_rate)
            written.append(out)
    else:
        logger.warning("Skipping task5: noise directory not found: %s", noise_dir)

    for codec, bitrates in bm_cfg["task6"]["codecs"].items():
        codec = codec.lower()
        if codec not in CODEC_OUTPUT:
            logger.warning("Skipping unsupported codec from config: %s", codec)
            continue
        _, ext = CODEC_OUTPUT[codec]
        for bitrate in bitrates:
            decoded_out = output_dir / f"task6_{codec}_{bitrate}kbps_decoded.wav"
            decoded = apply_codec_compression(
                waveform,
                sample_rate,
                codec,  # type: ignore[arg-type]
                int(bitrate),
            )
            _write_wav(decoded_out, decoded, sample_rate)
            written.append(decoded_out)

            if save_codec_containers:
                encoded_out = output_dir / f"task6_{codec}_{bitrate}kbps.{ext}"
                _encode_codec_file(
                    waveform,
                    sample_rate,
                    encoded_out,
                    codec,  # type: ignore[arg-type]
                    int(bitrate),
                )
                written.append(encoded_out)

    rir_dir = Path(paths["rir_data"])
    if rir_dir.exists():
        out = output_dir / "task7_reverberation.wav"
        augmented = apply_reverberation(waveform, str(rir_dir), sample_rate, seed=seed)
        _write_wav(out, augmented, sample_rate)
        written.append(out)
    else:
        logger.warning("Skipping task7: RIR directory not found: %s", rir_dir)

    return written


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input", type=Path, help="Input audio file.")
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("config.yaml"),
        help="Project config with benchmark parameters.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Output directory. Default: reports/augmented_audio/<input-stem>.",
    )
    parser.add_argument(
        "--no-vad",
        action="store_true",
        help="Keep the full file after mono/16 kHz conversion instead of using benchmark VAD.",
    )
    parser.add_argument(
        "--no-codec-containers",
        action="store_true",
        help="Only save decoded WAV artifacts for codecs, not MP3/M4A/Opus files.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=0,
        help="Seed used for background noise and RIR selection.",
    )
    parser.add_argument(
        "--log-level",
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    logging.basicConfig(
        level=getattr(logging, args.log_level),
        format="%(levelname)s | %(message)s",
    )

    if not args.input.exists():
        raise FileNotFoundError(f"Input file not found: {args.input}")

    cfg = load_config(str(args.config))
    output_dir = args.output_dir
    if output_dir is None:
        output_dir = Path(cfg["paths"]["reports"]) / "augmented_audio" / args.input.stem

    written = generate_variants(
        input_path=args.input,
        output_dir=output_dir,
        cfg=cfg,
        use_vad=not args.no_vad,
        save_codec_containers=not args.no_codec_containers,
        seed=args.seed,
    )
    print(f"Generated {len(written)} files in {output_dir}")


if __name__ == "__main__":
    main()
