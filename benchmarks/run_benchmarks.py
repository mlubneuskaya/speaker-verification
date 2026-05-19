"""Orchestrate all benchmark tasks (Task 0 calibration + Tasks 1-7) and save reports."""

import argparse
import json
import logging
from pathlib import Path
from typing import Any, Callable

import numpy as np
import pandas as pd

from benchmarks.augmentations import (
    add_environmental_noise,
    add_gaussian_noise,
    apply_codec_compression,
    apply_reverberation,
    interpolated_resample,
    naive_subsample,
    scale_amplitude,
)
from benchmarks.dataset import AudioPair, build_test_pairs, collect_speakers
from benchmarks.metrics import compute_eer, evaluate
from src.config import load_config
from src.database import SpeakerDatabase
from src.embeddings import EmbeddingExtractor
from src.preprocessing import preprocess_audio

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
)
logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _stratified_subset(pairs: list[AudioPair], n: int) -> list[AudioPair]:
    """Return a balanced subset of *n* pairs with equal genuine/impostor counts.

    Parameters
    ----------
    pairs : list[AudioPair]
        Full shuffled pair list.
    n : int
        Total number of pairs to return (split evenly between classes).

    Returns
    -------
    list[AudioPair]
        Subset with ``n // 2`` genuine and ``n // 2`` impostor pairs.
    """
    genuine = [p for p in pairs if p.label == 1]
    impostor = [p for p in pairs if p.label == 0]
    half = n // 2
    subset = genuine[:half] + impostor[:half]
    if len(subset) < n:
        logger.warning(
            f"Stratified subset requested {n} pairs but only {len(subset)} available"
        )
    return subset


def _extract_pair_scores(
    extractor: EmbeddingExtractor,
    pairs: list[AudioPair],
    augment_fn: Callable = None,
    sample_rate: int = 16000,
    baseline_scores: dict[tuple[str, str], float] | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, list[tuple[str, str]]]:
    """Extract embeddings for each pair and compute cosine similarity scores.

    ``augment_fn``, when provided, must accept ``(waveform, seed: int)`` so
    that each pair receives a unique seed, preventing both clips in a pair from
    receiving identical stochastic augmentation.

    Parameters
    ----------
    extractor : EmbeddingExtractor
        Initialized model.
    pairs : list[AudioPair]
        Test pairs with ground-truth labels.
    augment_fn : callable or None
        Optional ``(waveform, seed) -> waveform`` augmentation.
    sample_rate : int
        Target sample rate for preprocessing.
    baseline_scores : dict or None
        Mapping of ``(path1, path2) -> clean_score`` from Task 1. When
        provided, per-pair deltas (clean − degraded) are computed.

    Returns
    -------
    tuple[np.ndarray, np.ndarray, np.ndarray, list[tuple[str, str]]]
        ``(scores, labels, deltas, processed_keys)`` where ``deltas`` contains
        ``NaN`` for any pair absent from ``baseline_scores``.
    """
    scores: list[float] = []
    labels: list[int] = []
    deltas: list[float] = []
    processed_keys: list[tuple[str, str]] = []

    for idx, pair in enumerate(pairs):
        try:
            wav1 = preprocess_audio(pair.path1, target_sr=sample_rate)
            wav2 = preprocess_audio(pair.path2, target_sr=sample_rate)

            if augment_fn is not None:
                wav1 = augment_fn(wav1, idx * 2)
                wav2 = augment_fn(wav2, idx * 2 + 1)

            emb1 = extractor.extract(wav1)
            emb2 = extractor.extract(wav2)
            score = float(np.dot(emb1, emb2))

            key = (pair.path1, pair.path2)
            scores.append(score)
            labels.append(pair.label)
            processed_keys.append(key)

            if baseline_scores is not None:
                baseline = baseline_scores.get(key, float("nan"))
                deltas.append(baseline - score)
            else:
                deltas.append(float("nan"))

            if (idx + 1) % 50 == 0:
                logger.info(f"  Processed {idx + 1}/{len(pairs)} pairs")

        except (OSError, RuntimeError, ValueError) as exc:
            logger.warning(f"Skipping pair {pair.path1} / {pair.path2}: {exc}")

    return (
        np.array(scores, dtype=np.float32),
        np.array(labels, dtype=int),
        np.array(deltas, dtype=np.float32),
        processed_keys,
    )


def _save_results(
    results: dict[str, Any],
    scores: np.ndarray,
    labels: np.ndarray,
    output_dir: Path,
    task_name: str,
    deltas: np.ndarray | None = None,
) -> None:
    """Persist per-pair scores and summary metrics to CSV files.

    Parameters
    ----------
    results : dict[str, Any]
        Metric dict from ``evaluate()``.
    scores : np.ndarray
        Per-pair similarity scores.
    labels : np.ndarray
        Per-pair ground-truth labels.
    output_dir : Path
        Directory to write reports into.
    task_name : str
        Prefix for output file names.
    deltas : np.ndarray or None
        Per-pair score drops (clean − degraded). Written as a ``delta`` column
        when provided. Not aggregated here; caller handles that separately.
    """
    output_dir.mkdir(parents=True, exist_ok=True)

    pairs_df = pd.DataFrame({"score": scores, "label": labels})
    if deltas is not None:
        pairs_df["delta"] = deltas
    pairs_df.to_csv(output_dir / f"{task_name}_pairs.csv", index=False)

    summary_df = pd.DataFrame([{"task": task_name, **results}])
    summary_path = output_dir / "summary.csv"
    if summary_path.exists():
        existing = pd.read_csv(summary_path)
        summary_df = pd.concat([existing, summary_df], ignore_index=True)
    summary_df.to_csv(summary_path, index=False)

    logger.info(
        f"[{task_name}] EER={results['eer']:.4f} | "
        f"EER-threshold={results['eer_threshold']:.4f} | "
        f"Applied-threshold={results['applied_threshold']:.4f} | "
        f"Accuracy={results['accuracy']:.4f}"
    )


# ---------------------------------------------------------------------------
# Task 0: Threshold calibration (unenrolled speakers only)
# ---------------------------------------------------------------------------


def run_task0_calibrate(
    extractor: EmbeddingExtractor,
    unenrolled_ids: list[str],
    output_dir: Path,
    calibrated_threshold_path: Path,
    vc1_dir: str,
    vc2_dir: str,
    sample_rate: int,
    n_genuine: int,
    n_impostor: int,
    seed: int,
) -> float:
    """Task 0 – Calibrate the cosine similarity threshold on unenrolled speakers.

    Builds genuine/impostor pairs exclusively from speakers NOT in the database,
    computes the EER threshold, and persists it to a JSON file.

    Parameters
    ----------
    extractor : EmbeddingExtractor
        Initialized model.
    unenrolled_ids : list[str]
        Speaker IDs to use for calibration (must not overlap with enrolled set).
    output_dir : Path
        Directory to write per-pair scores.
    calibrated_threshold_path : Path
        Path to save the resulting ``{"threshold": <float>}`` JSON.
    vc1_dir : str
        VoxCeleb1_test root.
    vc2_dir : str
        VoxCeleb2_test root.
    sample_rate : int
        Target sample rate.
    n_genuine : int
        Number of genuine pairs to sample.
    n_impostor : int
        Number of impostor pairs to sample.
    seed : int
        Random seed for reproducibility.

    Returns
    -------
    float
        The calibrated EER threshold, also written to ``calibrated_threshold_path``.
    """
    logger.info("=== Task 0: Threshold Calibration ===")
    logger.info(
        f"Using {len(unenrolled_ids)} unenrolled speakers for calibration pairs"
    )

    pairs = build_test_pairs(
        vc1_dir=vc1_dir,
        vc2_dir=vc2_dir,
        n_genuine=n_genuine,
        n_impostor=n_impostor,
        seed=seed,
        speaker_ids=unenrolled_ids,
    )

    if not pairs:
        logger.error("No calibration pairs could be built; aborting Task 0")
        raise RuntimeError("Task 0 failed: no pairs available from unenrolled speakers")

    scores, labels, _, _ = _extract_pair_scores(extractor, pairs, sample_rate=sample_rate)
    eer, eer_threshold = compute_eer(scores, labels)

    calibrated_threshold_path.parent.mkdir(parents=True, exist_ok=True)
    with open(calibrated_threshold_path, "w") as f:
        json.dump({"threshold": eer_threshold, "eer": eer}, f, indent=2)

    results = evaluate(scores, labels)
    _save_results(results, scores, labels, output_dir, "task0_calibration")

    logger.info(
        f"[Task 0] Calibrated threshold={eer_threshold:.4f} (EER={eer:.4f}) "
        f"saved to {calibrated_threshold_path}"
    )
    return eer_threshold


# ---------------------------------------------------------------------------
# Task runners
# ---------------------------------------------------------------------------


def run_task1(
    extractor: EmbeddingExtractor,
    pairs: list[AudioPair],
    output_dir: Path,
    sample_rate: int,
    fixed_threshold: float | None = None,
) -> tuple[dict[str, float], dict[tuple[str, str], float]]:
    """Task 1 – Baseline effectiveness on the clean test set.

    Returns
    -------
    tuple[dict[str, float], dict[tuple[str, str], float]]
        ``(metrics, baseline_scores)`` where ``baseline_scores`` maps
        ``(path1, path2) -> clean_cosine_similarity`` for use as delta
        reference in Tasks 2-7.
    """
    logger.info("=== Task 1: Baseline ===")
    scores, labels, _, processed_keys = _extract_pair_scores(
        extractor, pairs, sample_rate=sample_rate
    )
    results = evaluate(scores, labels, fixed_threshold=fixed_threshold)
    _save_results(results, scores, labels, output_dir, "task1_baseline")
    baseline_scores: dict[tuple[str, str], float] = {
        key: float(score) for key, score in zip(processed_keys, scores)
    }
    logger.info(f"  Cached {len(baseline_scores)} baseline scores for delta tracking")
    return results, baseline_scores


def run_task2(
        extractor: EmbeddingExtractor,
        pairs: list[AudioPair],
        output_dir: Path,
        sample_rate: int,
        amplitude_factors: list[float],
        n_samples: int,
        fixed_threshold: float | None = None,
        baseline_scores: dict[tuple[str, str], float] | None = None,
) -> dict[str, dict[str, float]]:
    """Task 2 – Amplitude scaling applied randomly (uniformly) from a set of factors."""
    logger.info("=== Task 2: Random Amplitude Scaling ===")

    subset = _stratified_subset(pairs, n_samples)

    def random_scale_aug(waveform, seed):
        """Randomly choose an amplitude factor using the unique file/pair seed."""
        chosen_factor = np.random.choice(amplitude_factors)
        return scale_amplitude(waveform, chosen_factor)

    tag = "task2_amplitude_random_mixed"
    logger.info(f"  Uniformly sampling factors from: {amplitude_factors}")

    scores, labels, deltas, _ = _extract_pair_scores(
        extractor,
        subset,
        augment_fn=random_scale_aug,
        sample_rate=sample_rate,
        baseline_scores=baseline_scores,
    )

    results = evaluate(scores, labels, fixed_threshold=fixed_threshold)
    _save_results(results, scores, labels, output_dir, tag, deltas=deltas)

    return {"random_mixed": results}


def run_task3(
        extractor: EmbeddingExtractor,
        pairs: list[AudioPair],
        output_dir: Path,
        sample_rate: int,
        naive_steps: list[int],
        interp_factors: list[int],
        n_samples: int,
        fixed_threshold: float | None = None,
        baseline_scores: dict[tuple[str, str], float] | None = None,
) -> dict[str, dict[str, float]]:
    """Task 3 – Resampling: naive subsampling vs. interpolated downsampling."""
    logger.info("=== Task 3: Resampling ===")

    subset = _stratified_subset(pairs, n_samples)
    all_results: dict[str, dict[str, float]] = {}

    for step in naive_steps:
        tag = f"task3_naive_step{step}"

        nyquist_limit = (sample_rate // step) / 2
        logger.info(
            f"  Naive step={step} | Data retained: {100 / step:.1f}% | "
            f"Max frequency: {nyquist_limit} Hz | "
            f"Theoretical required sample length: ~{step}x"
        )

        aug = lambda w, _seed, s=step, sr=sample_rate: naive_subsample(
            w, step=s, orig_sr=sr, target_sr=sr
        )

        scores, labels, deltas, _ = _extract_pair_scores(
            extractor, subset, augment_fn=aug, sample_rate=sample_rate,
            baseline_scores=baseline_scores,
        )
        results = evaluate(scores, labels, fixed_threshold=fixed_threshold)
        _save_results(results, scores, labels, output_dir, tag, deltas=deltas)
        all_results[tag] = results

    for factor in interp_factors:
        tag = f"task3_interp_factor{factor}"

        nyquist_limit = (sample_rate // factor) / 2
        logger.info(
            f"  Interpolated factor={factor} | Data retained: {100 / factor:.1f}% | "
            f"Max frequency: {nyquist_limit} Hz | "
            f"Theoretical required sample length: ~{factor}x"
        )

        aug = lambda w, _seed, f=factor, sr=sample_rate: interpolated_resample(
            w, orig_sr=sr, factor=f, target_sr=sr
        )

        scores, labels, deltas, _ = _extract_pair_scores(
            extractor, subset, augment_fn=aug, sample_rate=sample_rate,
            baseline_scores=baseline_scores,
        )
        results = evaluate(scores, labels, fixed_threshold=fixed_threshold)
        _save_results(results, scores, labels, output_dir, tag, deltas=deltas)
        all_results[tag] = results

    return all_results


def run_task4(
    extractor: EmbeddingExtractor,
    pairs: list[AudioPair],
    output_dir: Path,
    sample_rate: int,
    snr_levels: list[float],
    n_samples: int,
    fixed_threshold: float | None = None,
    baseline_scores: dict[tuple[str, str], float] | None = None,
) -> dict[str, dict[str, float]]:
    """Task 4 – Additive Gaussian noise at three SNR levels."""
    logger.info("=== Task 4: Gaussian Noise ===")
    subset = _stratified_subset(pairs, n_samples)
    all_results: dict[str, dict[str, float]] = {}

    for snr in snr_levels:
        tag = f"task4_gaussian_snr{snr}dB"
        logger.info(f"  SNR={snr} dB")
        aug = lambda w, _seed, s=snr: add_gaussian_noise(w, s)  # noqa: E731
        scores, labels, deltas, _ = _extract_pair_scores(
            extractor, subset, augment_fn=aug, sample_rate=sample_rate,
            baseline_scores=baseline_scores,
        )
        results = evaluate(scores, labels, fixed_threshold=fixed_threshold)
        _save_results(results, scores, labels, output_dir, tag, deltas=deltas)
        all_results[str(snr)] = results

    return all_results


def _ensure_urbansound8k(noise_dir: str) -> None:
    """Download UrbanSound8K via soundata if the directory is absent.

    Parameters
    ----------
    noise_dir : str
        Expected path to the UrbanSound8K root (e.g. ``data/input/UrbanSound8K``).
    """
    import soundata as _soundata

    if Path(noise_dir).exists():
        return
    data_home = str(Path(noise_dir).parent)
    logger.info(
        f"UrbanSound8K not found at {noise_dir}; downloading (~6 GB) via soundata …"
    )
    dataset = _soundata.initialize("urbansound8k", data_home=data_home)
    dataset.download()
    dataset.validate()
    logger.info("UrbanSound8K download complete.")


def run_task5(
    extractor: EmbeddingExtractor,
    pairs: list[AudioPair],
    output_dir: Path,
    sample_rate: int,
    snr_levels: list[float],
    noise_dir: str,
    n_samples: int,
    fixed_threshold: float | None = None,
    baseline_scores: dict[tuple[str, str], float] | None = None,
) -> dict[str, dict[str, float]]:
    """Task 5 – Environmental noise from UrbanSound8K at three SNR levels."""
    logger.info("=== Task 5: Environmental Noise ===")
    _ensure_urbansound8k(noise_dir)

    subset = _stratified_subset(pairs, n_samples)
    all_results: dict[str, dict[str, float]] = {}

    for snr in snr_levels:
        tag = f"task5_env_snr{snr}dB"
        logger.info(f"  SNR={snr} dB")
        aug = lambda w, seed, s=snr, nd=noise_dir, sr=sample_rate: add_environmental_noise(  # noqa: E731
            w, nd, s, sr, seed=seed
        )
        scores, labels, deltas, _ = _extract_pair_scores(
            extractor, subset, augment_fn=aug, sample_rate=sample_rate,
            baseline_scores=baseline_scores,
        )
        results = evaluate(scores, labels, fixed_threshold=fixed_threshold)
        _save_results(results, scores, labels, output_dir, tag, deltas=deltas)
        all_results[str(snr)] = results

    return all_results


def run_task6(
    extractor: EmbeddingExtractor,
    pairs: list[AudioPair],
    output_dir: Path,
    sample_rate: int,
    codec_config: dict[str, list[int]],
    n_samples: int,
    fixed_threshold: float | None = None,
    baseline_scores: dict[tuple[str, str], float] | None = None,
) -> dict[str, dict[str, float]]:
    """Task 6 – Lossy codec compression (MP3, AAC, Opus) at multiple bitrates."""
    logger.info("=== Task 6: Lossy Compression ===")
    subset = _stratified_subset(pairs, n_samples)
    all_results: dict[str, dict[str, float]] = {}

    for codec, bitrates in codec_config.items():
        for bitrate in bitrates:
            tag = f"task6_{codec}_{bitrate}kbps"
            logger.info(f"  Codec={codec} bitrate={bitrate} kbps")
            try:
                aug = lambda w, _seed, c=codec, b=bitrate, sr=sample_rate: apply_codec_compression(  # noqa: E731
                    w, sr, c, b
                )
                scores, labels, deltas, _ = _extract_pair_scores(
                    extractor, subset, augment_fn=aug, sample_rate=sample_rate,
                    baseline_scores=baseline_scores,
                )
                results = evaluate(scores, labels, fixed_threshold=fixed_threshold)
                _save_results(results, scores, labels, output_dir, tag, deltas=deltas)
                all_results[tag] = results
            except Exception as exc:
                logger.error(f"  Task 6 failed for {codec}@{bitrate}kbps: {exc}")

    return all_results


def run_task7(
    extractor: EmbeddingExtractor,
    pairs: list[AudioPair],
    output_dir: Path,
    sample_rate: int,
    rir_dir: str,
    n_samples: int,
    fixed_threshold: float | None = None,
    baseline_scores: dict[tuple[str, str], float] | None = None,
) -> dict[str, float]:
    """Task 7 – Reverberation via Room Impulse Response convolution."""
    logger.info("=== Task 7: Reverberation ===")
    rir_path = Path(rir_dir)
    if not rir_path.exists():
        logger.warning(f"RIR directory not found at {rir_dir}; skipping Task 7")
        return {}

    subset = _stratified_subset(pairs, n_samples)

    aug = lambda w, seed, r=rir_dir, sr=sample_rate: apply_reverberation(w, r, sr, seed=seed)  # noqa: E731
    scores, labels, deltas, _ = _extract_pair_scores(
        extractor, subset, augment_fn=aug, sample_rate=sample_rate,
        baseline_scores=baseline_scores,
    )
    results = evaluate(scores, labels, fixed_threshold=fixed_threshold)
    _save_results(results, scores, labels, output_dir, "task7_reverberation", deltas=deltas)
    return results


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def _load_calibrated_threshold(threshold_path: Path) -> float | None:
    """Load the calibrated threshold from JSON if it exists.

    Parameters
    ----------
    threshold_path : Path
        Path to the JSON file produced by Task 0.

    Returns
    -------
    float or None
        The threshold value, or ``None`` if the file is absent or unreadable.
    """
    if not threshold_path.exists():
        logger.warning(
            f"Calibrated threshold not found at {threshold_path}. "
            "Run Task 0 first, or accuracy metrics will use EER threshold."
        )
        return None
    try:
        with open(threshold_path, "r") as f:
            data = json.load(f)
        threshold = float(data["threshold"])
        logger.info(f"Loaded calibrated threshold={threshold:.4f} from {threshold_path}")
        return threshold
    except (KeyError, ValueError, OSError) as exc:
        logger.error(f"Failed to read calibrated threshold: {exc}")
        return None


def main() -> None:
    """Entry point for the benchmark runner."""
    parser = argparse.ArgumentParser(
        description="Run speaker identification benchmark suite"
    )
    parser.add_argument(
        "--config",
        type=str,
        default="config.yaml",
        help="Path to the YAML configuration file",
    )
    parser.add_argument(
        "--tasks",
        nargs="+",
        type=int,
        choices=range(0, 8),
        default=list(range(0, 8)),
        metavar="N",
        help="Which tasks to run (0=calibrate, 1-7=benchmark). Defaults to all.",
    )
    args = parser.parse_args()

    cfg = load_config(args.config)

    model_cfg = cfg["model"]
    audio_cfg = cfg["audio"]
    bm_cfg = cfg["benchmarks"]
    paths = cfg["paths"]

    sample_rate: int = audio_cfg["sample_rate"]
    output_dir = Path(paths["reports"])
    calibrated_threshold_path = Path(paths["calibrated_threshold"])

    extractor = EmbeddingExtractor(
        model_source=model_cfg["source"],
        savedir=model_cfg["savedir"],
    )

    # Load the speaker database to identify enrolled speakers and their videos
    db = SpeakerDatabase(db_path=paths["speaker_db"])
    enrolled_ids = db.list_speakers()
    enrollment_video_map = db.get_enrollment_video_map()
    logger.info(f"Enrolled speakers in DB: {len(enrolled_ids)}")

    # Derive unenrolled speakers for Task 0 calibration
    full_index = collect_speakers(
        vc1_dir=paths["vc1_data"],
        vc2_dir=paths["vc2_data"],
    )
    enrolled_set = set(enrolled_ids)
    unenrolled_ids = [sid for sid in full_index if sid not in enrolled_set]
    logger.info(
        f"Unenrolled speakers available for calibration: {len(unenrolled_ids)}"
    )

    tasks = set(args.tasks)

    # Task 0: calibrate threshold on unenrolled speakers
    if 0 in tasks:
        run_task0_calibrate(
            extractor=extractor,
            unenrolled_ids=unenrolled_ids,
            output_dir=output_dir,
            calibrated_threshold_path=calibrated_threshold_path,
            vc1_dir=paths["vc1_data"],
            vc2_dir=paths["vc2_data"],
            sample_rate=sample_rate,
            n_genuine=bm_cfg["n_genuine"],
            n_impostor=bm_cfg["n_impostor"],
            seed=bm_cfg["random_seed"],
        )

    # Tasks 1-7 use only enrolled speakers, excluding their enrollment videos
    if tasks & set(range(1, 8)):
        fixed_threshold = _load_calibrated_threshold(calibrated_threshold_path)

        logger.info(
            "Building test pairs for enrolled speakers "
            "(enrollment videos excluded) …"
        )
        pairs = build_test_pairs(
            vc1_dir=paths["vc1_data"],
            vc2_dir=paths["vc2_data"],
            n_genuine=bm_cfg["n_genuine"],
            n_impostor=bm_cfg["n_impostor"],
            seed=bm_cfg["random_seed"],
            speaker_ids=enrolled_ids,
            excluded_videos=enrollment_video_map,
        )

        baseline_scores: dict[tuple[str, str], float] = {}

        if 1 in tasks:
            _, baseline_scores = run_task1(
                extractor, pairs, output_dir, sample_rate,
                fixed_threshold=fixed_threshold,
            )

        if 2 in tasks:
            t2 = bm_cfg["task2"]
            run_task2(
                extractor,
                pairs,
                output_dir,
                sample_rate,
                amplitude_factors=t2["amplitude_factors"],
                n_samples=t2["n_samples"],
                fixed_threshold=fixed_threshold,
                baseline_scores=baseline_scores or None,
            )

        if 3 in tasks:
            t3 = bm_cfg["task3"]
            run_task3(
                extractor,
                pairs,
                output_dir,
                sample_rate,
                naive_steps=t3["naive_steps"],
                interp_factors=t3["interp_factors"],
                n_samples=bm_cfg["n_genuine"] + bm_cfg["n_impostor"],
                fixed_threshold=fixed_threshold,
                baseline_scores=baseline_scores or None,
            )

        if 4 in tasks:
            t4 = bm_cfg["task4"]
            run_task4(
                extractor,
                pairs,
                output_dir,
                sample_rate,
                snr_levels=t4["snr_levels_db"],
                n_samples=t4["n_samples"],
                fixed_threshold=fixed_threshold,
                baseline_scores=baseline_scores or None,
            )

        if 5 in tasks:
            t5 = bm_cfg["task5"]
            run_task5(
                extractor,
                pairs,
                output_dir,
                sample_rate,
                snr_levels=t5["snr_levels_db"],
                noise_dir=paths["urban_sound"],
                n_samples=t5["n_samples"],
                fixed_threshold=fixed_threshold,
                baseline_scores=baseline_scores or None,
            )

        if 6 in tasks:
            t6 = bm_cfg["task6"]
            run_task6(
                extractor,
                pairs,
                output_dir,
                sample_rate,
                codec_config=t6["codecs"],
                n_samples=t6["n_samples"],
                fixed_threshold=fixed_threshold,
                baseline_scores=baseline_scores or None,
            )

        if 7 in tasks:
            t7 = bm_cfg["task7"]
            run_task7(
                extractor,
                pairs,
                output_dir,
                sample_rate,
                rir_dir=paths["rir_data"],
                n_samples=t7["n_samples"],
                fixed_threshold=fixed_threshold,
                baseline_scores=baseline_scores or None,
            )

    logger.info(f"All requested tasks complete. Reports saved to {output_dir}")


if __name__ == "__main__":
    main()
