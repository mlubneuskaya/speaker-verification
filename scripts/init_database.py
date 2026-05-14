"""Populate the speaker database from VoxCeleb1_test and VoxCeleb2_test.

One speaker = one enrolled user. Enrollment uses exactly 3 clips drawn from
3 **distinct videos** so the master template is not biased toward a single
recording session.

Run from the project root:
    python -m scripts.init_database
    python -m scripts.init_database --n-users 97 --seed 42
"""

import argparse
import logging
import random
import sys

from benchmarks.dataset import collect_speakers
from src.config import load_config
from src.database import SpeakerDatabase
from src.embeddings import EmbeddingExtractor
from src.enrollment import enroll_speaker
from src.utils import save_enrollment_timing

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
)
logger = logging.getLogger(__name__)

ENROLLMENT_SAMPLES = 3  # clips, one from each of 3 distinct videos


def main() -> None:
    """Entry point for the database initialisation script."""
    parser = argparse.ArgumentParser(
        description="Initialise speaker database from VoxCeleb1_test and VoxCeleb2_test"
    )
    parser.add_argument(
        "--config", type=str, default="config.yaml", help="Path to config YAML"
    )
    parser.add_argument(
        "--n-users", type=int, default=None,
        help="Number of speakers to enroll (overrides init_db.n_users in config)"
    )
    parser.add_argument(
        "--seed", type=int, default=None,
        help="Random seed (overrides init_db.seed in config)"
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Clear the existing database before enrolling",
    )
    args = parser.parse_args()

    cfg = load_config(args.config)
    audio_cfg = cfg["audio"]
    paths = cfg["paths"]
    init_cfg = cfg["init_db"]

    n_users: int = args.n_users if args.n_users is not None else init_cfg["n_users"]
    seed: int = args.seed if args.seed is not None else init_cfg["seed"]

    db = SpeakerDatabase(db_path=paths["speaker_db"])

    if args.overwrite:
        for uid in list(db.list_speakers()):
            db.delete_speaker(uid)
        logger.info("Existing database cleared")

    already_enrolled = set(db.list_speakers())
    if already_enrolled:
        logger.info(f"{len(already_enrolled)} speaker(s) already in database")

    # Build a combined speaker index from both datasets
    index = collect_speakers(
        vc1_dir=paths["vc1_data"],
        vc2_dir=paths["vc2_data"],
    )

    # Only speakers with >= 3 distinct videos can provide 3 independent clips
    eligible = {
        spk: videos
        for spk, videos in index.items()
        if len(videos) >= ENROLLMENT_SAMPLES
    }
    logger.info(
        f"{len(eligible)}/{len(index)} speakers have >= {ENROLLMENT_SAMPLES} videos"
    )

    if len(eligible) < n_users:
        logger.warning(
            f"Only {len(eligible)} eligible speakers available; "
            f"will enroll all of them instead of {n_users}"
        )

    rng = random.Random(seed)
    speaker_list = sorted(eligible.keys())
    rng.shuffle(speaker_list)

    extractor = EmbeddingExtractor(
        model_source=cfg["model"]["source"],
        savedir=cfg["model"]["savedir"],
    )

    enrolled = 0
    skipped = 0

    for speaker_id in speaker_list:
        if enrolled >= n_users:
            break

        if speaker_id in already_enrolled:
            logger.debug(f"Skipping already-enrolled speaker: {speaker_id}")
            skipped += 1
            continue

        videos = eligible[speaker_id]
        # Pick 3 distinct videos, then one random clip from each
        chosen_videos = rng.sample(list(videos.keys()), ENROLLMENT_SAMPLES)
        sample_paths = [rng.choice(videos[vid]) for vid in chosen_videos]

        try:
            _, elapsed = enroll_speaker(
                extractor=extractor,
                db=db,
                user_id=speaker_id,
                name=speaker_id,
                audio_paths=sample_paths,
                target_sr=audio_cfg["sample_rate"],
                energy_threshold=audio_cfg["vad_energy_threshold"],
                frame_length_ms=audio_cfg["vad_frame_length_ms"],
            )
            enrolled += 1
            save_enrollment_timing(
                user_id=speaker_id,
                n_samples=ENROLLMENT_SAMPLES,
                total_seconds=elapsed,
                reports_dir=cfg["paths"]["reports"],
            )
            logger.info(
                f"[{enrolled}/{args.n_users}] Enrolled {speaker_id} ({elapsed:.1f}s)"
            )
        except Exception as exc:
            logger.warning(f"Failed to enroll {speaker_id}: {exc}")

    logger.info(
        f"Done. Enrolled {enrolled} new speaker(s), skipped {skipped} existing. "
        f"Database total: {len(db)} speaker(s)."
    )


if __name__ == "__main__":
    main()
