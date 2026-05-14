"""Build genuine/impostor test pairs from VoxCeleb1_test and VoxCeleb2_test.

Structure (both datasets share the same layout):
    <root>/<speaker_id>/<video_id>/<clip>.(wav|m4a)

One speaker = one identity. Genuine pairs are always drawn from two
**different videos** of the same speaker to avoid trivially correlated clips.
"""

import logging
import random
from pathlib import Path
from typing import NamedTuple

logger = logging.getLogger(__name__)

# Mapping: speaker_id -> {video_id -> [clip_paths]}
SpeakerIndex = dict[str, dict[str, list[str]]]

_AUDIO_EXTENSIONS = {".wav", ".m4a", ".flac"}


class AudioPair(NamedTuple):
    """A pair of audio file paths with a ground-truth label."""

    path1: str
    path2: str
    label: int  # 1 = genuine (same speaker), 0 = impostor (different speaker)


def _index_dataset(root: Path) -> SpeakerIndex:
    """Index one dataset root into a speaker -> video -> clips mapping.

    Parameters
    ----------
    root : Path
        Dataset root (e.g. ``data/input/VoxCeleb1_test``).

    Returns
    -------
    SpeakerIndex
        Nested dict ``{speaker_id: {video_id: [clip_path, ...]}}``.
    """
    index: SpeakerIndex = {}
    for speaker_dir in sorted(root.iterdir()):
        if not speaker_dir.is_dir():
            continue
        videos: dict[str, list[str]] = {}
        for video_dir in sorted(speaker_dir.iterdir()):
            if not video_dir.is_dir():
                continue
            clips = [
                str(f)
                for f in sorted(video_dir.iterdir())
                if f.suffix.lower() in _AUDIO_EXTENSIONS
            ]
            if clips:
                videos[video_dir.name] = clips
        if videos:
            index[speaker_dir.name] = videos
    return index


def collect_speakers(vc1_dir: str, vc2_dir: str) -> SpeakerIndex:
    """Combine VoxCeleb1_test and VoxCeleb2_test into a single speaker index.

    Parameters
    ----------
    vc1_dir : str
        Root of VoxCeleb1_test (WAV clips).
    vc2_dir : str
        Root of VoxCeleb2_test (M4A clips).

    Returns
    -------
    SpeakerIndex
        Combined index across both datasets. Speaker IDs do not overlap
        between VC1 (id1xxxx) and VC2 (id0xxxx).
    """
    index: SpeakerIndex = {}

    for label, path_str in (("VoxCeleb1", vc1_dir), ("VoxCeleb2", vc2_dir)):
        root = Path(path_str)
        if not root.exists():
            logger.warning(f"{label} directory not found: {root}; skipping")
            continue
        partial = _index_dataset(root)
        overlap = set(partial) & set(index)
        if overlap:
            logger.warning(f"Speaker ID collision between datasets: {overlap}")
        index.update(partial)
        logger.info(f"{label}: indexed {len(partial)} speakers from {root}")

    logger.info(f"Combined index: {len(index)} speakers total")
    return index


def build_test_pairs(
    vc1_dir: str,
    vc2_dir: str,
    n_genuine: int = 250,
    n_impostor: int = 250,
    seed: int = 42,
    speaker_ids: list[str] | None = None,
    excluded_videos: dict[str, list[str]] | None = None,
) -> list[AudioPair]:
    """Sample balanced genuine and impostor pairs for evaluation.

    Genuine pairs use clips from two **different videos** of the same speaker.
    Impostor pairs use clips from two **different speakers**.

    Parameters
    ----------
    vc1_dir : str
        Root of VoxCeleb1_test.
    vc2_dir : str
        Root of VoxCeleb2_test.
    n_genuine : int
        Number of genuine (same-speaker, different-video) pairs.
    n_impostor : int
        Number of impostor (different-speaker) pairs.
    seed : int
        Random seed for reproducibility.
    speaker_ids : list[str] or None
        If provided, only these speakers are used. Otherwise all indexed
        speakers are eligible.
    excluded_videos : dict[str, list[str]] or None
        Mapping of speaker_id -> list of video IDs to exclude (e.g. videos
        used during enrollment). Clips from these videos are never sampled.

    Returns
    -------
    list[AudioPair]
        Shuffled list of ``AudioPair`` instances.
    """
    rng = random.Random(seed)
    index = collect_speakers(vc1_dir, vc2_dir)
    excluded_videos = excluded_videos or {}

    if speaker_ids is not None:
        missing = set(speaker_ids) - set(index)
        if missing:
            logger.warning(f"{len(missing)} requested speaker(s) not in index: {missing}")
        index = {s: index[s] for s in speaker_ids if s in index}

    # Build per-speaker available video map (after exclusions)
    available: SpeakerIndex = {}
    for spk, videos in index.items():
        excl = set(excluded_videos.get(spk, []))
        filtered = {vid: clips for vid, clips in videos.items() if vid not in excl}
        if filtered:
            available[spk] = filtered

    multi_video_speakers = [s for s in available if len(available[s]) >= 2]

    genuine_pairs: list[AudioPair] = []
    attempts = 0
    while len(genuine_pairs) < n_genuine and attempts < n_genuine * 20:
        attempts += 1
        if not multi_video_speakers:
            break
        speaker = rng.choice(multi_video_speakers)
        vid1, vid2 = rng.sample(list(available[speaker].keys()), 2)
        clip1 = rng.choice(available[speaker][vid1])
        clip2 = rng.choice(available[speaker][vid2])
        genuine_pairs.append(AudioPair(path1=clip1, path2=clip2, label=1))

    if len(genuine_pairs) < n_genuine:
        logger.warning(
            f"Only sampled {len(genuine_pairs)} genuine pairs (requested {n_genuine})"
        )

    speakers = list(available.keys())
    impostor_pairs: list[AudioPair] = []
    attempts = 0
    while len(impostor_pairs) < n_impostor and attempts < n_impostor * 20:
        attempts += 1
        if len(speakers) < 2:
            break
        sp1, sp2 = rng.sample(speakers, 2)
        vid1 = rng.choice(list(available[sp1].keys()))
        vid2 = rng.choice(list(available[sp2].keys()))
        clip1 = rng.choice(available[sp1][vid1])
        clip2 = rng.choice(available[sp2][vid2])
        impostor_pairs.append(AudioPair(path1=clip1, path2=clip2, label=0))

    all_pairs = genuine_pairs + impostor_pairs
    rng.shuffle(all_pairs)
    logger.info(
        f"Test set: {len(genuine_pairs)} genuine + {len(impostor_pairs)} impostor "
        f"= {len(all_pairs)} total pairs "
        f"({len(available)} speakers, exclusions applied)"
    )
    return all_pairs
