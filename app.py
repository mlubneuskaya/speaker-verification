"""Streamlit interface for speaker enrollment and verification."""

import json
import logging
import os
import random
import re
import tempfile
import uuid
from pathlib import Path
from typing import Any

import numpy as np
import streamlit as st
import torch

from benchmarks.dataset import collect_speakers
from src.config import load_config
from src.database import SpeakerDatabase
from src.embeddings import EmbeddingExtractor
from src.enrollment import enroll_speaker
from src.preprocessing import preprocess_audio
from src.utils import save_enrollment_timing

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
)
logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Cached resource loading
# ---------------------------------------------------------------------------


@st.cache_resource
def get_config() -> dict[str, Any]:
    """Load and cache the project configuration."""
    return load_config("config.yaml")


@st.cache_resource
def get_extractor(model_source: str, savedir: str) -> EmbeddingExtractor:
    """Load and cache the ECAPA-TDNN model (runs once per session)."""
    return EmbeddingExtractor(model_source=model_source, savedir=savedir)


@st.cache_resource
def get_database(db_path: str) -> SpeakerDatabase:
    """Load and cache the speaker database (shared across pages)."""
    return SpeakerDatabase(db_path=db_path)


def _load_calibrated_threshold(threshold_path: str, fallback: float) -> float:
    """Return the calibrated threshold from JSON, or ``fallback`` if absent.

    Parameters
    ----------
    threshold_path : str
        Path to the JSON file produced by Task 0 calibration.
    fallback : float
        Value to use when the file does not exist or cannot be parsed.
    """
    path = Path(threshold_path)
    if not path.exists():
        return fallback
    try:
        with open(path, "r") as f:
            data = json.load(f)
        return float(data["threshold"])
    except (KeyError, ValueError, OSError) as exc:
        logger.warning(f"Could not read calibrated threshold: {exc}; using fallback")
        return fallback


# ---------------------------------------------------------------------------
# Utility helpers
# ---------------------------------------------------------------------------


def _save_upload_to_temp(uploaded_file: Any, suffix: str = ".wav") -> str:
    """Write a Streamlit UploadedFile to a temporary file and return its path."""
    tmp = tempfile.NamedTemporaryFile(suffix=suffix, delete=False)
    tmp.write(uploaded_file.read())
    tmp.flush()
    tmp.close()
    return tmp.name


def _cleanup_temp_files(paths: list[str]) -> None:
    """Remove temporary files created during a request."""
    for p in paths:
        try:
            os.unlink(p)
        except OSError:
            pass


def _generate_speaker_id(name: str) -> str:
    """Derive a unique speaker ID from a display name.

    Parameters
    ----------
    name : str
        Human-readable display name (e.g. ``"Alice Smith"``).

    Returns
    -------
    str
        URL-safe ID such as ``"alice_smith_3f2a1c"``.
    """
    slug = re.sub(r"[^a-z0-9]+", "_", name.lower().strip()).strip("_") or "speaker"
    return f"{slug}_{uuid.uuid4().hex[:6]}"


# ---------------------------------------------------------------------------
# Page: Add Speaker
# ---------------------------------------------------------------------------


def page_add_speaker(
    extractor: EmbeddingExtractor,
    db: SpeakerDatabase,
    cfg: dict[str, Any],
) -> None:
    """Render the speaker enrollment page."""
    st.header("Add Speaker")

    name = st.text_input("Name", placeholder="e.g. Alice Smith")

    st.caption("Record 3 samples, 3–8 s each.")
    uploads = []
    cols = st.columns(3)
    for i, col in enumerate(cols):
        with col:
            f = st.audio_input(f"Sample {i + 1}", key=f"enroll_record_{i}")
            if f:
                uploads.append(f)

    if st.button("Enroll Speaker", type="primary"):
        if not name.strip():
            st.error("Please enter a name.")
            return
        if len(uploads) < 3:
            st.error(f"Please provide all 3 samples (got {len(uploads)}).")
            return

        user_id = _generate_speaker_id(name.strip())
        temp_paths: list[str] = []
        try:
            with st.spinner("Processing samples …"):
                for up in uploads:
                    suffix = getattr(up, "name", "audio.wav")
                    suffix = Path(suffix).suffix or ".wav"
                    tmp_path = _save_upload_to_temp(up, suffix=suffix)
                    temp_paths.append(tmp_path)

                audio_cfg = cfg["audio"]
                _, elapsed = enroll_speaker(
                    extractor=extractor,
                    db=db,
                    user_id=user_id,
                    name=name.strip(),
                    audio_paths=temp_paths,
                    target_sr=audio_cfg["sample_rate"],
                    energy_threshold=audio_cfg["vad_energy_threshold"],
                    frame_length_ms=audio_cfg["vad_frame_length_ms"],
                )
                save_enrollment_timing(
                    user_id=user_id,
                    n_samples=len(temp_paths),
                    total_seconds=elapsed,
                    reports_dir=cfg["paths"]["reports"],
                )
            st.success(
                f"Speaker **{name.strip()}** enrolled (id: `{user_id}`). "
                f"Took **{elapsed:.1f}s** ({elapsed / len(temp_paths):.1f}s/sample)."
            )
        except Exception as exc:
            logger.error(f"Enrollment failed: {exc}")
            st.error(f"Enrollment failed: {exc}")
        finally:
            _cleanup_temp_files(temp_paths)


# ---------------------------------------------------------------------------
# Page: Verify Speaker
# ---------------------------------------------------------------------------


def _find_speaker_by_name(db: SpeakerDatabase, name: str) -> list[tuple[str, dict]]:
    """Return all (user_id, record) pairs whose display name matches *name*.

    Comparison is case-insensitive.

    Parameters
    ----------
    db : SpeakerDatabase
        Initialized speaker database.
    name : str
        Display name to search for.

    Returns
    -------
    list[tuple[str, dict]]
        Matching ``(user_id, record)`` pairs (empty when no match).
    """
    target = name.strip().lower()
    matches = []
    for uid in db.list_speakers():
        try:
            record = db.get_speaker(uid)
            if record["name"].lower() == target:
                matches.append((uid, record))
        except Exception as exc:
            logger.warning(f"Could not read record for {uid}: {exc}")
    return matches


def page_verify_speaker(
    extractor: EmbeddingExtractor,
    db: SpeakerDatabase,
    cfg: dict[str, Any],
) -> None:
    """Render the speaker verification page."""
    st.header("Verify Speaker")

    enrolled_ids = db.list_speakers()
    if not enrolled_ids:
        st.warning("No speakers enrolled yet. Go to **Add Speaker** first.")
        return

    name = st.text_input("Name", placeholder="e.g. Alice Smith")

    st.caption("Click the microphone and speak for 3–8 s.")
    probe_file = st.audio_input("Probe recording", key="verify_probe_record")

    threshold = _load_calibrated_threshold(
        threshold_path=cfg["paths"]["calibrated_threshold"],
        fallback=float(cfg["thresholds"]["cosine_similarity"]),
    )

    if st.button("Verify", type="primary"):
        if not name.strip():
            st.error("Please enter a name.")
            return
        if not probe_file:
            st.error("Please provide a recording.")
            return

        matches = _find_speaker_by_name(db, name)
        if not matches:
            st.error(f"No enrolled speaker named **{name.strip()}**.")
            return
        if len(matches) > 1:
            st.warning(
                f"{len(matches)} speakers share the name **{name.strip()}**; "
                "using the first enrolled entry."
            )

        uid, record = matches[0]
        template: np.ndarray = record["embedding"]

        tmp_path: str | None = None
        try:
            with st.spinner("Verifying …"):
                suffix = Path(getattr(probe_file, "name", "audio.wav")).suffix or ".wav"
                tmp_path = _save_upload_to_temp(probe_file, suffix=suffix)
                audio_cfg = cfg["audio"]
                waveform = preprocess_audio(
                    tmp_path,
                    target_sr=audio_cfg["sample_rate"],
                    energy_threshold=audio_cfg["vad_energy_threshold"],
                    frame_length_ms=audio_cfg["vad_frame_length_ms"],
                )
                probe_emb = extractor.extract(waveform)
                score = float(np.dot(probe_emb, template))
                is_match = score >= threshold

            if is_match:
                st.success(
                    f"**Verified** — recording matches **{record['name']}**  \n"
                    f"Similarity score: `{score:.4f}`"
                )
            else:
                st.error(
                    f"**Not verified** — score `{score:.4f}` is below "
                    f"threshold `{threshold:.4f}`."
                )

            st.metric("Cosine Similarity", f"{score:.4f}")

        except Exception as exc:
            logger.error(f"Verification failed: {exc}")
            st.error(f"Verification failed: {exc}")
        finally:
            if tmp_path:
                _cleanup_temp_files([tmp_path])


# ---------------------------------------------------------------------------
# Page: Enrolled Speakers
# ---------------------------------------------------------------------------


def page_list_speakers(db: SpeakerDatabase) -> None:
    """Render a table of all enrolled speakers with per-row delete buttons."""
    st.header("Enrolled Speakers")

    enrolled_ids = db.list_speakers()
    if not enrolled_ids:
        st.info("No speakers enrolled yet.")
        return

    st.caption(f"Total: {len(enrolled_ids)} speaker(s)")

    # Header row
    h_name, h_id, h_date, h_action = st.columns([2, 2, 2, 1])
    h_name.markdown("**Name**")
    h_id.markdown("**ID**")
    h_date.markdown("**Enrolled**")
    h_action.markdown("**Delete**")
    st.divider()

    for uid in enrolled_ids:
        try:
            record = db.get_speaker(uid)
        except Exception as exc:
            logger.warning(f"Could not read record for {uid}: {exc}")
            continue

        col_name, col_id, col_date, col_btn = st.columns([2, 2, 2, 1])
        col_name.write(record["name"])
        col_id.write(f"`{uid}`")
        col_date.write(record["metadata"].get("enrollment_date", "—"))

        if col_btn.button("🗑", key=f"del_{uid}", help=f"Delete {record['name']}"):
            try:
                db.delete_speaker(uid)
                st.rerun()
            except Exception as exc:
                st.error(f"Delete failed: {exc}")


# ---------------------------------------------------------------------------
# Database auto-initialisation
# ---------------------------------------------------------------------------


def _auto_init_database(
    extractor: EmbeddingExtractor,
    db: SpeakerDatabase,
    cfg: dict[str, Any],
) -> None:
    """Enroll speakers from VoxCeleb into an empty database.

    Runs once on first app launch when the database has no entries. Progress
    is displayed via a Streamlit status block so the user knows what is
    happening.

    Parameters
    ----------
    extractor : EmbeddingExtractor
        Loaded ECAPA model.
    db : SpeakerDatabase
        Empty speaker database.
    cfg : dict[str, Any]
        Full project configuration.
    """
    paths = cfg["paths"]
    audio_cfg = cfg["audio"]
    n_users: int = cfg.get("init_db", {}).get("n_users", 97)
    seed: int = cfg.get("init_db", {}).get("seed", 42)
    enrollment_samples: int = 3

    index = collect_speakers(
        vc1_dir=paths["vc1_data"],
        vc2_dir=paths["vc2_data"],
    )
    eligible = {
        spk: videos
        for spk, videos in index.items()
        if len(videos) >= enrollment_samples
    }

    rng = random.Random(seed)
    speaker_list = sorted(eligible.keys())
    rng.shuffle(speaker_list)
    speaker_list = speaker_list[:n_users]

    status = st.status(
        f"Database is empty — enrolling {len(speaker_list)} speakers …",
        expanded=True,
    )
    enrolled = 0
    for speaker_id in speaker_list:
        videos = eligible[speaker_id]
        chosen_videos = rng.sample(list(videos.keys()), enrollment_samples)
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
            status.write(
                f"[{enrolled}/{len(speaker_list)}] Enrolled {speaker_id} "
                f"({elapsed:.1f}s)"
            )
        except Exception as exc:
            logger.warning(f"Skipped {speaker_id}: {exc}")

    status.update(
        label=f"Database ready — {enrolled} speaker(s) enrolled.",
        state="complete",
        expanded=False,
    )


# ---------------------------------------------------------------------------
# App entry point
# ---------------------------------------------------------------------------


def main() -> None:
    """Configure and launch the Streamlit application."""
    st.set_page_config(
        page_title="Speaker Identification",
        page_icon="🎙️",
        layout="wide",
    )

    cfg = get_config()
    extractor = get_extractor(
        model_source=cfg["model"]["source"],
        savedir=cfg["model"]["savedir"],
    )
    db = get_database(db_path=cfg["paths"]["speaker_db"])

    if len(db) == 0:
        _auto_init_database(extractor, db, cfg)

    st.sidebar.title("Speaker ID System")
    st.sidebar.caption("ECAPA-TDNN · VoxCeleb")

    page = st.sidebar.radio(
        "Navigation",
        options=["Add Speaker", "Verify Speaker", "Enrolled Speakers"],
    )

    if page == "Add Speaker":
        page_add_speaker(extractor, db, cfg)
    elif page == "Verify Speaker":
        page_verify_speaker(extractor, db, cfg)
    else:
        page_list_speakers(db)


if __name__ == "__main__":
    main()
