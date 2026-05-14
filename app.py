"""Streamlit interface for speaker enrollment and verification."""

import logging
import os
import random
import tempfile
from pathlib import Path
from typing import Any

import streamlit as st
import torch

from benchmarks.dataset import collect_speakers
from src.config import load_config
from src.database import SpeakerDatabase
from src.embeddings import EmbeddingExtractor
from src.enrollment import enroll_speaker
from src.preprocessing import preprocess_audio
from src.utils import save_enrollment_timing
from src.verification import verify_speaker

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

    user_id = st.text_input("Speaker ID", placeholder="e.g. alice_01")
    name = st.text_input("Display Name", placeholder="e.g. Alice")

    mode = st.radio(
        "Input method",
        options=["Upload files", "Record audio"],
        horizontal=True,
    )

    uploads = []
    cols = st.columns(3)

    if mode == "Upload files":
        for i, col in enumerate(cols):
            with col:
                f = st.file_uploader(
                    f"Sample {i + 1}",
                    type=["wav", "flac"],
                    key=f"enroll_upload_{i}",
                )
                if f:
                    st.audio(f, format="audio/wav")
                    uploads.append(f)
    else:
        st.caption("Click the microphone to record each sample (3–8 s recommended).")
        for i, col in enumerate(cols):
            with col:
                f = st.audio_input(f"Sample {i + 1}", key=f"enroll_record_{i}")
                if f:
                    uploads.append(f)

    if st.button("Enroll Speaker", type="primary"):
        if not user_id.strip():
            st.error("Please enter a Speaker ID.")
            return
        if not name.strip():
            st.error("Please enter a Display Name.")
            return
        if len(uploads) < 3:
            st.error(f"Please provide all 3 samples (got {len(uploads)}).")
            return

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
                    user_id=user_id.strip(),
                    name=name.strip(),
                    audio_paths=temp_paths,
                    target_sr=audio_cfg["sample_rate"],
                    energy_threshold=audio_cfg["vad_energy_threshold"],
                    frame_length_ms=audio_cfg["vad_frame_length_ms"],
                )
                save_enrollment_timing(
                    user_id=user_id.strip(),
                    n_samples=len(temp_paths),
                    total_seconds=elapsed,
                    reports_dir=cfg["paths"]["reports"],
                )
            st.success(
                f"Speaker **{name}** enrolled successfully (id: `{user_id}`). "
                f"Enrollment took **{elapsed:.1f}s** "
                f"({elapsed / len(temp_paths):.1f}s per sample)."
            )
        except Exception as exc:
            logger.error(f"Enrollment failed: {exc}")
            st.error(f"Enrollment failed: {exc}")
        finally:
            _cleanup_temp_files(temp_paths)


# ---------------------------------------------------------------------------
# Page: Verify Speaker
# ---------------------------------------------------------------------------


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

    st.write(
        "Upload a probe recording. The system will compare it against all "
        "enrolled speakers and return the best match."
    )

    probe_file = st.file_uploader(
        "Probe Audio", type=["wav", "flac"], key="verify_probe"
    )
    threshold = st.slider(
        "Match Threshold (cosine similarity)",
        min_value=0.0,
        max_value=1.0,
        value=float(cfg["thresholds"]["cosine_similarity"]),
        step=0.01,
        help="Scores above this value are declared a match.",
    )

    if probe_file:
        st.audio(probe_file, format="audio/wav")

    if st.button("Verify", type="primary"):
        if not probe_file:
            st.error("Please upload a probe audio file.")
            return

        tmp_path: str | None = None
        try:
            with st.spinner("Verifying …"):
                tmp_path = _save_upload_to_temp(
                    probe_file, suffix=Path(probe_file.name).suffix or ".wav"
                )
                audio_cfg = cfg["audio"]
                best_id, best_score, is_match = verify_speaker(
                    extractor=extractor,
                    db=db,
                    audio_path=tmp_path,
                    threshold=threshold,
                    target_sr=audio_cfg["sample_rate"],
                    energy_threshold=audio_cfg["vad_energy_threshold"],
                    frame_length_ms=audio_cfg["vad_frame_length_ms"],
                )

            if is_match:
                record = db.get_speaker(best_id)
                st.success(
                    f"**Match found**: {record['name']} (id: `{best_id}`)  \n"
                    f"Similarity score: `{best_score:.4f}`"
                )
            else:
                st.error(
                    f"**No match.** Best score: `{best_score:.4f}` "
                    f"(id: `{best_id}`) — below threshold `{threshold:.2f}`."
                )

            st.metric("Best Cosine Similarity", f"{best_score:.4f}")

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
    """Render a table of all enrolled speakers."""
    st.header("Enrolled Speakers")

    enrolled_ids = db.list_speakers()
    if not enrolled_ids:
        st.info("No speakers enrolled yet.")
        return

    rows = []
    for uid in enrolled_ids:
        try:
            record = db.get_speaker(uid)
            rows.append(
                {
                    "ID": uid,
                    "Name": record["name"],
                    "Enrolled": record["metadata"].get("enrollment_date", "—"),
                }
            )
        except Exception as exc:
            logger.warning(f"Could not read record for {uid}: {exc}")

    import pandas as pd

    st.dataframe(pd.DataFrame(rows), use_container_width=True)
    st.caption(f"Total: {len(rows)} speaker(s)")

    with st.expander("Delete a speaker"):
        del_id = st.selectbox("Select speaker to delete", enrolled_ids)
        if st.button("Delete", type="secondary"):
            try:
                db.delete_speaker(del_id)
                st.success(f"Speaker `{del_id}` deleted.")
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
