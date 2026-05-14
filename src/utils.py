import csv
import logging
import shutil
from datetime import datetime
from functools import lru_cache
from pathlib import Path

import torch

logger = logging.getLogger(__name__)

_TIMING_COLUMNS = ["timestamp", "user_id", "n_samples", "total_seconds", "seconds_per_sample"]


def save_enrollment_timing(
    user_id: str,
    n_samples: int,
    total_seconds: float,
    reports_dir: str = "reports",
) -> None:
    """Append one enrollment timing record to ``reports/enrollment_timing.csv``.

    Parameters
    ----------
    user_id : str
        Enrolled speaker ID.
    n_samples : int
        Number of audio samples used for enrollment.
    total_seconds : float
        Total wall-clock time for the enrollment in seconds.
    reports_dir : str
        Directory where the CSV file is written.
    """
    out_dir = Path(reports_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    csv_path = out_dir / "enrollment_timing.csv"

    write_header = not csv_path.exists()
    try:
        with open(csv_path, "a", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=_TIMING_COLUMNS)
            if write_header:
                writer.writeheader()
            writer.writerow({
                "timestamp": datetime.now().isoformat(timespec="seconds"),
                "user_id": user_id,
                "n_samples": n_samples,
                "total_seconds": round(total_seconds, 3),
                "seconds_per_sample": round(total_seconds / n_samples, 3),
            })
    except OSError as exc:
        logger.warning(f"Could not write enrollment timing: {exc}")


def get_device():
    if torch.cuda.is_available():
        return "cuda"
    if torch.mps.is_available():
        return "mps"
    else:
        return "cpu"


@lru_cache(maxsize=1)
def find_ffmpeg() -> str:
    """Return the absolute path to the ffmpeg executable.

    Checks $PATH first, then common Homebrew and system locations on macOS/Linux.

    Returns
    -------
    str
        Absolute path to ffmpeg.

    Raises
    ------
    FileNotFoundError
        If ffmpeg cannot be located.
    """
    path = shutil.which("ffmpeg")
    if path:
        return path

    candidates = [
        "/opt/homebrew/bin/ffmpeg",   # Apple Silicon Homebrew
        "/usr/local/bin/ffmpeg",       # Intel Homebrew / Linux
        "/usr/bin/ffmpeg",
    ]
    for candidate in candidates:
        if Path(candidate).exists():
            return candidate

    raise FileNotFoundError(
        "ffmpeg not found. Install it with: brew install ffmpeg  "
        "(or apt install ffmpeg on Linux)"
    )
