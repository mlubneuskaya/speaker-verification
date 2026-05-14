"""Download and extract the RIRS_NOISES dataset (OpenSLR SLR28).

The dataset contains real and simulated Room Impulse Responses used by
benchmark Task 7 (reverberation). Uncompressed size is ~15 GB.

Run from the project root:
    python -m scripts.download_rir
"""

import logging
import tarfile
import urllib.request
from pathlib import Path

from src.config import load_config

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
)
logger = logging.getLogger(__name__)

_OPENSLR_URL = "https://www.openslr.org/resources/28/rirs_noises.zip"
_ARCHIVE_NAME = "rirs_noises.zip"


def _progress_hook(block_num: int, block_size: int, total_size: int) -> None:
    downloaded = block_num * block_size
    if total_size > 0:
        pct = min(downloaded / total_size * 100, 100)
        mb = downloaded / 1_048_576
        total_mb = total_size / 1_048_576
        print(f"\r  {pct:.1f}%  {mb:.0f} / {total_mb:.0f} MB", end="", flush=True)


def main() -> None:
    """Download RIRS_NOISES and place WAV files under the configured RIR path."""
    cfg = load_config("config.yaml")
    rir_dir = Path(cfg["paths"]["rir_data"])
    rir_dir.mkdir(parents=True, exist_ok=True)

    existing_wavs = list(rir_dir.rglob("*.wav"))
    if existing_wavs:
        logger.info(
            f"Found {len(existing_wavs)} WAV file(s) already in {rir_dir}; "
            "skipping download. Delete the directory to re-download."
        )
        return

    archive_path = rir_dir / _ARCHIVE_NAME

    if not archive_path.exists():
        logger.info(f"Downloading RIRS_NOISES from {_OPENSLR_URL} …")
        logger.info("This is ~10 GB — it may take a while.")
        try:
            urllib.request.urlretrieve(_OPENSLR_URL, archive_path, _progress_hook)
            print()  # newline after progress bar
            logger.info(f"Download complete: {archive_path}")
        except Exception as exc:
            logger.error(f"Download failed: {exc}")
            archive_path.unlink(missing_ok=True)
            raise
    else:
        logger.info(f"Archive already present at {archive_path}; skipping download.")

    logger.info("Extracting archive …")
    try:
        import zipfile
        with zipfile.ZipFile(archive_path, "r") as zf:
            zf.extractall(rir_dir)
        logger.info(f"Extracted to {rir_dir}")
    except Exception as exc:
        logger.error(f"Extraction failed: {exc}")
        raise

    archive_path.unlink(missing_ok=True)
    logger.info("Archive removed after extraction.")

    wav_count = len(list(rir_dir.rglob("*.wav")))
    logger.info(f"Done. {wav_count} WAV files available in {rir_dir}")


if __name__ == "__main__":
    main()
