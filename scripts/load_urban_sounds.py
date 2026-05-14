"""Download UrbanSound8K via soundata into data/input/.

soundata expects the dataset at:
    data/input/UrbanSound8K/audio/fold{1..10}/
    data/input/UrbanSound8K/metadata/UrbanSound8K.csv

Run from the project root:
    python -m scripts.load_urban_sounds
"""

import logging

import soundata

from src.config import load_config

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
)
logger = logging.getLogger(__name__)


def main() -> None:
    """Download and validate the UrbanSound8K dataset."""
    cfg = load_config("config.yaml")
    data_home = str(__import__("pathlib").Path(cfg["paths"]["urban_sound"]).parent)

    logger.info(f"Initialising UrbanSound8K dataset (data_home={data_home})")
    dataset = soundata.initialize("urbansound8k", data_home=data_home)

    logger.info("Downloading UrbanSound8K (~6 GB) from Zenodo …")
    dataset.download()

    logger.info("Validating dataset …")
    dataset.validate()

    logger.info("UrbanSound8K ready.")


if __name__ == "__main__":
    main()
