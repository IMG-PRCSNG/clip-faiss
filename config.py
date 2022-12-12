from pathlib import Path
from pydantic import BaseSettings


class APIConfig(BaseSettings):
    images_dir: Path = Path("public/images")
    thumbs_dir: Path = Path("public/thumbs")
    dataset: Path = Path("dataset.h5")
    top_k: int = 10
    precision: int = 3
