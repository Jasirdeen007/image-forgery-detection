"""Central configuration for the document-forgery baseline.

Keep paths and training choices in one place so experiments do not depend on
machine-specific absolute paths embedded in training code.
"""

from dataclasses import dataclass
from pathlib import Path
import os


@dataclass(frozen=True)
class Settings:
    """Configuration shared by dataset, training, and evaluation modules."""

    # Set DOCUMENT_DATA_ROOT to the directory containing the downloaded data.
    data_root: Path = Path(os.getenv("DOCUMENT_DATA_ROOT", "data"))
    image_size: int = 300
    batch_size: int = 16
    epochs: int = 10
    learning_rate: float = 1e-3
    random_seed: int = 42
    output_dir: Path = Path("artifacts")


SETTINGS = Settings()
