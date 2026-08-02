"""SIDTD template-image discovery, splitting, and PyTorch dataset wrapper."""

from dataclasses import dataclass
from pathlib import Path
import random

from PIL import Image
from sklearn.model_selection import train_test_split
from torch.utils.data import Dataset


IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}


@dataclass(frozen=True)
class DocumentSample:
    """One image and its binary label: 0 genuine, 1 forged."""

    image_path: Path
    label: int
    forgery_type: str | None = None


def _find_class_dirs(data_root: Path, class_name: str) -> list[Path]:
    template_dirs = [
        path for path in data_root.rglob("*")
        if path.is_dir()
        and path.name.lower() == class_name
        and path.parent.name.lower() == "images"
        and path.parent.parent.name.lower() == "templates"
    ]
    if template_dirs:
        return template_dirs

    # Also allow a deliberately prepared root containing reals/ and fakes/.
    return [path for path in data_root.iterdir() if path.is_dir() and path.name.lower() == class_name]


def discover_samples(data_root: Path) -> list[DocumentSample]:
    """Discover images below SIDTD ``Images/reals`` and ``Images/fakes``."""

    if not data_root.exists():
        raise FileNotFoundError(f"Data root does not exist: {data_root}")

    samples: list[DocumentSample] = []
    for directory_name, label in (("reals", 0), ("fakes", 1)):
        directories = _find_class_dirs(data_root, directory_name)
        for directory in directories:
            for image_path in sorted(directory.iterdir()):
                if image_path.is_file() and image_path.suffix.lower() in IMAGE_EXTENSIONS:
                    samples.append(DocumentSample(image_path, label))

    if not samples:
        raise FileNotFoundError(
            f"No images found below {data_root}. Expected templates/Images/reals and fakes."
        )

    if {sample.label for sample in samples} != {0, 1}:
        raise ValueError("Both genuine (reals) and forged (fakes) images are required.")

    return samples


def split_samples(
    samples: list[DocumentSample], seed: int = 42
) -> tuple[list[DocumentSample], list[DocumentSample], list[DocumentSample]]:
    """Create the README-described stratified 80/10/10 hold-out split."""

    labels = [sample.label for sample in samples]
    train, remainder = train_test_split(
        samples, test_size=0.2, random_state=seed, stratify=labels
    )
    remainder_labels = [sample.label for sample in remainder]
    val, test = train_test_split(
        remainder, test_size=0.5, random_state=seed, stratify=remainder_labels
    )
    random.Random(seed).shuffle(train)
    return train, val, test


class DocumentDataset(Dataset):
    """Loads RGB document images and applies a torchvision transform."""

    def __init__(self, samples: list[DocumentSample], transform=None):
        self.samples = samples
        self.transform = transform

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int):
        sample = self.samples[index]
        with Image.open(sample.image_path) as image:
            image = image.convert("RGB")
        if self.transform is not None:
            image = self.transform(image)
        return image, sample.label
