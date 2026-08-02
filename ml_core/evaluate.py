"""Evaluate the saved Phase 1 checkpoint on the deterministic test split."""

import argparse
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from ml_core.config import SETTINGS
from ml_core.dataset import DocumentDataset, discover_samples, split_samples
from ml_core.train import build_model, make_transforms, metrics


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", type=Path, default=SETTINGS.data_root)
    parser.add_argument("--checkpoint", type=Path, default=SETTINGS.output_dir / "efficientnet_b3_best.pt")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    _, eval_transform = make_transforms(SETTINGS.image_size)
    _, _, test_samples = split_samples(discover_samples(args.data_root), SETTINGS.random_seed)
    loader = DataLoader(DocumentDataset(test_samples, eval_transform), SETTINGS.batch_size)

    model = build_model().to(device)
    model.load_state_dict(torch.load(args.checkpoint, map_location=device))
    model.eval()
    labels, predictions = [], []
    with torch.no_grad():
        for images, targets in loader:
            predictions.extend(model(images.to(device)).argmax(dim=1).cpu().tolist())
            labels.extend(targets.tolist())

    print(metrics(labels, predictions))


if __name__ == "__main__":
    main()
