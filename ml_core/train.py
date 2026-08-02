"""Phase 1: frozen EfficientNet-B3 SIDTD template classifier."""

import argparse
import json
from pathlib import Path
import random

import numpy as np
import timm
import torch
from sklearn.metrics import confusion_matrix, f1_score, precision_score, recall_score
from torch import nn
from torch.utils.data import DataLoader
from torchvision import transforms

from ml_core.config import SETTINGS
from ml_core.dataset import DocumentDataset, discover_samples, split_samples


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def make_transforms(image_size: int):
    normalize = transforms.Normalize(
        mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)
    )
    train_transform = transforms.Compose([
        transforms.Resize((image_size, image_size)),
        # Preserve document geometry; mirrored IDs are not realistic examples.
        transforms.ColorJitter(brightness=0.15, contrast=0.15, saturation=0.05),
        transforms.ToTensor(),
        normalize,
    ])
    eval_transform = transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.ToTensor(),
        normalize,
    ])
    return train_transform, eval_transform


def build_model() -> nn.Module:
    model = timm.create_model("efficientnet_b3", pretrained=True, num_classes=2)
    #Freeze all layers except the classifier head.
    for parameter in model.parameters():
        parameter.requires_grad = False
    for parameter in model.get_classifier().parameters():
        parameter.requires_grad = True
    return model


def metrics(labels: list[int], predictions: list[int]) -> dict:
    return {
        "precision": precision_score(labels, predictions, zero_division=0),
        "recall": recall_score(labels, predictions, zero_division=0),
        "f1": f1_score(labels, predictions, zero_division=0),
        "confusion_matrix": confusion_matrix(labels, predictions).tolist(),
    }


def run_epoch(model, loader, criterion, optimizer, device, training: bool):
    model.train(training)
    total_loss = 0.0
    labels, predictions = [], []
    context = torch.enable_grad() if training else torch.no_grad()
    with context:
        for images, targets in loader:
            images, targets = images.to(device), targets.to(device)
            logits = model(images)
            loss = criterion(logits, targets)
            if training:
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
            total_loss += loss.item() * len(targets)
            labels.extend(targets.cpu().tolist())
            predictions.extend(logits.argmax(dim=1).cpu().tolist())
    result = metrics(labels, predictions)
    result["loss"] = total_loss / len(loader.dataset)
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", type=Path, default=SETTINGS.data_root)
    parser.add_argument("--epochs", type=int, default=SETTINGS.epochs)
    parser.add_argument("--batch-size", type=int, default=SETTINGS.batch_size)
    parser.add_argument("--limit-per-class", type=int, default=None,
                        help="Optional balanced smoke-test limit; default uses all images.")
    args = parser.parse_args()

    set_seed(SETTINGS.random_seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    samples = discover_samples(args.data_root)
    if args.limit_per_class is not None:
        samples = [
            sample for label in (0, 1)
            for sample in [s for s in samples if s.label == label][:args.limit_per_class]
        ]
    train_samples, val_samples, test_samples = split_samples(samples, SETTINGS.random_seed)
    train_transform, eval_transform = make_transforms(SETTINGS.image_size)
    loaders = {
        "train": DataLoader(DocumentDataset(train_samples, train_transform), args.batch_size, shuffle=True),
        "val": DataLoader(DocumentDataset(val_samples, eval_transform), args.batch_size),
        "test": DataLoader(DocumentDataset(test_samples, eval_transform), args.batch_size),
    }

    model = build_model().to(device)
    optimizer = torch.optim.AdamW(model.get_classifier().parameters(), lr=SETTINGS.learning_rate)
    criterion = nn.CrossEntropyLoss()
    history = []
    best_f1 = -1.0
    SETTINGS.output_dir.mkdir(parents=True, exist_ok=True)

    for epoch in range(1, args.epochs + 1):
        train_result = run_epoch(model, loaders["train"], criterion, optimizer, device, True)
        val_result = run_epoch(model, loaders["val"], criterion, optimizer, device, False)
        record = {"epoch": epoch, "train": train_result, "val": val_result}
        history.append(record)
        print(json.dumps(record))
        if val_result["f1"] > best_f1:
            best_f1 = val_result["f1"]
            torch.save(model.state_dict(), SETTINGS.output_dir / "efficientnet_b3_best.pt")

    model.load_state_dict(torch.load(SETTINGS.output_dir / "efficientnet_b3_best.pt", map_location=device))
    test_result = run_epoch(model, loaders["test"], criterion, optimizer, device, False)
    print(json.dumps({"test": test_result}))
    (SETTINGS.output_dir / "history.json").write_text(json.dumps({"history": history, "test": test_result}, indent=2))


if __name__ == "__main__":
    main()
