"""Phase 2: fine-tune the final EfficientNet-B3 block on SIDTD templates.

This script starts from the Phase 1 checkpoint instead of ImageNet weights.
The early feature blocks remain frozen because the dataset is small; only the
last feature block, projection layers, and binary classifier are adapted.
"""

import argparse
import json
from pathlib import Path
import random

import numpy as np
import timm
import torch
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from torch import nn
from torch.utils.data import DataLoader

from ml_core.config import SETTINGS
from ml_core.dataset import DocumentDataset, discover_samples, split_samples
from ml_core.train import make_transforms


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def build_finetune_model(checkpoint: Path) -> nn.Module:
    """Load Phase 1 weights and unfreeze the final EfficientNet feature stage."""

    model = timm.create_model("efficientnet_b3", pretrained=False, num_classes=2)
    state_dict = torch.load(checkpoint, map_location="cpu", weights_only=True)
    model.load_state_dict(state_dict)

    for parameter in model.parameters():
        parameter.requires_grad = False

    # Adapt the highest-level visual features and the binary decision head.
    for parameter in model.blocks[-1].parameters():
        parameter.requires_grad = True
    for layer in (model.conv_head, model.bn2, model.get_classifier()):
        for parameter in layer.parameters():
            parameter.requires_grad = True
    return model


def calculate_metrics(labels: list[int], probabilities: list[float], threshold: float) -> dict:
    predictions = [int(probability >= threshold) for probability in probabilities]
    result = {
        "threshold": threshold,
        "accuracy": accuracy_score(labels, predictions),
        "precision": precision_score(labels, predictions, zero_division=0),
        "recall": recall_score(labels, predictions, zero_division=0),
        "f1": f1_score(labels, predictions, zero_division=0),
        "confusion_matrix": confusion_matrix(labels, predictions).tolist(),
    }
    if len(set(labels)) == 2:
        result["roc_auc"] = roc_auc_score(labels, probabilities)
        result["pr_auc"] = average_precision_score(labels, probabilities)
    return result


def select_threshold(labels: list[int], probabilities: list[float]) -> dict:
    """Choose the validation threshold with the highest F1 score."""

    candidates = [round(value / 100, 2) for value in range(20, 81)]
    results = [calculate_metrics(labels, probabilities, threshold) for threshold in candidates]
    return max(results, key=lambda result: (result["f1"], result["precision"]))


def run_epoch(model, loader, criterion, optimizer, device, training: bool):
    model.train(training)
    total_loss = 0.0
    labels, probabilities = [], []
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
            probabilities.extend(torch.softmax(logits, dim=1)[:, 1].cpu().tolist())

    return labels, probabilities, total_loss / len(loader.dataset)


def make_loaders(samples, batch_size: int):
    train_samples, val_samples, test_samples = split_samples(samples, SETTINGS.random_seed)
    train_transform, eval_transform = make_transforms(SETTINGS.image_size)
    loaders = {
        "train": DataLoader(DocumentDataset(train_samples, train_transform), batch_size, shuffle=True),
        "val": DataLoader(DocumentDataset(val_samples, eval_transform), batch_size),
        "test": DataLoader(DocumentDataset(test_samples, eval_transform), batch_size),
    }
    return loaders


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", type=Path, default=SETTINGS.data_root)
    parser.add_argument("--checkpoint", type=Path, default=Path("efficientnet_b3_best.pt"))
    parser.add_argument("--output", type=Path, default=Path("artifacts/efficientnet_b3_finetuned.pt"))
    parser.add_argument("--history-output", type=Path, default=Path("artifacts/phase2_history.json"))
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--learning-rate", type=float, default=1e-4)
    parser.add_argument("--limit-per-class", type=int, default=None)
    parser.add_argument("--weighted-loss", action="store_true")
    args = parser.parse_args()

    set_seed(SETTINGS.random_seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    samples = discover_samples(args.data_root)
    if args.limit_per_class is not None:
        samples = [
            sample for label in (0, 1)
            for sample in [s for s in samples if s.label == label][:args.limit_per_class]
        ]

    loaders = make_loaders(samples, args.batch_size)
    model = build_finetune_model(args.checkpoint).to(device)
    trainable_parameters = [parameter for parameter in model.parameters() if parameter.requires_grad]
    optimizer = torch.optim.AdamW(trainable_parameters, lr=args.learning_rate, weight_decay=1e-4)

    if args.weighted_loss:
        train_labels = [sample.label for sample in split_samples(samples, SETTINGS.random_seed)[0]]
        counts = torch.bincount(torch.tensor(train_labels), minlength=2).float()
        weights = (counts.sum() / (2 * counts)).to(device)
        criterion = nn.CrossEntropyLoss(weight=weights)
    else:
        criterion = nn.CrossEntropyLoss()

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.history_output.parent.mkdir(parents=True, exist_ok=True)
    best_f1 = -1.0
    best_threshold = 0.5
    history = []

    print(f"device={device}; trainable_parameters={sum(p.numel() for p in trainable_parameters)}")
    for epoch in range(1, args.epochs + 1):
        train_labels, train_probs, train_loss = run_epoch(
            model, loaders["train"], criterion, optimizer, device, True
        )
        val_labels, val_probs, val_loss = run_epoch(
            model, loaders["val"], criterion, optimizer, device, False
        )
        val_result = select_threshold(val_labels, val_probs)
        record = {
            "epoch": epoch,
            "train_loss": train_loss,
            "val_loss": val_loss,
            "val": val_result,
        }
        history.append(record)
        print(json.dumps(record))

        if val_result["f1"] > best_f1:
            best_f1 = val_result["f1"]
            best_threshold = val_result["threshold"]
            torch.save(model.state_dict(), args.output)

    model.load_state_dict(torch.load(args.output, map_location=device, weights_only=True))
    test_labels, test_probs, test_loss = run_epoch(
        model, loaders["test"], criterion, optimizer, device, False
    )
    test_result = calculate_metrics(test_labels, test_probs, best_threshold)
    test_result["loss"] = test_loss
    test_default_threshold = calculate_metrics(test_labels, test_probs, 0.5)
    summary = {
        "checkpoint": str(args.checkpoint),
        "output": str(args.output),
        "device": str(device),
        "weighted_loss": args.weighted_loss,
        "best_threshold": best_threshold,
        "history": history,
        "test": test_result,
        "test_default_threshold": test_default_threshold,
    }
    args.history_output.write_text(json.dumps(summary, indent=2))
    print(json.dumps({"test": test_result}))


if __name__ == "__main__":
    main()
