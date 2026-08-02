"""Phase 4: test-set error analysis for the fine-tuned SIDTD classifier."""

import argparse
import csv
import json
from collections import Counter, defaultdict
from pathlib import Path

import cv2
from PIL import Image
import numpy as np
from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.image import show_cam_on_image
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget
import torch
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from torchvision import transforms

from ml_core.config import SETTINGS
from ml_core.dataset import DocumentSample, discover_samples, split_samples
from ml_core.gradcam import load_model


def annotation_for(sample: DocumentSample) -> dict:
    """Read the SIDTD JSON paired with a forged image, if one exists."""

    if sample.label == 0:
        return {}
    templates_root = sample.image_path.parent.parent.parent
    annotation_path = templates_root / "Annotations" / sample.image_path.parent.name / f"{sample.image_path.stem}.json"
    if not annotation_path.is_file():
        return {}
    return json.loads(annotation_path.read_text())


def subtype_for(sample: DocumentSample, annotation: dict) -> str:
    if sample.label == 0:
        return "genuine"
    ctype = annotation.get("ctype", "unknown")
    return {
        "Crop_and_Replace": "crop_and_replace",
        "Inpaint_and_Rewrite": "inpainting",
    }.get(ctype, ctype.lower().replace(" ", "_"))


def make_input(image: Image.Image) -> torch.Tensor:
    transform = transforms.Compose([
        transforms.Resize((SETTINGS.image_size, SETTINGS.image_size)),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=(0.485, 0.456, 0.406),
            std=(0.229, 0.224, 0.225),
        ),
    ])
    return transform(image).unsqueeze(0)


def metrics_for(records: list[dict], threshold: float) -> dict:
    labels = [record["label"] for record in records]
    probabilities = [record["forged_probability"] for record in records]
    predictions = [int(value >= threshold) for value in probabilities]
    result = {
        "samples": len(records),
        "accuracy": accuracy_score(labels, predictions),
        "precision": precision_score(labels, predictions, zero_division=0),
        "recall": recall_score(labels, predictions, zero_division=0),
        "f1": f1_score(labels, predictions, zero_division=0),
        "confusion_matrix": confusion_matrix(labels, predictions, labels=[0, 1]).tolist(),
    }
    if len(set(labels)) == 2:
        result["roc_auc"] = roc_auc_score(labels, probabilities)
    return result


def generate_cam(model, record: dict, device: torch.device, output_path: Path) -> None:
    image = Image.open(record["image_path"]).convert("RGB")
    resized = image.resize((SETTINGS.image_size, SETTINGS.image_size))
    input_tensor = make_input(resized).to(device)
    predicted_class = record["prediction"]
    with GradCAM(model=model, target_layers=[model.conv_head]) as cam:
        grayscale_cam = cam(
            input_tensor=input_tensor,
            targets=[ClassifierOutputTarget(predicted_class)],
        )[0]
    rgb_image = np.asarray(resized).astype(np.float32) / 255.0
    overlay = show_cam_on_image(rgb_image, grayscale_cam, use_rgb=True)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(cv2.cvtColor(overlay, cv2.COLOR_RGB2BGR)).save(output_path)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", type=Path, default=SETTINGS.data_root)
    parser.add_argument("--checkpoint", type=Path, default=Path("artifacts/efficientnet_b3_finetuned.pt"))
    parser.add_argument("--output-dir", type=Path, default=Path("artifacts/error_analysis"))
    parser.add_argument("--threshold", type=float, default=0.5)
    parser.add_argument("--max-error-cams", type=int, default=20)
    args = parser.parse_args()

    if "finetuned" not in args.checkpoint.stem.lower():
        raise ValueError("Phase 4 requires a fine-tuned checkpoint filename containing 'finetuned'.")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = load_model(args.checkpoint, device)
    samples = discover_samples(args.data_root)
    _, _, test_samples = split_samples(samples, SETTINGS.random_seed)
    records = []

    for sample in test_samples:
        annotation = annotation_for(sample)
        image = Image.open(sample.image_path).convert("RGB")
        input_tensor = make_input(image).to(device)
        with torch.no_grad():
            probabilities = torch.softmax(model(input_tensor), dim=1)[0]
        forged_probability = float(probabilities[1].item())
        prediction = int(forged_probability >= args.threshold)
        records.append({
            "image_path": str(sample.image_path),
            "label": sample.label,
            "actual_label": "forged" if sample.label else "genuine",
            "prediction": prediction,
            "predicted_label": "forged" if prediction else "genuine",
            "forged_probability": forged_probability,
            "threshold": args.threshold,
            "correct": prediction == sample.label,
            "error_type": (
                "false_positive" if sample.label == 0 and prediction == 1
                else "false_negative" if sample.label == 1 and prediction == 0
                else "correct"
            ),
            "subtype": subtype_for(sample, annotation),
            "ctype": annotation.get("ctype", ""),
            "field": annotation.get("field", ""),
            "source": annotation.get("src", ""),
            "second_source": annotation.get("second_src", ""),
        })

    args.output_dir.mkdir(parents=True, exist_ok=True)
    fields = list(records[0].keys())
    with (args.output_dir / "predictions.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(records)

    by_subtype = defaultdict(list)
    for record in records:
        # Subtype analysis is defined only for forged samples. Genuine images
        # have no forgery subtype and would make positive-class metrics invalid.
        if record["label"] == 1:
            by_subtype[record["subtype"]].append(record)
    summary = {
        "checkpoint": str(args.checkpoint),
        "device": str(device),
        "threshold": args.threshold,
        "test_metrics": metrics_for(records, args.threshold),
        "subtype_metrics": {
            subtype: metrics_for(subtype_records, args.threshold)
            for subtype, subtype_records in sorted(by_subtype.items())
        },
        "error_counts": dict(Counter(record["error_type"] for record in records)),
        "annotation_counts": dict(Counter(record["ctype"] for record in records if record["ctype"])),
    }
    (args.output_dir / "summary.json").write_text(json.dumps(summary, indent=2))

    errors = [record for record in records if record["error_type"] != "correct"]
    errors.sort(key=lambda record: abs(record["forged_probability"] - 0.5))
    for index, record in enumerate(errors[:args.max_error_cams], start=1):
        output_path = args.output_dir / "gradcam_errors" / f"{index:03d}_{record['error_type']}.jpg"
        generate_cam(model, record, device, output_path)

    print(json.dumps(summary, indent=2))
    print(f"Saved predictions to {args.output_dir / 'predictions.csv'}")
    print(f"Saved summary to {args.output_dir / 'summary.json'}")


if __name__ == "__main__":
    main()
