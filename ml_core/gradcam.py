"""Generate Grad-CAM explanations for an EfficientNet-B3 prediction."""

import argparse
import json
from pathlib import Path

import cv2
import numpy as np
from PIL import Image
import timm
import torch
from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.image import show_cam_on_image
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget
from torchvision import transforms

from ml_core.config import SETTINGS


def load_model(checkpoint: Path, device: torch.device):
    """Load either the Phase 1 or Phase 2 EfficientNet checkpoint."""

    model = timm.create_model("efficientnet_b3", pretrained=False, num_classes=2)
    state_dict = torch.load(checkpoint, map_location="cpu", weights_only=True)
    model.load_state_dict(state_dict)
    model.to(device)
    model.eval()
    return model


def make_input(image: Image.Image):
    normalize = transforms.Normalize(
        mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)
    )
    transform = transforms.Compose([
        transforms.Resize((SETTINGS.image_size, SETTINGS.image_size)),
        transforms.ToTensor(),
        normalize,
    ])
    return transform(image).unsqueeze(0)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--image", type=Path, required=True)
    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=Path("artifacts/efficientnet_b3_finetuned.pt"),
        help="Phase 2 fine-tuned checkpoint; other checkpoint names are rejected.",
    )
    parser.add_argument("--output", type=Path, default=Path("artifacts/gradcam_overlay.jpg"))
    parser.add_argument("--metadata-output", type=Path, default=Path("artifacts/gradcam_prediction.json"))
    parser.add_argument("--threshold", type=float, default=0.5,
                        help="Forged decision threshold; use the Phase 2 validation threshold when available.")
    args = parser.parse_args()

    if not args.image.is_file():
        raise FileNotFoundError(f"Image not found: {args.image}")
    if not args.checkpoint.is_file():
        raise FileNotFoundError(f"Checkpoint not found: {args.checkpoint}")
    if "finetuned" not in args.checkpoint.stem.lower():
        raise ValueError(
            "Grad-CAM is restricted to the Phase 2 fine-tuned checkpoint. "
            "Use a checkpoint filename containing 'finetuned'."
        )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    original = Image.open(args.image).convert("RGB")
    resized = original.resize((SETTINGS.image_size, SETTINGS.image_size))
    input_tensor = make_input(resized).to(device)
    model = load_model(args.checkpoint, device)

    with torch.no_grad():
        probabilities = torch.softmax(model(input_tensor), dim=1)[0]
    forged_probability = float(probabilities[1].item())
    predicted_class = int(forged_probability >= args.threshold)

    # conv_head is the final spatial convolution before global pooling.
    target_layers = [model.conv_head]
    with GradCAM(model=model, target_layers=target_layers) as cam:
        grayscale_cam = cam(
            input_tensor=input_tensor,
            targets=[ClassifierOutputTarget(predicted_class)],
        )[0]

    rgb_image = np.asarray(resized).astype(np.float32) / 255.0
    overlay = show_cam_on_image(rgb_image, grayscale_cam, use_rgb=True)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.metadata_output.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(cv2.cvtColor(overlay, cv2.COLOR_RGB2BGR)).save(args.output)

    metadata = {
        "image": str(args.image),
        "checkpoint": str(args.checkpoint),
        "device": str(device),
        "label": "forged" if predicted_class else "genuine",
        "predicted_class": predicted_class,
        "genuine_probability": float(probabilities[0].item()),
        "forged_probability": forged_probability,
        "threshold": args.threshold,
        "heatmap_target": "forged" if predicted_class else "genuine",
    }
    args.metadata_output.write_text(json.dumps(metadata, indent=2))
    print(json.dumps(metadata))


if __name__ == "__main__":
    main()
