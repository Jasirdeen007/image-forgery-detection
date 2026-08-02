"""FastAPI service for fine-tuned document authenticity verification."""

import base64
from io import BytesIO
import os
from pathlib import Path
from threading import Lock

import cv2
from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from PIL import Image, UnidentifiedImageError
import numpy as np
from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.image import show_cam_on_image
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget
import torch

from ml_core.config import SETTINGS
from ml_core.gradcam import load_model, make_input


MODEL_CHECKPOINT = Path(os.getenv(
    "MODEL_CHECKPOINT", "artifacts/efficientnet_b3_finetuned.pt"
))
FORGERY_THRESHOLD = float(os.getenv("FORGERY_THRESHOLD", "0.5"))
MAX_UPLOAD_BYTES = int(os.getenv("MAX_UPLOAD_BYTES", str(10 * 1024 * 1024)))
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

if "finetuned" not in MODEL_CHECKPOINT.stem.lower():
    raise RuntimeError("MODEL_CHECKPOINT must be the Phase 2 fine-tuned model.")
if not MODEL_CHECKPOINT.is_file():
    raise FileNotFoundError(f"Fine-tuned checkpoint not found: {MODEL_CHECKPOINT}")

MODEL = load_model(MODEL_CHECKPOINT, DEVICE)
INFERENCE_LOCK = Lock()
app = FastAPI(
    title="Document Forgery Verification API",
    version="1.0.0",
    description="Binary genuine/forged classification with a Grad-CAM explanation.",
)
FRONTEND_DIR = Path(__file__).resolve().parent.parent / "frontend"
app.mount("/static", StaticFiles(directory=FRONTEND_DIR), name="static")


@app.get("/", include_in_schema=False)
def home() -> FileResponse:
    return FileResponse(FRONTEND_DIR / "index.html")


def encode_jpeg(image: np.ndarray) -> str:
    success, encoded = cv2.imencode(".jpg", cv2.cvtColor(image, cv2.COLOR_RGB2BGR))
    if not success:
        raise RuntimeError("Could not encode the Grad-CAM image.")
    value = base64.b64encode(encoded.tobytes()).decode("ascii")
    return f"data:image/jpeg;base64,{value}"


def predict_with_heatmap(image: Image.Image) -> tuple[dict, str]:
    resized = image.resize((SETTINGS.image_size, SETTINGS.image_size))
    input_tensor = make_input(resized).to(DEVICE)

    with INFERENCE_LOCK:
        with torch.no_grad():
            probabilities = torch.softmax(MODEL(input_tensor), dim=1)[0]
        forged_probability = float(probabilities[1].item())
        predicted_class = int(forged_probability >= FORGERY_THRESHOLD)

        with GradCAM(model=MODEL, target_layers=[MODEL.conv_head]) as cam:
            grayscale_cam = cam(
                input_tensor=input_tensor,
                targets=[ClassifierOutputTarget(predicted_class)],
            )[0]

    rgb_image = np.asarray(resized).astype(np.float32) / 255.0
    overlay = show_cam_on_image(rgb_image, grayscale_cam, use_rgb=True)
    result = {
        "label": "forged" if predicted_class else "genuine",
        "predicted_class": predicted_class,
        "genuine_probability": float(probabilities[0].item()),
        "forged_probability": forged_probability,
        "threshold": FORGERY_THRESHOLD,
        "device": str(DEVICE),
        "checkpoint": str(MODEL_CHECKPOINT),
    }
    return result, encode_jpeg(overlay)


@app.get("/health")
def health() -> dict:
    return {
        "status": "ok",
        "device": str(DEVICE),
        "cuda_available": torch.cuda.is_available(),
        "checkpoint": str(MODEL_CHECKPOINT),
        "model_loaded": True,
    }


@app.post("/verify")
async def verify(file: UploadFile = File(...)) -> dict:
    if file.content_type not in {"image/jpeg", "image/png", "image/webp", "image/bmp"}:
        raise HTTPException(status_code=415, detail="Upload a JPEG, PNG, WebP, or BMP image.")

    data = await file.read(MAX_UPLOAD_BYTES + 1)
    if len(data) > MAX_UPLOAD_BYTES:
        raise HTTPException(status_code=413, detail="Uploaded image exceeds the size limit.")

    try:
        image = Image.open(BytesIO(data)).convert("RGB")
    except (UnidentifiedImageError, OSError) as error:
        raise HTTPException(status_code=400, detail="The uploaded file is not a valid image.") from error

    result, heatmap = predict_with_heatmap(image)
    result.update({
        "filename": file.filename,
        "heatmap": heatmap,
    })
    return result
