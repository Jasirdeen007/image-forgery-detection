# Explainable Document Forgery Detection

An end-to-end computer-vision application for classifying synthetic identity-document images as genuine or forged. The application uses a fine-tuned EfficientNet-B3 model and returns a Grad-CAM heatmap showing which image regions influenced the prediction.

> This is a portfolio and research-prototype project. It is not a production KYC decision system and must not be used as the sole basis for identity verification.

## What the application does

```text
uploaded document image
        ↓
RGB conversion and 300×300 resize
        ↓
fine-tuned EfficientNet-B3
        ↓
genuine/forged probabilities
        ↓
Grad-CAM explanation
        ↓
browser UI or FastAPI /verify response
```

The prediction target is binary:

```text
0 = genuine / bona fide
1 = forged / tampered
```

SIDTD subtype metadata is used during error analysis for crop-and-replace and inpaint-and-rewrite.

## Repository structure

```text
document_forgery/
├── backend/
│   ├── __init__.py
│   └── serve.py                 # FastAPI API and website entry point
├── frontend/
│   ├── index.html               # Upload UI
│   ├── styles.css
│   └── app.js
├── ml_core/
│   ├── __init__.py
│   ├── config.py                # Paths and hyperparameters
│   ├── dataset.py               # Discovery, split, and PyTorch dataset
│   ├── extract_sidtd.py         # Safe ZIP extraction
│   ├── train.py                 # Phase 1 frozen-backbone baseline
│   ├── fine_tune.py             # Phase 2 fine-tuning
│   ├── evaluate.py              # Test-set evaluation
│   ├── gradcam.py               # Single-image Grad-CAM
│   └── error_analysis.py        # Phase 4 report and error heatmaps
├── artifacts/                   # Local only: checkpoint and reports
├── data/                        # Local only: SIDTD images/annotations
├── Dockerfile
├── docker-compose.yml
├── requirements.txt
└── requirements-docker.txt
```

## Dataset

The project uses the SIDTD template/composite dataset. Download the `templates.zip` archive from the official SIDTD source, then extract it:

```powershell
python -m ml_core.extract_sidtd --archive templates.zip --output data
```

Expected layout:

```text
data/
└── templates/
    ├── Images/
    │   ├── reals/
    │   └── fakes/
    └── Annotations/
        ├── reals/
        └── fakes/
```

The dataset, archive, and checkpoints are intentionally excluded from Git by `.gitignore`.

## Local setup

```powershell
python -m venv env
.\env\Scripts\Activate.ps1
python -m pip install -r requirements.txt
```

For NVIDIA CUDA execution, install the CUDA-enabled PyTorch build appropriate for the machine before installing the remaining requirements. Confirm it with:

```powershell
python -c "import torch; print(torch.cuda.is_available()); print(torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU')"
```

## Training workflow

Run commands from the repository root using module notation.

### Phase 1 baseline

```powershell
python -m ml_core.train --data-root data --epochs 10 --batch-size 16
```

### Phase 2 fine-tuning

Place the Phase 1 checkpoint at `efficientnet_b3_best.pt` or provide its path:

```powershell
python -m ml_core.fine_tune `
  --data-root data `
  --checkpoint efficientnet_b3_best.pt `
  --epochs 20 `
  --batch-size 8 `
  --output artifacts/efficientnet_b3_finetuned.pt `
  --history-output artifacts/phase2_history.json
```

### Evaluation and analysis

```powershell
python -m ml_core.evaluate --data-root data --checkpoint artifacts/efficientnet_b3_finetuned.pt
python -m ml_core.gradcam --image data/templates/Images/fakes/alb_id_00_fake_6_25.jpg --checkpoint artifacts/efficientnet_b3_finetuned.pt
python -m ml_core.error_analysis --data-root data --checkpoint artifacts/efficientnet_b3_finetuned.pt --output-dir artifacts/error_analysis --threshold 0.5 --max-error-cams 20
```

Fine-tuning unfreezes the final EfficientNet feature block and classifier, selects a validation threshold, and reports test metrics at both the default and tuned thresholds. Error analysis creates per-image predictions, subtype metrics, false-positive/false-negative counts, and Grad-CAM images for errors.

## Run the website locally

```powershell
.\env\Scripts\Activate.ps1
python -m uvicorn backend.serve:app --host 127.0.0.1 --port 8001
```

Open the website at `http://127.0.0.1:8001/`.

Available endpoints:

```text
GET  /health
POST /verify
GET  /docs
```

The `/verify` endpoint accepts a multipart form field named `file` and returns the label, probabilities, device, checkpoint path, and a base64-encoded Grad-CAM JPEG.

## Run with Docker

Start Docker Desktop, then run:

```powershell
docker compose up --build -d
```

The host port is `8001` to avoid conflicts with other local services:

```text
Website: http://localhost:8001/
Swagger: http://localhost:8001/docs
Health:  http://localhost:8001/health
```

Ensure `artifacts/efficientnet_b3_finetuned.pt` exists before starting. View logs and stop the service with:

```powershell
docker compose logs -f document-forgery-api
docker compose down
```

The default Docker image uses CPU PyTorch for portability. Native execution can use an NVIDIA GPU when the environment has a CUDA-enabled PyTorch installation.

## Current validation result

On the current fixed test split, the fine-tuned checkpoint achieved approximately:

```text
Accuracy: 98.21%
Precision: 97.60%
Recall: 99.19%
F1: 98.39%
ROC-AUC: 99.98%
```

Confusion matrix:

```text
[[97, 3],
 [1, 122]]
```

These results are dataset-specific and should not be treated as production performance. The dataset is synthetic, and image-level random splits can be easier than independent real-world identity-document evaluation.

## Limitations and future work

- SIDTD is synthetic and may not represent real capture conditions.
- The current target is binary authenticity, not document identity verification.
- Grad-CAM is an explanation aid, not a tamper-proof localization method.
- Thresholds should be calibrated on a deployment-specific validation set.
- Future work includes stronger group/template splits, external validation, model calibration, and review workflows for uncertain predictions.

## Dataset reference

SIDTD is a synthetic dataset of ID and travel documents generated from MIDV-2020 for identity-document verification research. See the [SIDTD project](https://github.com/Oriolrt/SIDTD_Dataset) and dataset documentation before redistributing data.

## Pretrained model

Download the fine-tuned EfficientNet-B3 checkpoint from Hugging Face:

[Download `efficientnet_b3_finetuned.pt`](https://huggingface.co/Jasirdeen/efficientnet-b3-image-forgery-detection/resolve/main/efficientnet_b3_finetuned.pt)
