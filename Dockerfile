FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1

WORKDIR /app

# Runtime libraries required by OpenCV's headless image processing path.
RUN apt-get update \
    && apt-get install -y --no-install-recommends libglib2.0-0 libgl1 \
    && rm -rf /var/lib/apt/lists/*

# Keep the container portable: the default serving image uses CPU PyTorch.
RUN python -m pip install --upgrade pip \
    && python -m pip install torch==2.9.1 torchvision==0.24.1 \
       --index-url https://download.pytorch.org/whl/cpu

COPY requirements-docker.txt .
RUN python -m pip install -r requirements-docker.txt

COPY backend ./backend
COPY ml_core ./ml_core
COPY frontend ./frontend
RUN mkdir -p /app/artifacts

ENV MODEL_CHECKPOINT=/app/artifacts/efficientnet_b3_finetuned.pt \
    FORGERY_THRESHOLD=0.5 \
    MAX_UPLOAD_BYTES=10485760

EXPOSE 8000

CMD ["uvicorn", "backend.serve:app", "--host", "0.0.0.0", "--port", "8000"]
