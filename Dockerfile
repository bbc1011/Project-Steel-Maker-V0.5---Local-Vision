FROM python:3.13-slim AS builder

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    PATH="/opt/venv/bin:$PATH"

WORKDIR /app

RUN apt-get update \
    && apt-get install -y --no-install-recommends git ca-certificates \
    && rm -rf /var/lib/apt/lists/*

RUN python -m venv /opt/venv

COPY requirements.txt .
RUN pip install --upgrade pip setuptools wheel \
    && pip install --no-cache-dir -r requirements.txt

# Copy only the files needed to preload the local detector first.
COPY config.py local_detector.py preload_models.py ./

# Use defaults at build time; Railway secrets are not required to download
# the public model assets.
RUN DETECTOR_BACKEND=yolo_world \
    YOLO_WORLD_MODEL=yolov8s-worldv2.pt \
    YOLO_WORLD_CLASSES="LEGO minifigure;LEGO minifig" \
    YOLO_DEVICE=cpu \
    python preload_models.py


FROM python:3.13-slim AS runtime

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PATH="/opt/venv/bin:$PATH"

WORKDIR /app

COPY --from=builder /opt/venv /opt/venv

# Preserve model/text-encoder caches downloaded during the build.
COPY --from=builder /root/.cache /root/.cache
COPY --from=builder /app/yolov8s-worldv2.pt /app/yolov8s-worldv2.pt

COPY . .

CMD ["python", "main.py"]
