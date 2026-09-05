# syntax=docker/dockerfile:1.7

FROM python:3.11-slim AS runtime-cpu

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    RETINAGRADE_DEVICE=auto \
    RETINAGRADE_CHECKPOINT=/models/model_inference.pt

WORKDIR /app

COPY pyproject.toml README.md deployment/requirements-runtime.txt ./
COPY src ./src
COPY deployment ./deployment
COPY configs ./configs
COPY Presentation/Metrics/normalization_stats.json ./Presentation/Metrics/normalization_stats.json

RUN python -m pip install --index-url https://download.pytorch.org/whl/cpu \
      "torch==2.5.1" "torchvision==0.20.1" \
    && python -m pip install -r requirements-runtime.txt \
    && python -m pip install --no-deps .

EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=5s --start-period=90s --retries=3 \
  CMD python -c "import urllib.request; urllib.request.urlopen('http://127.0.0.1:8000/health', timeout=3)"

CMD ["uvicorn", "deployment.api:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "1"]


FROM pytorch/pytorch:2.5.1-cuda12.4-cudnn9-runtime AS runtime-gpu

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    RETINAGRADE_DEVICE=auto \
    RETINAGRADE_CHECKPOINT=/models/model_inference.pt

WORKDIR /app

COPY pyproject.toml README.md deployment/requirements-runtime.txt ./
COPY src ./src
COPY deployment ./deployment
COPY configs ./configs
COPY Presentation/Metrics/normalization_stats.json ./Presentation/Metrics/normalization_stats.json

RUN python -m pip install -r requirements-runtime.txt \
    && python -m pip install --no-deps .

EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=5s --start-period=90s --retries=3 \
  CMD python -c "import urllib.request; urllib.request.urlopen('http://127.0.0.1:8000/health', timeout=3)"

CMD ["uvicorn", "deployment.api:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "1"]


FROM node:22-alpine AS frontend-build

WORKDIR /frontend

COPY frontend/package.json frontend/package-lock.json ./
RUN npm ci

COPY frontend ./
RUN npm run build


FROM runtime-cpu AS space-cpu

ENV RETINAGRADE_API_PREFIX=/api \
    RETINAGRADE_STATIC_DIR=/app/frontend-dist \
    RETINAGRADE_CHECKPOINT=/models/model_inference.pt

COPY --from=frontend-build /frontend/dist /app/frontend-dist
COPY outputs/checkpoints/deployment/model_inference.pt /models/model_inference.pt

EXPOSE 7860

HEALTHCHECK --interval=30s --timeout=5s --start-period=90s --retries=3 \
  CMD python -c "import urllib.request; urllib.request.urlopen('http://127.0.0.1:7860/api/health', timeout=3)"

CMD ["uvicorn", "deployment.api:app", "--host", "0.0.0.0", "--port", "7860", "--workers", "1"]
