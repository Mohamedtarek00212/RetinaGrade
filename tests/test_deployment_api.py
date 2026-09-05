"""Tests for the deployment API without loading the production model."""

from pathlib import Path

import cv2
import numpy as np
import pytest
from fastapi.testclient import TestClient

from deployment.api import APISettings, create_app
from deployment.inference import Prediction


class _FakePredictor:
    device = "cpu"

    def predict(self, image_path: Path) -> Prediction:
        assert image_path.exists()
        return Prediction(
            grade=2,
            label="Moderate DR",
            confidence=0.8,
            probabilities=[0.02, 0.03, 0.8, 0.1, 0.05],
            ordinal_probabilities=[0.95, 0.85, 0.2, 0.05],
        )


def _fundus_bytes(size: int = 256) -> bytes:
    image = np.zeros((size, size, 3), dtype=np.uint8)
    cv2.circle(image, (size // 2, size // 2), size // 2 - 8, (20, 80, 160), -1)
    success, encoded = cv2.imencode(".png", image)
    assert success
    return encoded.tobytes()


def _client(max_upload_bytes: int = 1024 * 1024, api_prefix: str = "") -> TestClient:
    settings = APISettings(max_upload_bytes=max_upload_bytes)
    return TestClient(
        create_app(settings=settings, predictor=_FakePredictor(), api_prefix=api_prefix)
    )


def test_health_reports_loaded_model() -> None:
    with _client() as client:
        response = client.get("/health")

    assert response.status_code == 200
    assert response.json() == {"status": "ready", "model_loaded": True, "device": "cpu"}


def test_api_prefix_supports_single_container_deployment() -> None:
    with _client(api_prefix="/api/") as client:
        response = client.get("/api/health")

    assert response.status_code == 200
    assert response.json()["model_loaded"] is True


def test_predict_returns_structured_result() -> None:
    with _client() as client:
        response = client.post(
            "/predict",
            files={"image": ("fundus.png", _fundus_bytes(), "image/png")},
        )

    assert response.status_code == 200
    assert response.json()["grade"] == 2
    assert response.json()["label"] == "Moderate DR"
    assert response.json()["assessment_status"] == "conclusive"
    assert response.json()["confidence_margin"] == pytest.approx(0.7)
    assert response.json()["quality"]["acceptable"] is True
    assert response.json()["inference_ms"] >= 0


def test_predict_rejects_unsupported_file_type() -> None:
    with _client() as client:
        response = client.post("/predict", files={"image": ("notes.txt", b"text", "text/plain")})

    assert response.status_code == 415


def test_predict_rejects_oversized_upload() -> None:
    with _client(max_upload_bytes=16) as client:
        response = client.post("/predict", files={"image": ("fundus.png", b"x" * 17, "image/png")})

    assert response.status_code == 413


def test_predict_rejects_low_resolution_image() -> None:
    with _client() as client:
        response = client.post(
            "/predict",
            files={"image": ("small.png", _fundus_bytes(size=128), "image/png")},
        )

    assert response.status_code == 422
    assert "224 px minimum" in response.json()["detail"]
