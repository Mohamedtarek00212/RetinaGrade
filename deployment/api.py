"""FastAPI application serving the RetinaGrade inference model."""

from __future__ import annotations

import asyncio
import os
import tempfile
import time
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Annotated

from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.concurrency import run_in_threadpool
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from deployment.inference import RetinaGradePredictor
from deployment.quality import ImageQualityReport, assess_image_quality

PROJECT_ROOT = Path(__file__).resolve().parents[1]
ALLOWED_IMAGE_TYPES = {"image/jpeg": ".jpg", "image/png": ".png"}


@dataclass(frozen=True)
class APISettings:
    checkpoint: Path = PROJECT_ROOT / "outputs/checkpoints/deployment/model_inference.pt"
    data_config: Path = PROJECT_ROOT / "configs/data.yaml"
    model_config: Path = PROJECT_ROOT / "configs/model.yaml"
    normalization_stats: Path = PROJECT_ROOT / "Presentation/Metrics/normalization_stats.json"
    device: str = "auto"
    max_upload_bytes: int = 15 * 1024 * 1024
    allowed_origins: tuple[str, ...] = (
        "http://localhost:5173",
        "http://127.0.0.1:5173",
        "http://localhost:5174",
        "http://127.0.0.1:5174",
    )

    @classmethod
    def from_environment(cls) -> APISettings:
        default_origins = ",".join(cls.allowed_origins)
        origins = tuple(
            origin.strip()
            for origin in os.getenv("RETINAGRADE_ALLOWED_ORIGINS", default_origins).split(",")
            if origin.strip()
        )
        return cls(
            checkpoint=Path(
                os.getenv(
                    "RETINAGRADE_CHECKPOINT",
                    PROJECT_ROOT / "outputs/checkpoints/deployment/model_inference.pt",
                )
            ),
            device=os.getenv("RETINAGRADE_DEVICE", "auto"),
            max_upload_bytes=int(os.getenv("RETINAGRADE_MAX_UPLOAD_MB", "15")) * 1024 * 1024,
            allowed_origins=origins,
        )


class HealthResponse(BaseModel):
    status: str
    model_loaded: bool
    device: str | None = None


class PredictionResponse(BaseModel):
    grade: int
    label: str
    confidence: float
    probabilities: list[float]
    ordinal_probabilities: list[float]
    assessment_status: str
    confidence_margin: float
    inference_ms: float
    quality: ImageQualityReport


def create_app(
    *,
    settings: APISettings | None = None,
    predictor: RetinaGradePredictor | None = None,
    api_prefix: str = "",
    static_dir: Path | None = None,
) -> FastAPI:
    active_settings = settings or APISettings.from_environment()
    inference_lock = asyncio.Lock()
    normalized_prefix = f"/{api_prefix.strip('/')}" if api_prefix.strip("/") else ""

    @asynccontextmanager
    async def lifespan(app: FastAPI) -> AsyncIterator[None]:
        if predictor is None:
            app.state.predictor = await run_in_threadpool(
                RetinaGradePredictor,
                active_settings.checkpoint,
                data_config=active_settings.data_config,
                model_config=active_settings.model_config,
                normalization_stats=active_settings.normalization_stats,
                device=active_settings.device,
            )
        else:
            app.state.predictor = predictor
        warm_up = getattr(app.state.predictor, "warm_up", None)
        if warm_up is not None:
            await run_in_threadpool(warm_up)
        yield
        app.state.predictor = None

    app = FastAPI(
        title="RetinaGrade API",
        version="0.1.0",
        description="Research-only diabetic retinopathy grading API.",
        lifespan=lifespan,
    )
    app.add_middleware(
        CORSMiddleware,
        allow_origins=list(active_settings.allowed_origins),
        allow_credentials=False,
        allow_methods=["GET", "POST"],
        allow_headers=["Content-Type"],
    )

    @app.get(f"{normalized_prefix}/health", response_model=HealthResponse)
    async def health() -> HealthResponse:
        loaded_predictor = getattr(app.state, "predictor", None)
        device = str(loaded_predictor.device) if loaded_predictor is not None else None
        return HealthResponse(status="ready", model_loaded=True, device=device)

    @app.post(f"{normalized_prefix}/predict", response_model=PredictionResponse)
    async def predict(image: Annotated[UploadFile, File()]) -> PredictionResponse:
        suffix = ALLOWED_IMAGE_TYPES.get(image.content_type or "")
        if suffix is None:
            raise HTTPException(status_code=415, detail="Only PNG and JPEG images are supported")

        content = await image.read(active_settings.max_upload_bytes + 1)
        await image.close()
        if not content:
            raise HTTPException(status_code=400, detail="Uploaded image is empty")
        if len(content) > active_settings.max_upload_bytes:
            raise HTTPException(status_code=413, detail="Uploaded image exceeds the size limit")

        temporary_path: Path | None = None
        try:
            with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as temporary:
                temporary.write(content)
                temporary_path = Path(temporary.name)
            quality = await run_in_threadpool(assess_image_quality, temporary_path)
            if not quality.acceptable:
                raise HTTPException(status_code=422, detail=quality.warnings[0])
            started_at = time.perf_counter()
            async with inference_lock:
                result = await run_in_threadpool(app.state.predictor.predict, temporary_path)
            inference_ms = (time.perf_counter() - started_at) * 1000
        except HTTPException:
            raise
        except ValueError as exc:
            raise HTTPException(status_code=422, detail="Could not decode the uploaded image") from exc
        finally:
            if temporary_path is not None:
                temporary_path.unlink(missing_ok=True)

        ranked_probabilities = sorted(result.probabilities, reverse=True)
        confidence_margin = ranked_probabilities[0] - ranked_probabilities[1]
        assessment_status = (
            "conclusive"
            if result.confidence >= 0.5 and confidence_margin >= 0.15
            else "review_recommended"
        )
        return PredictionResponse(
            **asdict(result),
            assessment_status=assessment_status,
            confidence_margin=confidence_margin,
            inference_ms=inference_ms,
            quality=quality,
        )

    if static_dir is not None:
        app.mount("/", StaticFiles(directory=static_dir, html=True), name="frontend")

    return app


static_dir_value = os.getenv("RETINAGRADE_STATIC_DIR")
app = create_app(
    api_prefix=os.getenv("RETINAGRADE_API_PREFIX", ""),
    static_dir=Path(static_dir_value) if static_dir_value else None,
)
