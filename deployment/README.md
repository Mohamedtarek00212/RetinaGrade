# Deployment

This directory contains inference-only code. It deliberately does not contain
training data, optimizer setup, or medical decision logic.

Download `best.pt` from the GitHub `v0.1.0` release and place it at
`outputs/checkpoints/training/best.pt`. Export a deployment copy that excludes
the optimizer and scheduler states:

```bash
python scripts/export_inference_checkpoint.py
```

Then run single-image inference with the smaller artifact:

```bash
python scripts/predict.py path/to/fundus.png \
  --checkpoint outputs/checkpoints/deployment/model_inference.pt
```

The output is JSON so a future web UI or API can consume it without parsing
human-formatted text. Predictions are for research use only and are not a
medical diagnosis.

## FastAPI backend

Install the deployment dependencies and start the API from the project root:

```bash
python -m pip install -e ".[deployment]"
uvicorn deployment.api:app --host 0.0.0.0 --port 8000
```

The model is loaded once during server startup. Its pretrained backbone download
is disabled because the deployment checkpoint already contains every model
weight. A synthetic warm-up pass runs during startup so the first visitor does
not pay the model's one-time initialization cost.

Check readiness:

```bash
curl http://localhost:8000/health
```

Submit a PNG or JPEG fundus image:

```bash
curl -X POST http://localhost:8000/predict \
  -F "image=@Presentation/SampleImages/grade0_NoDR_e5197d77ec68.png"
```

Interactive API documentation is available at `http://localhost:8000/docs`.

Prediction responses include model confidence, the top-two confidence margin,
inference time, and a conservative image-quality report. Images below 224 px or
with unusable exposure/visible area are rejected. Blur and atypical retinal color
produce warnings rather than automatic rejection because the audited APTOS
samples vary substantially in capture quality.

### Environment variables

| Variable | Default | Purpose |
|---|---|---|
| `RETINAGRADE_CHECKPOINT` | `outputs/checkpoints/deployment/model_inference.pt` | Model checkpoint path |
| `RETINAGRADE_DEVICE` | `auto` | `auto`, `cpu`, or `cuda` |
| `RETINAGRADE_MAX_UPLOAD_MB` | `15` | Maximum uploaded image size |
| `RETINAGRADE_ALLOWED_ORIGINS` | Local Vite ports `5173` and `5174` | Comma-separated React origins |
