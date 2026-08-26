# Deployment

This directory contains inference-only code. It deliberately does not contain
training data, optimizer setup, or medical decision logic.

Download `best.pt` from the GitHub `v0.1.0` release and place it at
`outputs/checkpoints/training/best.pt`, then run:

```bash
python scripts/predict.py path/to/fundus.png
```

The output is JSON so a future web UI or API can consume it without parsing
human-formatted text. Predictions are for research use only and are not a
medical diagnosis.
