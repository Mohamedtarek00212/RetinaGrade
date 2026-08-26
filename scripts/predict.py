"""Predict a diabetic-retinopathy grade for one fundus image."""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict

from deployment.inference import RetinaGradePredictor


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("image", help="Path to a fundus image")
    parser.add_argument("--checkpoint", default="outputs/checkpoints/training/best.pt")
    parser.add_argument("--data-config", default="configs/data.yaml")
    parser.add_argument("--model-config", default="configs/model.yaml")
    parser.add_argument(
        "--normalization-stats",
        default="Presentation/Metrics/normalization_stats.json",
    )
    parser.add_argument("--device", default="auto", choices=["auto", "cpu", "cuda"])
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    predictor = RetinaGradePredictor(
        args.checkpoint,
        data_config=args.data_config,
        model_config=args.model_config,
        normalization_stats=args.normalization_stats,
        device=args.device,
    )
    print(json.dumps(asdict(predictor.predict(args.image)), indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
