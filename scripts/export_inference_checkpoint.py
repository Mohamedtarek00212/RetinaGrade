"""Export a training checkpoint as a smaller inference-only artifact."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

import torch


def build_inference_payload(checkpoint: Any) -> dict[str, Any]:
    """Keep model weights and lightweight provenance needed for deployment."""
    if not isinstance(checkpoint, dict):
        raise TypeError("checkpoint must be a dictionary")

    state_dict = checkpoint.get("model_state_dict")
    if state_dict is None:
        # A raw state dict maps parameter names directly to tensors.
        if checkpoint and all(isinstance(key, str) for key in checkpoint):
            state_dict = checkpoint
        else:
            raise ValueError("checkpoint does not contain model_state_dict")

    metadata = {
        key: checkpoint[key]
        for key in ("epoch", "metrics", "extra")
        if key in checkpoint
    }
    return {
        "format": "retinagrade-inference-v1",
        "model_state_dict": state_dict,
        "metadata": metadata,
    }


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--checkpoint",
        default="outputs/checkpoints/training/best.pt",
        help="Training checkpoint to convert",
    )
    parser.add_argument(
        "--output",
        default="outputs/checkpoints/deployment/model_inference.pt",
        help="Destination for the inference-only checkpoint",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    source = Path(args.checkpoint)
    destination = Path(args.output)

    if not source.is_file():
        raise FileNotFoundError(f"checkpoint not found: {source}")
    if source.resolve() == destination.resolve():
        raise ValueError("output must not overwrite the training checkpoint")

    checkpoint = torch.load(source, map_location="cpu", weights_only=False)
    payload = build_inference_payload(checkpoint)
    destination.parent.mkdir(parents=True, exist_ok=True)
    torch.save(payload, destination)

    source_mb = source.stat().st_size / (1024**2)
    destination_mb = destination.stat().st_size / (1024**2)
    reduction = (1 - destination_mb / source_mb) * 100
    print(f"Exported {destination}")
    print(f"Size: {source_mb:.1f} MB -> {destination_mb:.1f} MB ({reduction:.1f}% smaller)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
