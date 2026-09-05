"""Export and quantize the browser inference model."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import onnx
import onnxruntime as ort
import torch
from onnxruntime.quantization import QuantType, quantize_dynamic
from torch import Tensor, nn

from deployment.inference import DEFAULT_TEXT_PROMPTS, prepare_inference_config
from src.models import build_model
from src.models.config import load_model_config
from src.models.semantic_prior.text_adapter import HashingTextAdapter
from src.utils.config import load_data_config


class BrowserExportModel(nn.Module):
    """Expose only the two tensors consumed by the browser interface."""

    def __init__(self, model: nn.Module) -> None:
        super().__init__()
        self.model = model

    def forward(self, image: Tensor) -> tuple[Tensor, Tensor]:
        outputs = self.model(image)
        return outputs["classification_logits"], outputs["ordinal_logits"]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=Path("outputs/checkpoints/deployment/model_inference.pt"),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("frontend/public/models/retinagrade.int8.onnx"),
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    full_precision = args.output.with_name("retinagrade.fp32.onnx")
    quantization_source = args.output.with_name("retinagrade.quant-source.onnx")

    data_config = load_data_config("configs/data.yaml")
    architecture = prepare_inference_config(load_model_config("configs/model.yaml"))
    adapter = HashingTextAdapter(embedding_dim=architecture.spm.text_embedding_dim)
    model = build_model(
        architecture,
        num_classes=data_config.classes.num_classes,
        text_adapter=adapter,
        text_prompts=DEFAULT_TEXT_PROMPTS,
    )
    payload = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    model.load_state_dict(payload.get("model_state_dict", payload))
    wrapper = BrowserExportModel(model.eval()).eval()
    sample = torch.zeros(
        1,
        3,
        data_config.preprocessing.image_size,
        data_config.preprocessing.image_size,
    )

    torch.onnx.export(
        wrapper,
        (sample,),
        full_precision,
        input_names=["image"],
        output_names=["classification_logits", "ordinal_logits"],
        opset_version=18,
        dynamo=True,
        external_data=False,
    )
    onnx.checker.check_model(onnx.load(full_precision))

    quantization_model = onnx.load(full_precision)
    del quantization_model.graph.value_info[:]
    onnx.save(quantization_model, quantization_source)
    quantize_dynamic(
        quantization_source,
        args.output,
        op_types_to_quantize=["MatMul", "Gemm"],
        weight_type=QuantType.QInt8,
        per_channel=True,
    )
    quantization_source.unlink(missing_ok=True)

    with torch.inference_mode():
        expected = wrapper(sample)
    session = ort.InferenceSession(args.output, providers=["CPUExecutionProvider"])
    actual = session.run(None, {"image": sample.numpy()})
    differences = [
        float(np.max(np.abs(reference.numpy() - candidate)))
        for reference, candidate in zip(expected, actual, strict=True)
    ]

    print(f"FP32 model: {full_precision.stat().st_size / 1024 / 1024:.1f} MB")
    print(f"INT8 browser model: {args.output.stat().st_size / 1024 / 1024:.1f} MB")
    print(f"Maximum zero-input tensor differences: {differences}")
    full_precision.unlink(missing_ok=True)


if __name__ == "__main__":
    main()
