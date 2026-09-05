"""Export and quantize the browser inference model."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import onnx
import onnxruntime as ort
import torch
from onnx import helper, numpy_helper
from onnxruntime.quantization import QuantType, quantize_dynamic
from torch import Tensor, nn

from deployment.inference import DEFAULT_TEXT_PROMPTS, prepare_inference_config
from src.models import build_model
from src.models.config import load_model_config
from src.models.neck.shared_feature_neck import GlobalAveragePooling
from src.models.semantic_prior.text_adapter import HashingTextAdapter
from src.utils.config import load_data_config


class BrowserExportModel(nn.Module):
    """Expose predictions and a class-specific spatial contribution map."""

    def __init__(self, model: nn.Module) -> None:
        super().__init__()
        self.model = model
        if not isinstance(self.model.neck.pooling, GlobalAveragePooling):
            raise ValueError("class activation export requires global average pooling")
        if not isinstance(self.model.neck.activation, nn.Identity):
            raise ValueError("class activation export requires an identity neck activation")
        if self.model.neck.dropout.p != 0:
            raise ValueError("class activation export requires zero neck dropout")
        classifier_weight = self.model.dual_head.classification_head.linear.weight
        neck_weight = self.model.neck.fc.weight
        with torch.no_grad():
            spatial_class_weights = (classifier_weight @ neck_weight).detach()
        self.register_buffer(
            "_spatial_class_weights",
            spatial_class_weights,
            persistent=True,
        )

    def forward(self, image: Tensor) -> tuple[Tensor, Tensor, Tensor]:
        outputs = self.model(image)
        classification_logits = outputs["classification_logits"]
        spatial_features = outputs["spatial_features"]
        predicted_grades = classification_logits.argmax(dim=1)
        selected_weights = self._spatial_class_weights[predicted_grades, :, None, None]
        activation_map = torch.relu((spatial_features * selected_weights).sum(dim=1, keepdim=True))
        minimum = activation_map.amin(dim=(2, 3), keepdim=True)
        maximum = activation_map.amax(dim=(2, 3), keepdim=True)
        activation_map = (activation_map - minimum) / (maximum - minimum + 1e-6)
        return classification_logits, outputs["ordinal_logits"], activation_map


def quantize_conv_weights(source: Path, output: Path) -> None:
    """Store Conv weights as per-channel INT8 while keeping activations in FP32."""
    model = onnx.load(source)
    initializers = {value.name: value for value in model.graph.initializer}
    removed = []
    added = []
    dequantizers = []

    for node in model.graph.node:
        if node.op_type != "Conv" or node.input[1] not in initializers:
            continue

        weight_proto = initializers[node.input[1]]
        weight = numpy_helper.to_array(weight_proto)
        reduction_axes = tuple(range(1, weight.ndim))
        scale = np.max(np.abs(weight), axis=reduction_axes).astype(np.float32) / 127.0
        scale = np.maximum(scale, np.finfo(np.float32).eps)
        broadcast_shape = (weight.shape[0],) + (1,) * (weight.ndim - 1)
        quantized = np.clip(
            np.rint(weight / scale.reshape(broadcast_shape)),
            -127,
            127,
        ).astype(np.int8)

        quantized_name = f"{weight_proto.name}_quantized"
        scale_name = f"{weight_proto.name}_scale"
        zero_name = f"{weight_proto.name}_zero_point"
        dequantized_name = f"{weight_proto.name}_dequantized"
        added.extend(
            [
                numpy_helper.from_array(quantized, quantized_name),
                numpy_helper.from_array(scale, scale_name),
                numpy_helper.from_array(
                    np.zeros(weight.shape[0], dtype=np.int8),
                    zero_name,
                ),
            ]
        )
        dequantizers.append(
            helper.make_node(
                "DequantizeLinear",
                [quantized_name, scale_name, zero_name],
                [dequantized_name],
                axis=0,
                name=f"{node.name or weight_proto.name}_weight_dequantize",
            )
        )
        node.input[1] = dequantized_name
        removed.append(weight_proto)

    if not removed:
        raise ValueError("ONNX graph does not contain constant Conv weights")

    for value in removed:
        model.graph.initializer.remove(value)
    model.graph.initializer.extend(added)
    original_nodes = list(model.graph.node)
    del model.graph.node[:]
    model.graph.node.extend(dequantizers + original_nodes)
    onnx.checker.check_model(model)
    onnx.save(model, output)


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
    matmul_quantized = args.output.with_name("retinagrade.matmul-int8.onnx")

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
        output_names=["classification_logits", "ordinal_logits", "activation_map"],
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
        matmul_quantized,
        op_types_to_quantize=["MatMul", "Gemm"],
        weight_type=QuantType.QInt8,
        per_channel=True,
    )
    quantize_conv_weights(matmul_quantized, args.output)
    quantization_source.unlink(missing_ok=True)
    matmul_quantized.unlink(missing_ok=True)

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
