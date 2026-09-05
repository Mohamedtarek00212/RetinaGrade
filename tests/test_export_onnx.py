"""Tests for browser-model ONNX weight quantization."""

from pathlib import Path

import numpy as np
import pytest
import torch
from torch import Tensor, nn

onnx = pytest.importorskip("onnx")
from onnx import TensorProto, helper, numpy_helper  # noqa: E402

from scripts.export_onnx import BrowserExportModel, quantize_conv_weights  # noqa: E402
from src.models import ClassificationHead, DualHead, IndependentOrdinalHead  # noqa: E402
from src.models.neck.shared_feature_neck import (  # noqa: E402
    GlobalAveragePooling,
    SharedFeatureNeck,
)


class _SpatialFixtureModel(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.neck = SharedFeatureNeck(
            pooling=GlobalAveragePooling(),
            in_channels=2,
            hidden_dim=2,
            dropout=0.0,
            activation_factory=nn.Identity,
        )
        self.dual_head = DualHead(
            ClassificationHead(hidden_dim=2, num_classes=2),
            IndependentOrdinalHead(hidden_dim=2, num_classes=2),
        )

    def forward(self, image: Tensor) -> dict[str, Tensor]:
        embedding = self.neck(image)
        return {
            **self.dual_head(embedding),
            "shared_embedding": embedding,
            "spatial_features": image,
        }


def test_browser_export_returns_selected_class_activation_map() -> None:
    model = _SpatialFixtureModel()
    with torch.no_grad():
        model.neck.fc.weight.copy_(torch.eye(2))
        model.neck.fc.bias.zero_()
        model.dual_head.classification_head.linear.weight.copy_(torch.eye(2))
        model.dual_head.classification_head.linear.bias.zero_()
        model.dual_head.ordinal_head.linear.weight.zero_()
        model.dual_head.ordinal_head.linear.bias.zero_()

    image = torch.tensor([[[[1.0, 2.0], [3.0, 4.0]], [[0.0, 0.0], [0.0, 0.0]]]])
    logits, _, activation = BrowserExportModel(model)(image)

    assert logits.argmax(dim=1).item() == 0
    torch.testing.assert_close(
        activation,
        torch.tensor([[[[0.0, 1 / 3], [2 / 3, 1.0]]]]),
        atol=1e-5,
        rtol=0,
    )


def test_quantize_conv_weights_uses_per_channel_int8(tmp_path: Path) -> None:
    weight = np.array(
        [
            [[[0.25, -0.5], [0.75, 1.0]]],
            [[[-2.0, 1.5], [0.5, -1.0]]],
        ],
        dtype=np.float32,
    )
    graph = helper.make_graph(
        [helper.make_node("Conv", ["image", "weight"], ["output"], name="conv")],
        "conv_model",
        [helper.make_tensor_value_info("image", TensorProto.FLOAT, [1, 1, 3, 3])],
        [helper.make_tensor_value_info("output", TensorProto.FLOAT, [1, 2, 2, 2])],
        [numpy_helper.from_array(weight, "weight")],
    )
    source = tmp_path / "source.onnx"
    output = tmp_path / "output.onnx"
    onnx.save(helper.make_model(graph, opset_imports=[helper.make_opsetid("", 18)]), source)

    quantize_conv_weights(source, output)

    model = onnx.load(output)
    initializers = {value.name: value for value in model.graph.initializer}
    assert "weight" not in initializers
    assert initializers["weight_quantized"].data_type == TensorProto.INT8
    assert [node.op_type for node in model.graph.node].count("DequantizeLinear") == 1
    assert model.graph.node[-1].input[1] == "weight_dequantized"

    quantized = numpy_helper.to_array(initializers["weight_quantized"])
    scale = numpy_helper.to_array(initializers["weight_scale"])
    restored = quantized.astype(np.float32) * scale.reshape(2, 1, 1, 1)
    np.testing.assert_allclose(restored, weight, atol=float(scale.max() / 2))
