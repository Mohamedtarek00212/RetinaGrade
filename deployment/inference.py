"""Single-image inference using the training-time model and preprocessing."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import torch

from src.data.preprocessing.pipeline import PreprocessingPipeline
from src.data.statistics import NormalizationStats
from src.models import build_model
from src.models.config import load_model_config
from src.models.semantic_prior.text_adapter import HashingTextAdapter
from src.utils.config import load_data_config
from src.utils.helpers import read_image_rgb

DEFAULT_TEXT_PROMPTS = [
    "No diabetic retinopathy: healthy retina with no visible lesions.",
    "Mild diabetic retinopathy: presence of microaneurysms only.",
    "Moderate diabetic retinopathy: microaneurysms, dot-blot hemorrhages, and hard exudates.",
    "Severe diabetic retinopathy: extensive hemorrhages, venous beading, and intraretinal microvascular abnormalities.",
    "Proliferative diabetic retinopathy: neovascularization, preretinal hemorrhage, or fibrovascular proliferation.",
]


@dataclass(frozen=True)
class Prediction:
    grade: int
    label: str
    confidence: float
    probabilities: list[float]
    ordinal_probabilities: list[float]


class RetinaGradePredictor:
    """Load the trained checkpoint once and predict individual fundus images."""

    def __init__(
        self,
        checkpoint: str | Path,
        *,
        data_config: str | Path = "configs/data.yaml",
        model_config: str | Path = "configs/model.yaml",
        normalization_stats: str | Path = "Presentation/Metrics/normalization_stats.json",
        device: str = "auto",
    ) -> None:
        self.device = torch.device(
            "cuda" if device == "auto" and torch.cuda.is_available() else
            "cpu" if device == "auto" else device
        )
        self.data_config = load_data_config(data_config)
        architecture = load_model_config(model_config)
        stats = NormalizationStats.load(normalization_stats)
        self.transform = PreprocessingPipeline(self.data_config).build("test", stats)

        adapter = HashingTextAdapter(embedding_dim=architecture.spm.text_embedding_dim)
        self.model = build_model(
            architecture,
            num_classes=self.data_config.classes.num_classes,
            text_adapter=adapter,
            text_prompts=DEFAULT_TEXT_PROMPTS,
        ).to(self.device)

        checkpoint_path = Path(checkpoint)
        if not checkpoint_path.is_file():
            raise FileNotFoundError(f"checkpoint not found: {checkpoint_path}")
        payload = torch.load(checkpoint_path, map_location=self.device, weights_only=False)
        state = payload.get("model_state_dict", payload)
        self.model.load_state_dict(state)
        self.model.eval()

    @torch.inference_mode()
    def predict(self, image_path: str | Path) -> Prediction:
        image = read_image_rgb(image_path)
        if image is None:
            raise ValueError(f"could not decode image: {image_path}")
        tensor = self.transform(image=image)["image"].unsqueeze(0).to(self.device)
        outputs = self.model(tensor)
        probabilities = torch.softmax(outputs["classification_logits"], dim=1)[0]
        ordinal = torch.sigmoid(outputs["ordinal_logits"])[0]
        grade = int(probabilities.argmax().item())
        names = self.data_config.classes.names
        label = names.get(grade, str(grade)) if isinstance(names, dict) else names[grade]
        return Prediction(
            grade=grade,
            label=str(label),
            confidence=float(probabilities[grade].item()),
            probabilities=[float(value) for value in probabilities.cpu().tolist()],
            ordinal_probabilities=[float(value) for value in ordinal.cpu().tolist()],
        )
