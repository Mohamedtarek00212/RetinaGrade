"""Dual-SwinOrd model package: registries and the top-level model builder.

See ``docs/milestone_04_paper_gaps.md`` for the paper-fidelity policy this
package follows: only mechanisms explicitly stated in the paper are
hardcoded (the 4-stage backbone, PLKA's fixed dilation rates, SPM's
sigmoid gate, the classification head's ``K``-way output, and the ordinal
head's "> k" semantic); everything else is a required (no-default)
configuration field or an abstract interface with no concrete production
implementation.
"""

from __future__ import annotations

from src.models.attention.plka import PLKA, PLKAFusion, DefaultPLKAFusion
from src.models.backbones import BACKBONE_REGISTRY, build_backbone
from src.models.backbones.base import RetinalBackbone
from src.models.config import ModelConfig
from src.models.dual_head import DualHead
from src.models.dual_swinord import DualSwinOrd
from src.models.factories import activation_factory, normalization_factory
from src.models.heads.classification_head import ClassificationHead
from src.models.heads.ordinal_head import IndependentOrdinalHead, OrdinalHead
from src.models.neck.shared_feature_neck import GlobalAveragePooling, NeckPooling, SharedFeatureNeck
from src.models.registry import Registry
from src.models.semantic_prior.spm import DefaultSemanticPriorModulation, SemanticPriorModulation
from src.models.semantic_prior.text_adapter import HashingTextAdapter, TextAdapter
from src.utils.logger import get_logger

__all__ = [
    "BACKBONE_REGISTRY",
    "SPM_REGISTRY",
    "PLKA_FUSION_REGISTRY",
    "NECK_POOLING_REGISTRY",
    "ORDINAL_HEAD_REGISTRY",
    "MODEL_REGISTRY",
    "build_backbone",
    "build_model",
    "ModelConfig",
    "DualSwinOrd",
]

logger = get_logger(__name__)

#: Registered SPM strategies. Empty by default -- see PG-05/PG-06; no
#: concrete subclass ships in this milestone.
SPM_REGISTRY: Registry[SemanticPriorModulation] = Registry("spm")
SPM_REGISTRY.register("default")(DefaultSemanticPriorModulation)

#: Registered PLKA fusion strategies. Empty by default -- see PG-09.
PLKA_FUSION_REGISTRY: Registry[PLKAFusion] = Registry("plka_fusion")
PLKA_FUSION_REGISTRY.register("default")(DefaultPLKAFusion)

#: Registered neck pooling strategies. Empty by default -- see PG-11.
NECK_POOLING_REGISTRY: Registry[NeckPooling] = Registry("neck_pooling")
NECK_POOLING_REGISTRY.register("default")(GlobalAveragePooling)

#: Registered ordinal-head parameterizations. Empty by default -- see PG-13/PG-14.
ORDINAL_HEAD_REGISTRY: Registry[OrdinalHead] = Registry("ordinal_head")
ORDINAL_HEAD_REGISTRY.register("default")(IndependentOrdinalHead)

#: Registered top-level models. Mirrors ``DATASET_REGISTRY``'s pattern.
MODEL_REGISTRY: Registry[DualSwinOrd] = Registry("model")
MODEL_REGISTRY.register("dual_swinord")(DualSwinOrd)


def build_model(
    model_config: ModelConfig,
    num_classes: int,
    text_adapter: TextAdapter,
    text_prompts: list[str],
    *,
    backbone_registry: Registry[RetinalBackbone] | None = None,
    spm_registry: Registry[SemanticPriorModulation] | None = None,
    plka_fusion_registry: Registry[PLKAFusion] | None = None,
    neck_pooling_registry: Registry[NeckPooling] | None = None,
    ordinal_head_registry: Registry[OrdinalHead] | None = None,
) -> DualSwinOrd:
    """Assemble a :class:`DualSwinOrd` from a fully-specified :class:`ModelConfig`.

    ``text_adapter`` and ``text_prompts`` are supplied by the caller rather
    than built from configuration, because no concrete
    :class:`~src.models.semantic_prior.text_adapter.TextAdapter` ships in
    this milestone (PG-03, PG-04).

    The ``*_registry`` keyword arguments default to the module-level
    globals (which are empty for every gap-driven slot, by design) and
    exist so tests can inject local registries containing test-only
    implementations without mutating global state -- see
    ``tests/model_doubles.py``.

    Args:
        model_config: Fully-specified model configuration.
        num_classes: ``K``, sourced from ``DataConfig.classes.num_classes``.
        text_adapter: Frozen text encoder instance.
        text_prompts: Clinical text prompts to encode once.

    Returns:
        The assembled :class:`DualSwinOrd`.

    Raises:
        KeyError: If any registry-resolved component name in
            ``model_config`` has no registered implementation -- expected,
            for ``spm``, ``plka.fusion``, ``neck.pooling``, and
            ``heads.ordinal``, until the corresponding Paper Gap is
            resolved and a concrete implementation is registered.
    """
    backbone_registry = backbone_registry or BACKBONE_REGISTRY
    spm_registry = spm_registry or SPM_REGISTRY
    plka_fusion_registry = plka_fusion_registry or PLKA_FUSION_REGISTRY
    neck_pooling_registry = neck_pooling_registry or NECK_POOLING_REGISTRY
    ordinal_head_registry = ordinal_head_registry or ORDINAL_HEAD_REGISTRY

    backbone = build_backbone(model_config.backbone, registry=backbone_registry)

    spm = spm_registry.build(
        model_config.spm.name,
        visual_channels=backbone.out_channels[model_config.spm.inject_at_stage],
        text_embedding_dim=model_config.spm.text_embedding_dim,
    )

    plka_channels = backbone.out_channels[model_config.plka.input_stage]
    fusion = plka_fusion_registry.build(model_config.plka.fusion.name, channels=plka_channels)
    plka = PLKA(
        channels=plka_channels,
        kernel_size=model_config.plka.kernel_size,
        activation_factory=activation_factory(model_config.plka.activation),
        normalization_factory=normalization_factory(model_config.plka.normalization),
        fusion=fusion,
    )

    pooling = neck_pooling_registry.build(model_config.neck.pooling)
    neck = SharedFeatureNeck(
        pooling=pooling,
        in_channels=plka_channels,
        hidden_dim=model_config.neck.hidden_dim,
        dropout=model_config.neck.dropout,
        activation_factory=activation_factory(model_config.neck.activation),
    )

    classification_head = ClassificationHead(model_config.neck.hidden_dim, num_classes)
    ordinal_head = ordinal_head_registry.build(
        model_config.heads.ordinal.name,
        hidden_dim=model_config.neck.hidden_dim,
        num_classes=num_classes,
    )
    dual_head = DualHead(classification_head, ordinal_head)

    return MODEL_REGISTRY.get(model_config.model_name)(
        backbone=backbone,
        spm=spm,
        spm_inject_at_stage=model_config.spm.inject_at_stage,
        text_adapter=text_adapter,
        text_prompts=text_prompts,
        plka=plka,
        plka_input_stage=model_config.plka.input_stage,
        neck=neck,
        dual_head=dual_head,
    )
