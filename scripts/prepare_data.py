"""Command-line entry point for the data-preparation pipeline.

Orchestration only. This script loads the configuration, seeds the RNGs, sets
up logging, and calls the stage classes in order. It contains no algorithmic
code, so the pipeline remains equally usable from Python, a notebook, or a
future Makefile/DVC target.

Pipeline stages::

    audit -> clean -> verify-splits -> stats -> (preview)

Usage::

    python scripts/prepare_data.py all
    python scripts/prepare_data.py audit --force
    python scripts/prepare_data.py stats --override statistics.normalization.mode=imagenet
    python scripts/prepare_data.py preview --limit 8
    python scripts/prepare_data.py regenerate-splits      # requires the config flag

Every subcommand accepts ``--config``, ``--override key=value`` (repeatable),
``--seed``, ``--log-level``, and ``--force``.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any

import pandas as pd

# Allow ``python scripts/prepare_data.py`` from a clean clone without installing.
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.data.audit import DatasetAuditor  # noqa: E402
from src.data.cleaning import DatasetCleaner  # noqa: E402
from src.data.splits import SplitRegenerator, SplitVerifier  # noqa: E402
from src.data.statistics import DatasetStatistics  # noqa: E402
from src.utils.config import DataConfig, load_data_config, parse_overrides  # noqa: E402
from src.utils.helpers import ensure_dir  # noqa: E402
from src.utils.logger import configure_logging, get_logger, log_section  # noqa: E402
from src.utils.seed import set_seed  # noqa: E402

logger = get_logger("scripts.prepare_data")


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _load_manifest(config: DataConfig, which: str) -> pd.DataFrame:
    """Load a previously written manifest.

    Args:
        config: Validated configuration.
        which: ``"audit"`` or ``"clean"``.

    Returns:
        The manifest.

    Raises:
        SystemExit: With an actionable message when the manifest is missing;
            a stack trace would add nothing for a user who simply has not run
            the previous stage yet.
    """
    path = config.resolve_path(
        config.outputs.audit_manifest if which == "audit" else config.outputs.clean_manifest
    )
    if not path.is_file():
        raise SystemExit(
            f"{which} manifest not found at {path}. Run "
            f"'python scripts/prepare_data.py {'audit' if which == 'audit' else 'clean'}' first."
        )
    return pd.read_csv(path)


def _build_config(args: argparse.Namespace) -> DataConfig:
    """Load the configuration with CLI overrides applied."""
    overrides: dict[str, Any] = parse_overrides(args.override)
    if args.seed is not None:
        overrides.setdefault("seed", args.seed)
    return load_data_config(args.config, overrides=overrides)


# ---------------------------------------------------------------------------
# Stage commands
# ---------------------------------------------------------------------------


def command_audit(config: DataConfig, args: argparse.Namespace) -> pd.DataFrame:
    """Run Stage 1 and return the audit manifest."""
    return DatasetAuditor(config).run(force=args.force).to_frame()


def command_clean(config: DataConfig, args: argparse.Namespace, audit: pd.DataFrame | None = None) -> pd.DataFrame:
    """Run Stage 2 and return the clean manifest."""
    frame = audit if audit is not None else _load_manifest(config, "audit")
    del args
    return DatasetCleaner(config).run(frame).frame


def command_verify_splits(
    config: DataConfig, args: argparse.Namespace, manifest: pd.DataFrame | None = None
) -> None:
    """Run Stage 3 verification."""
    frame = manifest if manifest is not None else _load_manifest(config, "clean")
    del args
    SplitVerifier(config).verify(frame)


def command_regenerate_splits(
    config: DataConfig, args: argparse.Namespace, manifest: pd.DataFrame | None = None
) -> None:
    """Run the opt-in Stage 3b regeneration.

    Writes only when ``splits_policy.regenerate.enabled`` is true, or when
    ``--write`` is passed explicitly. Otherwise a dry-run plan is reported.
    """
    frame = manifest if manifest is not None else _load_manifest(config, "clean")
    plan = SplitRegenerator(config).regenerate(frame, write=True if args.write else None)
    if not plan.written_files:
        logger.info(
            "dry run only: no split files were written. Pass --write, or set "
            "splits_policy.regenerate.enabled: true, to persist this plan."
        )


def command_stats(
    config: DataConfig, args: argparse.Namespace, manifest: pd.DataFrame | None = None
) -> None:
    """Run Stage 4 statistics."""
    from src.data.preprocessing import PreprocessingPipeline

    frame = manifest if manifest is not None else _load_manifest(config, "clean")
    DatasetStatistics(config).run(
        frame,
        transform=PreprocessingPipeline(config),
        compute_normalization=not args.no_normalization,
        force=args.force,
    )


def command_preview(config: DataConfig, args: argparse.Namespace) -> None:
    """Render before/after previews of preprocessing and augmentation.

    Cheap visual QA: a broken crop, an inverted channel order, or an overly
    aggressive augmentation is obvious in one glance and invisible in a metric.
    """
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np

    from src.data.augmentation import build_train_transforms
    from src.data.preprocessing import PreprocessingPipeline
    from src.utils.helpers import read_image_rgb

    log_section(logger, "Preview / preprocessing and augmentation samples")
    manifest = _load_manifest(config, "clean")
    usable = manifest[manifest["included"] & manifest["readable"]]
    if usable.empty:
        raise SystemExit("no included, readable images available to preview")

    rows = usable.sample(n=min(args.limit, len(usable)), random_state=config.seed)
    pipeline = PreprocessingPipeline(config)
    augmentation = build_train_transforms(config)
    import albumentations as A

    augment = A.Compose(augmentation) if augmentation else None

    destination = ensure_dir(config.resolve_path(config.outputs.preview_dir))
    figure, axes = plt.subplots(len(rows), 3, figsize=(9, 3 * len(rows)))
    axes = np.atleast_2d(axes)

    for position, (_, row) in enumerate(rows.iterrows()):
        raw = read_image_rgb(str(row["path"]))
        if raw is None:
            continue
        processed = pipeline(raw)
        augmented = augment(image=processed)["image"] if augment is not None else processed
        for column, (image, title) in enumerate(
            ((raw, "raw"), (processed, "preprocessed"), (augmented, "augmented"))
        ):
            axis = axes[position, column]
            axis.imshow(image)
            axis.set_title(f"{row['id_code']} | {title}", fontsize=8)
            axis.axis("off")

    figure.tight_layout()
    output = destination / f"preview_{config.preprocessing_hash}.png"
    figure.savefig(output, dpi=130)
    plt.close(figure)
    logger.info("preview written to %s", output)


def command_all(config: DataConfig, args: argparse.Namespace) -> None:
    """Run stages 1-4 in order, passing manifests in memory.

    Split regeneration is never part of ``all``: it changes every downstream
    number and must be an explicit, separate act.
    """
    audit = command_audit(config, args)
    clean = command_clean(config, args, audit=audit)
    command_verify_splits(config, args, manifest=clean)
    command_stats(config, args, manifest=clean)
    logger.info("data preparation complete; artefacts are in %s", config.outputs.root)


# ---------------------------------------------------------------------------
# CLI plumbing
# ---------------------------------------------------------------------------


def _common_options() -> argparse.ArgumentParser:
    """Build the shared option parser.

    Attached both to the root parser and to every subparser, so ``--force``
    works whether it precedes or follows the subcommand. Argparse otherwise
    rejects the (more natural) trailing form.
    """
    common = argparse.ArgumentParser(add_help=False)
    common.add_argument("--config", default="configs/data.yaml", help="path to the data configuration")
    common.add_argument(
        "--override",
        action="append",
        default=[],
        metavar="KEY=VALUE",
        help="dotted configuration override; repeatable",
    )
    common.add_argument("--seed", type=int, default=None, help="override the configured seed")
    common.add_argument("--force", action="store_true", help="ignore caches and recompute")
    common.add_argument(
        "--log-level",
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        help="console log level",
    )
    return common


def build_parser() -> argparse.ArgumentParser:
    """Build the argument parser."""
    common = _common_options()
    parser = argparse.ArgumentParser(
        prog="prepare_data",
        parents=[common],
        description="RetinaGrade data-preparation pipeline (audit, clean, verify, statistics).",
    )

    subparsers = parser.add_subparsers(dest="command", required=True)
    subparsers.add_parser(
        "audit", parents=[common], help="Stage 1: full-corpus, read-only measurement"
    )
    subparsers.add_parser(
        "clean", parents=[common], help="Stage 2: decision-only cleaning (never deletes files)"
    )
    subparsers.add_parser(
        "verify-splits", parents=[common], help="Stage 3: leakage and stratification checks"
    )

    regenerate = subparsers.add_parser(
        "regenerate-splits", parents=[common], help="Stage 3b: opt-in, group-aware split regeneration"
    )
    regenerate.add_argument(
        "--write", action="store_true", help="persist the plan even if the config flag is off"
    )

    stats = subparsers.add_parser(
        "stats", parents=[common], help="Stage 4: imbalance analysis and normalization statistics"
    )
    stats.add_argument(
        "--no-normalization", action="store_true", help="skip the normalization pixel pass"
    )

    preview = subparsers.add_parser(
        "preview", parents=[common], help="render preprocessing/augmentation samples"
    )
    preview.add_argument("--limit", type=int, default=6, help="number of images to render")

    subparsers.add_parser("all", parents=[common], help="run stages 1-4 (never regenerates splits)")
    return parser


def main(argv: list[str] | None = None) -> int:
    """Entry point.

    Args:
        argv: Command-line arguments; defaults to ``sys.argv[1:]``.

    Returns:
        Process exit code.
    """
    parser = build_parser()
    args = parser.parse_args(argv)

    config = _build_config(args)
    configure_logging(level=args.log_level, log_dir=config.resolve_path(config.outputs.log_dir))
    set_seed(
        config.seed,
        deterministic=config.reproducibility.deterministic,
        cudnn_benchmark=config.reproducibility.cudnn_benchmark,
    )
    logger.info(
        "config %s | profile %s | dataset %s | seed %d",
        config.config_hash,
        config.profile,
        config.dataset_name,
        config.seed,
    )

    handlers = {
        "audit": lambda: command_audit(config, args),
        "clean": lambda: command_clean(config, args),
        "verify-splits": lambda: command_verify_splits(config, args),
        "regenerate-splits": lambda: command_regenerate_splits(config, args),
        "stats": lambda: command_stats(config, args),
        "preview": lambda: command_preview(config, args),
        "all": lambda: command_all(config, args),
    }
    handlers[args.command]()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
