"""Per-epoch CSV metric logging.

Reuses :func:`src.utils.helpers.write_csv` (atomic write) rather than
reimplementing CSV serialisation or append semantics; the full row history
is kept in memory and rewritten each epoch, which is negligible overhead at
the paper's own 50-epoch budget.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from src.utils.helpers import ensure_dir, write_csv
from src.utils.logger import get_logger

__all__ = ["CSVEpochLogger"]

logger = get_logger(__name__)


class CSVEpochLogger:
    """Accumulates one row of metrics per epoch and writes them to CSV.

    Args:
        log_dir: Directory the CSV file is written into.
        filename: CSV file name.
        project_root: Root used to resolve ``log_dir`` if it is relative.
    """

    def __init__(self, log_dir: str | Path, filename: str, project_root: Path) -> None:
        directory = Path(log_dir) if Path(log_dir).is_absolute() else project_root / log_dir
        self.path = ensure_dir(directory) / filename
        self._rows: list[dict[str, Any]] = []

    def log(self, epoch: int, metrics: dict[str, float]) -> None:
        """Append one epoch's metrics and persist the full history to disk.

        Args:
            epoch: 0-based epoch index.
            metrics: Flat mapping of metric name to value.
        """
        row = {"epoch": epoch, **metrics}
        self._rows.append(row)
        write_csv(self.path, self._rows)

    @property
    def rows(self) -> list[dict[str, Any]]:
        """The full in-memory row history, in log order."""
        return list(self._rows)
