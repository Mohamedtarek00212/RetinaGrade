"""Structured logging for the data-preparation pipeline.

Every stage logs through a named child of the ``retinagrade`` root logger, so a
single call to :func:`configure_logging` controls console verbosity and file
output for the whole pipeline. Handlers are attached exactly once, which keeps
repeated calls (notebooks, pytest, CLI re-entry) from producing duplicated
lines.

Two design choices matter for research reproducibility:

* **Rotating file handler.** Every run appends to ``logs/data_prep.log`` with
  size-based rotation, so a failed audit three days ago is still diagnosable.
* **UTC timestamps and stage names in the format.** Logs from parallel stages
  remain attributable and comparable across machines and time zones.

Example
-------
>>> from src.utils.logger import configure_logging, get_logger
>>> configure_logging(level="INFO", log_dir="logs")          # doctest: +SKIP
>>> logger = get_logger(__name__)
>>> logger.info("audit finished: %d images", 3662)           # doctest: +SKIP
"""

from __future__ import annotations

import logging
import logging.handlers
import sys
import time
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path

from tqdm import tqdm

__all__ = [
    "ROOT_LOGGER_NAME",
    "configure_logging",
    "get_logger",
    "log_duration",
    "log_section",
]

#: Name of the package-level logger every module hangs off.
ROOT_LOGGER_NAME = "retinagrade"

#: Console format: compact enough to read, detailed enough to attribute.
CONSOLE_FORMAT = "%(asctime)s | %(levelname)-8s | %(name)s | %(message)s"

#: File format: adds module/line so post-hoc debugging does not need a rerun.
FILE_FORMAT = "%(asctime)s | %(levelname)-8s | %(name)s | %(module)s:%(lineno)d | %(message)s"

DATE_FORMAT = "%Y-%m-%d %H:%M:%S"

_CONSOLE_HANDLER_NAME = "retinagrade_console"
_FILE_HANDLER_NAME = "retinagrade_file"


class _UtcFormatter(logging.Formatter):
    """Formatter that emits UTC timestamps regardless of machine locale."""

    converter = time.gmtime


def _find_handler(logger: logging.Logger, name: str) -> logging.Handler | None:
    """Return the handler registered under ``name``, if any."""
    for handler in logger.handlers:
        if handler.get_name() == name:
            return handler
    return None


class _TqdmStreamHandler(logging.StreamHandler):
    """Stream handler that writes through ``tqdm.write`` to avoid clobbering bars."""

    def emit(self, record: logging.LogRecord) -> None:
        try:
            msg = self.format(record)
            tqdm.write(msg, file=self.stream)
            self.flush()
        except Exception:  # pragma: no cover - defensive
            self.handleError(record)


def configure_logging(
    level: int | str = logging.INFO,
    log_dir: str | Path | None = None,
    filename: str = "data_prep.log",
    file_level: int | str = logging.DEBUG,
    max_bytes: int = 5 * 1024 * 1024,
    backup_count: int = 5,
    force: bool = False,
    tqdm_compatible: bool = False,
) -> logging.Logger:
    """Configure the package logger with console and optional file handlers.

    Idempotent: calling this repeatedly updates levels but never duplicates
    handlers, unless ``force`` is set.

    Args:
        level: Console log level (name or numeric value).
        log_dir: Directory for the rotating log file. ``None`` disables file
            logging (useful in unit tests, which must not touch the filesystem).
        filename: Log file name inside ``log_dir``.
        file_level: Log level for the file handler; kept at ``DEBUG`` so the
            file remains a complete forensic record even when the console is
            quiet.
        max_bytes: Rotation threshold in bytes.
        backup_count: Number of rotated files to retain.
        force: Remove existing handlers before attaching new ones.
        tqdm_compatible: If ``True``, the console handler writes through
            ``tqdm.write`` so log lines do not overwrite active progress bars.

    Returns:
        The configured ``retinagrade`` logger.
    """
    logger = logging.getLogger(ROOT_LOGGER_NAME)
    # The logger itself must pass everything through; the handlers filter.
    logger.setLevel(logging.DEBUG)
    # Do not leak records into the root logger (avoids duplicate console output
    # when a host application, notebook, or pytest configures its own root).
    logger.propagate = False

    if force:
        for handler in list(logger.handlers):
            logger.removeHandler(handler)
            handler.close()

    console = _find_handler(logger, _CONSOLE_HANDLER_NAME)
    if console is None:
        console = (
            _TqdmStreamHandler(stream=sys.stdout)
            if tqdm_compatible
            else logging.StreamHandler(stream=sys.stdout)
        )
        console.set_name(_CONSOLE_HANDLER_NAME)
        console.setFormatter(_UtcFormatter(CONSOLE_FORMAT, datefmt=DATE_FORMAT))
        logger.addHandler(console)
    elif isinstance(console, logging.StreamHandler) and tqdm_compatible and not isinstance(console, _TqdmStreamHandler):
        # Replace a non-tqdm console handler with the tqdm-safe variant.
        logger.removeHandler(console)
        console.close()
        console = _TqdmStreamHandler(stream=sys.stdout)
        console.set_name(_CONSOLE_HANDLER_NAME)
        console.setFormatter(_UtcFormatter(CONSOLE_FORMAT, datefmt=DATE_FORMAT))
        logger.addHandler(console)
    console.setLevel(level)

    if log_dir is not None:
        directory = Path(log_dir)
        directory.mkdir(parents=True, exist_ok=True)
        file_handler = _find_handler(logger, _FILE_HANDLER_NAME)
        if file_handler is None:
            file_handler = logging.handlers.RotatingFileHandler(
                directory / filename,
                maxBytes=max_bytes,
                backupCount=backup_count,
                encoding="utf-8",
            )
            file_handler.set_name(_FILE_HANDLER_NAME)
            file_handler.setFormatter(_UtcFormatter(FILE_FORMAT, datefmt=DATE_FORMAT))
            logger.addHandler(file_handler)
        file_handler.setLevel(file_level)

    return logger


def get_logger(name: str | None = None) -> logging.Logger:
    """Return a child of the package logger.

    Args:
        name: Usually ``__name__``. A leading ``src.`` prefix is stripped and
            the remainder is nested under ``retinagrade`` so log lines read
            ``retinagrade.data.audit`` rather than ``src.data.audit``.

    Returns:
        The requested logger.

    Example:
        >>> get_logger("src.data.audit").name
        'retinagrade.data.audit'
    """
    if not name:
        return logging.getLogger(ROOT_LOGGER_NAME)
    suffix = name[4:] if name.startswith("src.") else name
    if suffix in ("", ROOT_LOGGER_NAME):
        return logging.getLogger(ROOT_LOGGER_NAME)
    return logging.getLogger(f"{ROOT_LOGGER_NAME}.{suffix}")


@contextmanager
def log_duration(logger: logging.Logger, message: str, level: int = logging.INFO) -> Iterator[None]:
    """Log the wall-clock duration of a block of work.

    Stage timings are recorded for every run, which turns "the audit felt slow"
    into a number that can be compared across configurations.

    Args:
        logger: Logger to write to.
        message: Description of the work being timed.
        level: Level used for the completion message.

    Yields:
        ``None``.

    Example:
        >>> import logging
        >>> with log_duration(get_logger("demo"), "hashing"):  # doctest: +SKIP
        ...     hash_files(paths)
    """
    logger.log(level, "%s ...", message)
    start = time.perf_counter()
    try:
        yield
    except Exception:
        elapsed = time.perf_counter() - start
        logger.exception("%s failed after %.2fs", message, elapsed)
        raise
    else:
        elapsed = time.perf_counter() - start
        logger.log(level, "%s finished in %.2fs", message, elapsed)


def log_section(logger: logging.Logger, title: str, level: int = logging.INFO) -> None:
    """Emit a visually distinct section banner.

    Args:
        logger: Logger to write to.
        title: Section title.
        level: Log level for the banner.
    """
    logger.log(level, "=" * 78)
    logger.log(level, title)
    logger.log(level, "=" * 78)
