"""Shared, dependency-light primitives used across the data pipeline.

This module deliberately holds only *primitives*: file IO, hashing, and
single-image measurements. Every stage-level decision (what counts as a
duplicate, what counts as too dark) lives in the stage modules, so the same
primitive can be reused without importing a policy.

Contents
--------
* Atomic JSON/CSV writes and safe reads.
* ``md5_file`` - streamed MD5, the **authoritative** exact-duplicate detector.
* ``dhash`` / ``phash`` - perceptual hashes implemented with NumPy + OpenCV
  only, used for **investigation, clustering, and reporting only**. They must
  never justify an automatic deletion or exclusion.
* Image quality measurements (brightness, contrast, resolution-normalized
  sharpness, Immerkaer noise sigma, tissue bounding box).

No third-party hashing or validation package is used, keeping the
reproducibility surface of the project small and the hash semantics fully
specified by this file.
"""

from __future__ import annotations

import hashlib
import json
import os
import tempfile
from collections.abc import Iterable, Mapping, Sequence
from pathlib import Path
from typing import Any

import cv2
import numpy as np

from src.utils.logger import get_logger

__all__ = [
    "ensure_dir",
    "atomic_write_text",
    "write_json",
    "read_json",
    "write_csv",
    "md5_file",
    "md5_bytes",
    "dhash",
    "phash",
    "hamming_distance",
    "hex_to_bits",
    "read_image_rgb",
    "downscale_long_side",
    "image_brightness",
    "image_contrast",
    "laplacian_variance",
    "normalized_sharpness",
    "estimate_noise_sigma",
    "tissue_bounding_box",
    "black_padding_ratio",
    "file_signature",
    "human_bytes",
    "resolve_column",
]

logger = get_logger(__name__)

#: Chunk size for streamed hashing (1 MiB). Large enough to amortise syscalls,
#: small enough to keep memory flat when hashing 4288x2848 PNGs.
_HASH_CHUNK_BYTES = 1024 * 1024


# ---------------------------------------------------------------------------
# Filesystem and serialisation
# ---------------------------------------------------------------------------


def ensure_dir(path: str | Path) -> Path:
    """Create a directory (including parents) if it does not exist.

    Args:
        path: Directory path.

    Returns:
        The directory as a :class:`~pathlib.Path`.
    """
    directory = Path(path)
    directory.mkdir(parents=True, exist_ok=True)
    return directory


def atomic_write_text(path: str | Path, content: str, encoding: str = "utf-8") -> Path:
    """Write text atomically via a temporary file and ``os.replace``.

    A half-written report is worse than no report: it silently corrupts the
    provenance trail of an experiment. Writing to a sibling temporary file and
    renaming makes the update atomic on all supported platforms.

    Args:
        path: Destination file path.
        content: Text to write.
        encoding: Text encoding.

    Returns:
        The destination path.
    """
    destination = Path(path)
    ensure_dir(destination.parent)
    handle_fd, temp_name = tempfile.mkstemp(dir=str(destination.parent), suffix=".tmp")
    try:
        with os.fdopen(handle_fd, "w", encoding=encoding, newline="") as handle:
            handle.write(content)
        os.replace(temp_name, destination)
    except BaseException:
        Path(temp_name).unlink(missing_ok=True)
        raise
    return destination


def write_json(path: str | Path, payload: Any, indent: int = 2) -> Path:
    """Serialise ``payload`` to JSON atomically.

    NumPy scalars and arrays are converted so that reports can be assembled
    from measurement code without manual casting at every call site.

    Args:
        path: Destination file path.
        payload: JSON-serialisable object (NumPy types allowed).
        indent: Indentation level.

    Returns:
        The destination path.
    """

    def _default(value: Any) -> Any:
        if isinstance(value, (np.integer,)):
            return int(value)
        if isinstance(value, (np.floating,)):
            return float(value)
        if isinstance(value, np.ndarray):
            return value.tolist()
        if isinstance(value, Path):
            return str(value)
        if isinstance(value, set):
            return sorted(value)
        return str(value)

    text = json.dumps(payload, indent=indent, sort_keys=False, default=_default)
    return atomic_write_text(path, text + "\n")


def read_json(path: str | Path) -> Any:
    """Read a JSON file.

    Args:
        path: Source file path.

    Returns:
        The decoded object.

    Raises:
        FileNotFoundError: If the file does not exist.
    """
    with Path(path).open("r", encoding="utf-8") as handle:
        return json.load(handle)


def write_csv(path: str | Path, rows: Sequence[Mapping[str, Any]], columns: Sequence[str] | None = None) -> Path:
    """Write a list of mappings to CSV atomically.

    Uses :mod:`csv` rather than pandas so that the helper stays usable in
    contexts (workers, tests) where importing pandas is unnecessary overhead.

    Args:
        path: Destination file path.
        rows: Rows to write.
        columns: Explicit column order; inferred from the union of row keys
            when omitted.

    Returns:
        The destination path.
    """
    import csv
    import io

    if columns is None:
        seen: dict[str, None] = {}
        for row in rows:
            for key in row:
                seen.setdefault(key, None)
        columns = list(seen)

    buffer = io.StringIO(newline="")
    writer = csv.DictWriter(buffer, fieldnames=list(columns), extrasaction="ignore")
    writer.writeheader()
    for row in rows:
        writer.writerow(row)
    return atomic_write_text(path, buffer.getvalue())


def resolve_column(available: Sequence[str], candidates: Sequence[str], kind: str) -> str:
    """Pick the first candidate column name that exists in a table.

    Candidate lists (rather than hard-coded names) let the same pipeline read
    APTOS, EyePACS, DDR, and Messidor manifests, which spell the id and label
    columns differently. Matching is case-insensitive and ignores surrounding
    whitespace, because exported CSVs frequently carry both.

    Args:
        available: Column names present in the table.
        candidates: Accepted names, in priority order.
        kind: Human-readable description used in the error message
            (for example ``"id"`` or ``"label"``).

    Returns:
        The matching column name exactly as it appears in ``available``.

    Raises:
        KeyError: If no candidate matches.

    Example:
        >>> resolve_column(["id_code", "diagnosis"], ["image_id", "id_code"], "id")
        'id_code'
    """
    normalised = {str(name).strip().lower(): name for name in available}
    for candidate in candidates:
        match = normalised.get(candidate.strip().lower())
        if match is not None:
            return match
    raise KeyError(
        f"could not resolve the {kind} column: tried {list(candidates)}, "
        f"available columns are {list(available)}"
    )


def file_signature(path: str | Path) -> tuple[int, int]:
    """Return ``(size_bytes, mtime_ns)`` for cache invalidation.

    Cheap enough to call for every file on every run, which is what makes the
    audit resumable without re-decoding images.

    Args:
        path: File path.

    Returns:
        Size in bytes and modification time in nanoseconds.
    """
    stat = Path(path).stat()
    return stat.st_size, stat.st_mtime_ns


def human_bytes(num_bytes: float) -> str:
    """Format a byte count for logs.

    Args:
        num_bytes: Number of bytes.

    Returns:
        A human-readable string such as ``"1.4 GB"``.
    """
    for unit in ("B", "KB", "MB", "GB", "TB"):
        if abs(num_bytes) < 1024.0 or unit == "TB":
            return f"{num_bytes:.1f} {unit}"
        num_bytes /= 1024.0
    return f"{num_bytes:.1f} TB"  # pragma: no cover - unreachable


# ---------------------------------------------------------------------------
# Hashing
# ---------------------------------------------------------------------------


def md5_file(path: str | Path, chunk_size: int = _HASH_CHUNK_BYTES) -> str:
    """Compute the MD5 digest of a file's raw bytes.

    MD5 over file bytes is the **authoritative** exact-duplicate criterion for
    this project: it is exact (no threshold, no tuning), reproducible across
    machines, and independent of any decoding library's version. Collisions are
    irrelevant here because the adversary is a duplicated export, not an
    attacker.

    Args:
        path: File path.
        chunk_size: Streaming chunk size in bytes.

    Returns:
        The lowercase hexadecimal digest.
    """
    digest = hashlib.md5()  # noqa: S324 - integrity/dedup only, not security
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(chunk_size), b""):
            digest.update(chunk)
    return digest.hexdigest()


def md5_bytes(payload: bytes) -> str:
    """Compute the MD5 digest of an in-memory buffer.

    Args:
        payload: Bytes to hash.

    Returns:
        The lowercase hexadecimal digest.
    """
    return hashlib.md5(payload).hexdigest()  # noqa: S324 - integrity/dedup only


def _to_gray(image: np.ndarray) -> np.ndarray:
    """Convert an RGB or grayscale image to single-channel ``uint8``."""
    if image.ndim == 2:
        return image
    if image.ndim == 3 and image.shape[2] == 3:
        return cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    raise ValueError(f"expected an RGB or grayscale image, got shape {image.shape}")


def dhash(image: np.ndarray, hash_size: int = 8) -> str:
    """Compute a difference (gradient) perceptual hash.

    Implemented internally with NumPy/OpenCV to avoid a third-party dependency
    whose default parameters could change between versions.

    **Investigation only.** dHash is used to cluster visually similar images so
    that content-level leakage can be quantified and (optionally) used as a
    grouping key for split regeneration. It must never drive an automatic
    exclusion: two different eyes of the same patient, or two acquisitions of
    the same eye, can be near-duplicates while remaining legitimately distinct
    training samples.

    Args:
        image: RGB or grayscale image array.
        hash_size: Output hash side length; the digest has ``hash_size**2`` bits.

    Returns:
        Hexadecimal string of length ``ceil(hash_size**2 / 4)``.

    Raises:
        ValueError: If ``hash_size`` is not positive or the image is malformed.
    """
    if hash_size <= 0:
        raise ValueError(f"hash_size must be positive, got {hash_size}")
    gray = _to_gray(image)
    # (hash_size + 1) columns so that the horizontal gradient yields hash_size bits.
    resized = cv2.resize(gray, (hash_size + 1, hash_size), interpolation=cv2.INTER_AREA)
    bits = resized[:, 1:] > resized[:, :-1]
    return _bits_to_hex(bits.flatten())


def phash(image: np.ndarray, hash_size: int = 8, image_size: int = 32) -> str:
    """Compute a DCT-based perceptual hash.

    Complements :func:`dhash`: pHash is robust to mild blur and brightness
    shifts, which matters for a corpus with the illumination variability
    documented in the EDA. Same policy as ``dhash`` - reporting only.

    Args:
        image: RGB or grayscale image array.
        hash_size: Side length of the retained low-frequency DCT block.
        image_size: Side length the image is resized to before the DCT.

    Returns:
        Hexadecimal string of ``hash_size**2`` bits.

    Raises:
        ValueError: If ``image_size < hash_size`` or either value is not positive.
    """
    if hash_size <= 0 or image_size <= 0:
        raise ValueError("hash_size and image_size must be positive")
    if image_size < hash_size:
        raise ValueError(f"image_size ({image_size}) must be >= hash_size ({hash_size})")

    gray = _to_gray(image)
    resized = cv2.resize(gray, (image_size, image_size), interpolation=cv2.INTER_AREA)
    dct = cv2.dct(resized.astype(np.float32))
    block = dct[:hash_size, :hash_size]
    # Exclude the DC term from the median: it encodes mean brightness, which is
    # exactly the nuisance variable this hash should be invariant to.
    coefficients = block.flatten()
    median = float(np.median(coefficients[1:]))
    return _bits_to_hex(coefficients > median)


def _bits_to_hex(bits: Iterable[bool] | np.ndarray) -> str:
    """Pack a boolean sequence into a hexadecimal string (MSB first)."""
    array = np.asarray(list(bits), dtype=bool) if not isinstance(bits, np.ndarray) else bits.astype(bool)
    padding = (-array.size) % 8
    if padding:
        array = np.concatenate([array, np.zeros(padding, dtype=bool)])
    packed = np.packbits(array)
    return packed.tobytes().hex()


def hex_to_bits(digest: str) -> np.ndarray:
    """Unpack a hexadecimal digest into a boolean bit array.

    Args:
        digest: Hexadecimal string produced by :func:`dhash` or :func:`phash`.

    Returns:
        A boolean array of bits, MSB first.
    """
    return np.unpackbits(np.frombuffer(bytes.fromhex(digest), dtype=np.uint8)).astype(bool)


def hamming_distance(digest_a: str, digest_b: str) -> int:
    """Return the Hamming distance between two hexadecimal digests.

    Args:
        digest_a: First digest.
        digest_b: Second digest.

    Returns:
        Number of differing bits.

    Raises:
        ValueError: If the digests have different lengths.
    """
    if len(digest_a) != len(digest_b):
        raise ValueError(f"digest length mismatch: {len(digest_a)} vs {len(digest_b)}")
    a = int(digest_a, 16)
    b = int(digest_b, 16)
    return int((a ^ b).bit_count())


# ---------------------------------------------------------------------------
# Image loading and measurements
# ---------------------------------------------------------------------------


def read_image_rgb(path: str | Path) -> np.ndarray | None:
    """Decode an image file as an RGB ``uint8`` array.

    OpenCV decodes to BGR; the conversion is done here, once, so that no
    downstream module has to remember the channel order. Returns ``None``
    instead of raising so the audit can record an undecodable file as a data
    point rather than crashing the run.

    Args:
        path: Image file path.

    Returns:
        ``H x W x 3`` RGB array, or ``None`` if the file cannot be decoded.
    """
    # np.fromfile keeps non-ASCII paths working on Windows, where cv2.imread
    # silently fails on such paths.
    try:
        buffer = np.fromfile(str(path), dtype=np.uint8)
    except OSError as exc:
        logger.debug("unreadable file %s: %s", path, exc)
        return None
    if buffer.size == 0:
        return None
    image = cv2.imdecode(buffer, cv2.IMREAD_COLOR)
    if image is None:
        return None
    return cv2.cvtColor(image, cv2.COLOR_BGR2RGB)


def downscale_long_side(image: np.ndarray, long_side: int) -> np.ndarray:
    """Downscale an image so its longest side equals ``long_side``.

    Used for the audit's quality metrics: measuring on a bounded proxy keeps a
    full-corpus pass tractable while preserving the ordering of the metrics.
    Images already smaller than the target are returned unchanged (upscaling
    would fabricate detail).

    Args:
        image: Input image.
        long_side: Target length of the longer side, in pixels.

    Returns:
        The (possibly) downscaled image.

    Raises:
        ValueError: If ``long_side`` is not positive.
    """
    if long_side <= 0:
        raise ValueError(f"long_side must be positive, got {long_side}")
    height, width = image.shape[:2]
    longest = max(height, width)
    if longest <= long_side:
        return image
    scale = long_side / float(longest)
    new_size = (max(1, int(round(width * scale))), max(1, int(round(height * scale))))
    return cv2.resize(image, new_size, interpolation=cv2.INTER_AREA)


def image_brightness(image: np.ndarray) -> float:
    """Mean grayscale intensity in ``[0, 255]``.

    Args:
        image: RGB or grayscale image.

    Returns:
        Mean intensity.
    """
    return float(_to_gray(image).mean())


def image_contrast(image: np.ndarray) -> float:
    """Grayscale standard deviation, used as a global contrast proxy.

    Args:
        image: RGB or grayscale image.

    Returns:
        Standard deviation of intensity.
    """
    return float(_to_gray(image).std())


def laplacian_variance(image: np.ndarray) -> float:
    """Raw Laplacian variance (classic focus measure).

    Args:
        image: RGB or grayscale image.

    Returns:
        Variance of the Laplacian response.

    Note:
        The EDA showed this quantity correlates r ~= -0.80 with image width,
        because unnormalized Laplacian variance scales with pixel-grid density.
        Use :func:`normalized_sharpness` for any thresholding decision.
    """
    return float(cv2.Laplacian(_to_gray(image), cv2.CV_64F).var())


def normalized_sharpness(image: np.ndarray, reference_long_side: int = 512) -> float:
    """Resolution-normalized sharpness.

    The image is first resized to a fixed reference scale, so the measurement
    reflects optical focus rather than acquisition resolution. This matters
    because resolution itself correlates with DR grade in APTOS (r = 0.57), so
    a raw sharpness threshold would systematically flag high-resolution - and
    therefore grade-biased - images as blurry.

    Args:
        image: RGB or grayscale image.
        reference_long_side: Common scale used for the comparison.

    Returns:
        Laplacian variance at the reference scale, divided by the squared
        dynamic range so that the value is also contrast-normalized.
    """
    gray = _to_gray(image)
    resized = cv2.resize(
        gray,
        (reference_long_side, reference_long_side),
        interpolation=cv2.INTER_AREA if max(gray.shape) > reference_long_side else cv2.INTER_LINEAR,
    )
    variance = float(cv2.Laplacian(resized, cv2.CV_64F).var())
    spread = float(resized.std()) or 1.0
    return variance / (spread**2)


def estimate_noise_sigma(image: np.ndarray) -> float:
    """Estimate additive Gaussian noise sigma (Immerkaer, 1996).

    Convolves with a Laplacian-like kernel that is orthogonal to smooth image
    content, so the residual magnitude is dominated by noise rather than by
    structure.

    Args:
        image: RGB or grayscale image.

    Returns:
        Estimated noise standard deviation in intensity units.
    """
    gray = _to_gray(image).astype(np.float64)
    height, width = gray.shape
    if height < 3 or width < 3:
        return 0.0
    kernel = np.array([[1, -2, 1], [-2, 4, -2], [1, -2, 1]], dtype=np.float64)
    response = np.abs(cv2.filter2D(gray, -1, kernel))
    sigma = response.sum() / (36.0 * (width - 2) * (height - 2)) * np.sqrt(0.5 * np.pi)
    return float(sigma)


def tissue_bounding_box(
    image: np.ndarray, threshold: int = 10, blur_kernel: int = 5
) -> tuple[int, int, int, int] | None:
    """Locate the fundus disc as an axis-aligned bounding box.

    The retinal disc is the only bright region in a fundus photograph; the
    surrounding black padding is an artefact of the camera export pipeline. A
    threshold plus largest-connected-component search is robust to the sensor
    noise and vignetting present in this corpus, and far cheaper than circle
    fitting.

    Args:
        image: RGB or grayscale image.
        threshold: Intensity at or below which a pixel is treated as background.
        blur_kernel: Odd kernel size used to smooth the mask (never the output).

    Returns:
        ``(x, y, width, height)`` of the disc, or ``None`` if no tissue is found
        (which happens for a fully black image and must be handled by the caller
        rather than silently cropped to nothing).

    Raises:
        ValueError: If ``blur_kernel`` is not a positive odd integer.
    """
    if blur_kernel <= 0 or blur_kernel % 2 == 0:
        raise ValueError(f"blur_kernel must be a positive odd integer, got {blur_kernel}")

    gray = _to_gray(image)
    # Median blur suppresses isolated hot pixels without eroding the disc edge.
    smoothed = cv2.medianBlur(gray, blur_kernel)
    mask = (smoothed > threshold).astype(np.uint8)
    if not mask.any():
        return None

    count, labels, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)
    if count <= 1:
        return None
    # Label 0 is the background component.
    largest = 1 + int(np.argmax(stats[1:, cv2.CC_STAT_AREA]))
    x = int(stats[largest, cv2.CC_STAT_LEFT])
    y = int(stats[largest, cv2.CC_STAT_TOP])
    width = int(stats[largest, cv2.CC_STAT_WIDTH])
    height = int(stats[largest, cv2.CC_STAT_HEIGHT])
    return x, y, width, height


def black_padding_ratio(image: np.ndarray, threshold: int = 10) -> float:
    """Fraction of the frame occupied by background (non-tissue) pixels.

    Quantifies the wasted-capacity problem that motivates black-border removal:
    wide-aspect APTOS images can spend a third of their pixels on padding.

    Args:
        image: RGB or grayscale image.
        threshold: Intensity at or below which a pixel counts as background.

    Returns:
        A ratio in ``[0, 1]``.
    """
    gray = _to_gray(image)
    return float((gray <= threshold).mean())
