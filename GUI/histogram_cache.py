"""Persistent cache support for GUI distance histograms."""

import logging
import os
import sqlite3
from collections.abc import Callable
from contextlib import closing
from dataclasses import dataclass
from pathlib import Path
from typing import Literal
from uuid import UUID, uuid4
from zipfile import BadZipFile

import numpy as np


LOGGER = logging.getLogger(__name__)

CACHE_TABLE = "histogram_cache"
CACHE_OUTCOME = Literal["loaded", "cached", "uncached"]
CACHE_FORMAT_VERSION = 1
CACHE_FORMAT_VERSION_FIELD = "cache_format_version"

_HISTOGRAM_DATA_FIELDS = {
    "bin_edges",
    "same_class_heights",
    "different_class_heights",
    "midpoint_threshold",
}


@dataclass(frozen=True)
class HistogramCacheKey:
    """Fields that identify one version of a selected checkpoint."""

    backbone_model: str
    distance_function: str
    model_name: str
    last_change_time: str


@dataclass(frozen=True)
class HistogramData:
    """The numerical values required to render a distance histogram."""

    bin_edges: np.ndarray
    same_class_heights: np.ndarray
    different_class_heights: np.ndarray
    midpoint_threshold: float


def get_model_last_change_time(checkpoint_path: Path) -> str:
    """Return the current checkpoint version used by the cache index."""
    # TODO: Replace the modification time with a SHA-256 hash of the checkpoint
    # contents when stronger model-version detection is needed.
    return str(checkpoint_path.stat().st_mtime_ns)


def histogram_data_from_distribution(distribution: np.ndarray) -> HistogramData:
    """Convert a two-column distance distribution into plottable values."""
    if distribution.ndim != 2 or distribution.shape[1] != 2:
        raise ValueError("Distance distribution must have shape (N, 2).")

    scores = np.asarray(distribution[:, 0], dtype=float)
    labels = distribution[:, 1].astype(int)
    same_class = scores[labels == 0]
    different_class = scores[labels == 1]

    if same_class.size == 0 or different_class.size == 0:
        raise ValueError(
            "The evaluation dataset must contain both same-class and "
            "different-class pairs."
        )
    if not np.all(np.isfinite(scores)):
        raise ValueError("Distance distribution contains non-finite scores.")

    minimum_score = float(scores.min())
    maximum_score = float(scores.max())
    if np.isclose(minimum_score, maximum_score):
        minimum_score -= 0.5
        maximum_score += 0.5

    bin_edges = np.linspace(minimum_score, maximum_score, 25)
    different_class_heights, _ = np.histogram(
        different_class,
        bins=bin_edges,
        density=True,
    )
    same_class_heights, _ = np.histogram(
        same_class,
        bins=bin_edges,
        density=True,
    )
    midpoint_threshold = float(
        (same_class.mean() + different_class.mean()) / 2
    )

    return validate_histogram_data(
        HistogramData(
            bin_edges=bin_edges,
            same_class_heights=same_class_heights,
            different_class_heights=different_class_heights,
            midpoint_threshold=midpoint_threshold,
        )
    )


def validate_histogram_data(data: HistogramData) -> HistogramData:
    """Validate and normalize histogram values loaded or calculated locally."""
    bin_edges = np.asarray(data.bin_edges, dtype=float)
    same_class_heights = np.asarray(data.same_class_heights, dtype=float)
    different_class_heights = np.asarray(
        data.different_class_heights,
        dtype=float,
    )
    midpoint_threshold = float(data.midpoint_threshold)

    if bin_edges.ndim != 1 or bin_edges.size < 2:
        raise ValueError("Histogram bin edges must be a one-dimensional array.")
    if not np.all(np.isfinite(bin_edges)) or not np.all(np.diff(bin_edges) > 0):
        raise ValueError("Histogram bin edges must be finite and increasing.")

    expected_height_count = bin_edges.size - 1
    for name, heights in (
        ("same-class", same_class_heights),
        ("different-class", different_class_heights),
    ):
        if heights.ndim != 1 or heights.size != expected_height_count:
            raise ValueError(
                f"The {name} heights must contain one value per histogram bin."
            )
        if not np.all(np.isfinite(heights)) or np.any(heights < 0):
            raise ValueError(f"The {name} heights must be finite and non-negative.")

    if not np.isfinite(midpoint_threshold):
        raise ValueError("The histogram midpoint threshold must be finite.")

    return HistogramData(
        bin_edges=bin_edges,
        same_class_heights=same_class_heights,
        different_class_heights=different_class_heights,
        midpoint_threshold=midpoint_threshold,
    )


def _initialize_database(connection: sqlite3.Connection) -> None:
    connection.execute(
        f"""
        CREATE TABLE IF NOT EXISTS {CACHE_TABLE} (
            backbone_model TEXT NOT NULL,
            distance_function TEXT NOT NULL,
            model_name TEXT NOT NULL,
            last_change_time TEXT NOT NULL,
            data_uuid TEXT NOT NULL UNIQUE,
            UNIQUE (
                backbone_model,
                distance_function,
                model_name,
                last_change_time
            )
        )
        """
    )


def _get_or_create_data_uuid(
    database_path: Path,
    key: HistogramCacheKey,
) -> tuple[str, bool]:
    """Return the UUID for a cache key and whether this call created its row."""
    database_path.parent.mkdir(parents=True, exist_ok=True)
    candidate_uuid = str(uuid4())
    identity = (
        key.backbone_model,
        key.distance_function,
        key.model_name,
        key.last_change_time,
    )

    with closing(sqlite3.connect(database_path, timeout=30)) as connection:
        with connection:
            _initialize_database(connection)
            cursor = connection.execute(
                f"""
                INSERT OR IGNORE INTO {CACHE_TABLE} (
                    backbone_model,
                    distance_function,
                    model_name,
                    last_change_time,
                    data_uuid
                ) VALUES (?, ?, ?, ?, ?)
                """,
                (*identity, candidate_uuid),
            )
            created = cursor.rowcount == 1
            row = connection.execute(
                f"""
                SELECT data_uuid
                FROM {CACHE_TABLE}
                WHERE backbone_model = ?
                  AND distance_function = ?
                  AND model_name = ?
                  AND last_change_time = ?
                """,
                identity,
            ).fetchone()

    if row is None:
        raise sqlite3.DatabaseError("Could not create or retrieve cache index row.")
    return str(row[0]), created


def _archive_path(cache_root: Path, data_uuid: str) -> Path:
    """Return a safe archive path for a UUID stored in the index."""
    normalized_uuid = str(UUID(data_uuid))
    return cache_root / f"{normalized_uuid}.npz"


def _load_archive(archive_path: Path) -> HistogramData:
    is_legacy_archive = False
    with np.load(archive_path, allow_pickle=False) as archive:
        archive_fields = set(archive.files)
        if archive_fields == _HISTOGRAM_DATA_FIELDS:
            # Archives written before CACHE_FORMAT_VERSION stored label 1 as
            # same-class and label 0 as different-class. The dataset's actual
            # semantics are the reverse, so migrate by swapping the heights.
            is_legacy_archive = True
            same_class_heights = archive["different_class_heights"]
            different_class_heights = archive["same_class_heights"]
        elif archive_fields == _HISTOGRAM_DATA_FIELDS | {
            CACHE_FORMAT_VERSION_FIELD
        }:
            version = np.asarray(archive[CACHE_FORMAT_VERSION_FIELD])
            if (
                version.shape != ()
                or not np.issubdtype(version.dtype, np.integer)
                or int(version) != CACHE_FORMAT_VERSION
            ):
                raise ValueError(
                    "Histogram cache archive has an unsupported version."
                )
            same_class_heights = archive["same_class_heights"]
            different_class_heights = archive["different_class_heights"]
        else:
            raise ValueError("Histogram cache archive has unexpected fields.")

        data = HistogramData(
            bin_edges=archive["bin_edges"],
            same_class_heights=same_class_heights,
            different_class_heights=different_class_heights,
            midpoint_threshold=float(archive["midpoint_threshold"]),
        )

    validated_data = validate_histogram_data(data)
    if is_legacy_archive:
        try:
            _write_archive_atomically(archive_path, validated_data)
        except OSError as error:
            # The swapped values are still valid for this request. Avoid an
            # expensive model-wide recalculation when only migration writing
            # is unavailable; a later read can retry the atomic upgrade.
            LOGGER.warning(
                "Could not migrate legacy histogram cache %s: %s",
                archive_path,
                error,
            )
    return validated_data


def _write_archive_atomically(
    archive_path: Path,
    data: HistogramData,
) -> None:
    validated_data = validate_histogram_data(data)
    temporary_path = archive_path.with_name(
        f".{archive_path.stem}.{uuid4().hex}.tmp.npz"
    )
    try:
        with temporary_path.open("wb") as temporary_file:
            np.savez_compressed(
                temporary_file,
                bin_edges=validated_data.bin_edges,
                same_class_heights=validated_data.same_class_heights,
                different_class_heights=validated_data.different_class_heights,
                midpoint_threshold=np.asarray(
                    validated_data.midpoint_threshold,
                    dtype=float,
                ),
                cache_format_version=np.asarray(
                    CACHE_FORMAT_VERSION,
                    dtype=np.int64,
                ),
            )
        os.replace(temporary_path, archive_path)
    finally:
        temporary_path.unlink(missing_ok=True)


def get_or_create_histogram_data(
    cache_root: Path,
    database_path: Path,
    key: HistogramCacheKey,
    calculate: Callable[[], HistogramData],
) -> tuple[HistogramData, CACHE_OUTCOME]:
    """Load cached histogram data or calculate and persist it when necessary."""
    try:
        if not cache_root.is_absolute():
            raise ValueError("HISTOGRAM_CACHE_ROOT must be an absolute path.")
        cache_root.mkdir(parents=True, exist_ok=True)
        data_uuid, _ = _get_or_create_data_uuid(database_path, key)
        archive_path = _archive_path(cache_root, data_uuid)
    except (OSError, ValueError, sqlite3.Error) as error:
        LOGGER.warning("Histogram cache is unavailable: %s", error)
        return calculate(), "uncached"

    try:
        return _load_archive(archive_path), "loaded"
    except (OSError, ValueError, EOFError, BadZipFile):
        pass

    data = calculate()
    try:
        _write_archive_atomically(archive_path, data)
    except (OSError, ValueError) as error:
        LOGGER.warning("Could not write histogram cache %s: %s", archive_path, error)
        return data, "uncached"
    return data, "cached"
