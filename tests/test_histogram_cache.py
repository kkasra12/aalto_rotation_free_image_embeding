import os
import sqlite3
import tempfile
import unittest
from contextlib import closing
from pathlib import Path
from unittest.mock import Mock

import numpy as np

from GUI.histogram_cache import (
    CACHE_FORMAT_VERSION,
    CACHE_FORMAT_VERSION_FIELD,
    CACHE_TABLE,
    HistogramCacheKey,
    HistogramData,
    get_model_last_change_time,
    get_or_create_histogram_data,
    histogram_data_from_distribution,
)


class HistogramCacheTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary_directory = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary_directory.name).resolve()
        self.cache_root = self.root / "cache"
        self.database_path = self.root / "GUI" / "histogram_cache.sqlite3"
        self.key = HistogramCacheKey(
            backbone_model="resnet18",
            distance_function="cosine",
            model_name="run-id_run-name",
            last_change_time="100",
        )
        self.histogram_data = HistogramData(
            bin_edges=np.array([0.0, 0.5, 1.0]),
            same_class_heights=np.array([0.25, 1.75]),
            different_class_heights=np.array([1.5, 0.5]),
            midpoint_threshold=0.55,
        )

    def tearDown(self) -> None:
        self.temporary_directory.cleanup()

    def _fetch_rows(self) -> list[tuple[str, str, str, str, str]]:
        with closing(sqlite3.connect(self.database_path)) as connection:
            return connection.execute(
                f"""
                SELECT backbone_model, distance_function, model_name,
                       last_change_time, data_uuid
                FROM {CACHE_TABLE}
                ORDER BY rowid
                """
            ).fetchall()

    def test_first_call_creates_exact_schema_row_and_uuid_archive(self) -> None:
        calculator = Mock(return_value=self.histogram_data)

        actual, outcome = get_or_create_histogram_data(
            self.cache_root,
            self.database_path,
            self.key,
            calculator,
        )

        self.assertEqual(outcome, "cached")
        calculator.assert_called_once_with()
        np.testing.assert_allclose(actual.bin_edges, self.histogram_data.bin_edges)

        with closing(sqlite3.connect(self.database_path)) as connection:
            columns = [
                row[1]
                for row in connection.execute(
                    f"PRAGMA table_info({CACHE_TABLE})"
                ).fetchall()
            ]
        self.assertEqual(
            columns,
            [
                "backbone_model",
                "distance_function",
                "model_name",
                "last_change_time",
                "data_uuid",
            ],
        )

        rows = self._fetch_rows()
        self.assertEqual(len(rows), 1)
        self.assertEqual(rows[0][:4], tuple(self.key.__dict__.values()))
        archive_path = self.cache_root / f"{rows[0][4]}.npz"
        self.assertTrue(archive_path.is_file())
        with np.load(archive_path, allow_pickle=False) as archive:
            self.assertEqual(
                set(archive.files),
                {
                    "bin_edges",
                    "same_class_heights",
                    "different_class_heights",
                    "midpoint_threshold",
                    CACHE_FORMAT_VERSION_FIELD,
                },
            )
            self.assertEqual(archive[CACHE_FORMAT_VERSION_FIELD].shape, ())
            self.assertEqual(
                int(archive[CACHE_FORMAT_VERSION_FIELD]),
                CACHE_FORMAT_VERSION,
            )

    def test_unchanged_key_loads_archive_without_calculation(self) -> None:
        get_or_create_histogram_data(
            self.cache_root,
            self.database_path,
            self.key,
            lambda: self.histogram_data,
        )
        calculator = Mock(side_effect=AssertionError("calculator must not run"))

        actual, outcome = get_or_create_histogram_data(
            self.cache_root,
            self.database_path,
            self.key,
            calculator,
        )

        self.assertEqual(outcome, "loaded")
        calculator.assert_not_called()
        np.testing.assert_allclose(
            actual.same_class_heights,
            self.histogram_data.same_class_heights,
        )

    def test_changed_model_version_preserves_history(self) -> None:
        get_or_create_histogram_data(
            self.cache_root,
            self.database_path,
            self.key,
            lambda: self.histogram_data,
        )
        changed_key = HistogramCacheKey(
            backbone_model=self.key.backbone_model,
            distance_function=self.key.distance_function,
            model_name=self.key.model_name,
            last_change_time="200",
        )

        _, outcome = get_or_create_histogram_data(
            self.cache_root,
            self.database_path,
            changed_key,
            lambda: self.histogram_data,
        )

        self.assertEqual(outcome, "cached")
        rows = self._fetch_rows()
        self.assertEqual(len(rows), 2)
        self.assertNotEqual(rows[0][4], rows[1][4])
        self.assertEqual({row[3] for row in rows}, {"100", "200"})
        self.assertTrue(all((self.cache_root / f"{row[4]}.npz").is_file() for row in rows))

    def test_selection_fields_have_separate_records(self) -> None:
        keys = [
            self.key,
            HistogramCacheKey("resnet50", "cosine", "run-id_run-name", "100"),
            HistogramCacheKey("resnet18", "euclidean", "run-id_run-name", "100"),
            HistogramCacheKey("resnet18", "cosine", "another-run", "100"),
        ]
        for key in keys:
            get_or_create_histogram_data(
                self.cache_root,
                self.database_path,
                key,
                lambda: self.histogram_data,
            )

        rows = self._fetch_rows()
        self.assertEqual(len(rows), len(keys))
        self.assertEqual(len({row[4] for row in rows}), len(keys))

    def test_corrupt_archive_is_rebuilt_under_same_uuid(self) -> None:
        get_or_create_histogram_data(
            self.cache_root,
            self.database_path,
            self.key,
            lambda: self.histogram_data,
        )
        original_uuid = self._fetch_rows()[0][4]
        archive_path = self.cache_root / f"{original_uuid}.npz"
        archive_path.write_bytes(b"not a NumPy archive")
        calculator = Mock(return_value=self.histogram_data)

        _, outcome = get_or_create_histogram_data(
            self.cache_root,
            self.database_path,
            self.key,
            calculator,
        )

        self.assertEqual(outcome, "cached")
        calculator.assert_called_once_with()
        self.assertEqual(self._fetch_rows()[0][4], original_uuid)
        with np.load(archive_path, allow_pickle=False) as archive:
            self.assertIn("bin_edges", archive.files)

    def test_missing_archive_is_rebuilt_under_same_uuid(self) -> None:
        get_or_create_histogram_data(
            self.cache_root,
            self.database_path,
            self.key,
            lambda: self.histogram_data,
        )
        original_uuid = self._fetch_rows()[0][4]
        archive_path = self.cache_root / f"{original_uuid}.npz"
        archive_path.unlink()

        _, outcome = get_or_create_histogram_data(
            self.cache_root,
            self.database_path,
            self.key,
            lambda: self.histogram_data,
        )

        self.assertEqual(outcome, "cached")
        self.assertEqual(self._fetch_rows()[0][4], original_uuid)
        self.assertTrue(archive_path.is_file())

    def test_legacy_archive_is_swapped_and_rewritten_without_calculation(self) -> None:
        get_or_create_histogram_data(
            self.cache_root,
            self.database_path,
            self.key,
            lambda: self.histogram_data,
        )
        original_uuid = self._fetch_rows()[0][4]
        archive_path = self.cache_root / f"{original_uuid}.npz"
        np.savez_compressed(
            archive_path,
            bin_edges=self.histogram_data.bin_edges,
            # The legacy format named these arrays using the inverted labels.
            same_class_heights=self.histogram_data.different_class_heights,
            different_class_heights=self.histogram_data.same_class_heights,
            midpoint_threshold=np.asarray(
                self.histogram_data.midpoint_threshold,
                dtype=float,
            ),
        )
        calculator = Mock(side_effect=AssertionError("calculator must not run"))

        actual, outcome = get_or_create_histogram_data(
            self.cache_root,
            self.database_path,
            self.key,
            calculator,
        )

        self.assertEqual(outcome, "loaded")
        calculator.assert_not_called()
        self.assertEqual(self._fetch_rows()[0][4], original_uuid)
        np.testing.assert_allclose(
            actual.same_class_heights,
            self.histogram_data.same_class_heights,
        )
        np.testing.assert_allclose(
            actual.different_class_heights,
            self.histogram_data.different_class_heights,
        )
        with np.load(archive_path, allow_pickle=False) as archive:
            self.assertEqual(
                int(archive[CACHE_FORMAT_VERSION_FIELD]),
                CACHE_FORMAT_VERSION,
            )
            np.testing.assert_allclose(
                archive["same_class_heights"],
                self.histogram_data.same_class_heights,
            )
            np.testing.assert_allclose(
                archive["different_class_heights"],
                self.histogram_data.different_class_heights,
            )

    def test_unknown_or_non_scalar_archive_version_is_recalculated(self) -> None:
        invalid_versions = (
            np.asarray(CACHE_FORMAT_VERSION + 1, dtype=np.int64),
            np.asarray([CACHE_FORMAT_VERSION], dtype=np.int64),
            np.asarray(CACHE_FORMAT_VERSION, dtype=float),
        )
        for index, invalid_version in enumerate(invalid_versions):
            with self.subTest(version=invalid_version):
                key = HistogramCacheKey(
                    backbone_model=self.key.backbone_model,
                    distance_function=self.key.distance_function,
                    model_name=f"{self.key.model_name}-{index}",
                    last_change_time=self.key.last_change_time,
                )
                get_or_create_histogram_data(
                    self.cache_root,
                    self.database_path,
                    key,
                    lambda: self.histogram_data,
                )
                data_uuid = self._fetch_rows()[-1][4]
                archive_path = self.cache_root / f"{data_uuid}.npz"
                np.savez_compressed(
                    archive_path,
                    bin_edges=self.histogram_data.bin_edges,
                    same_class_heights=self.histogram_data.same_class_heights,
                    different_class_heights=(
                        self.histogram_data.different_class_heights
                    ),
                    midpoint_threshold=np.asarray(
                        self.histogram_data.midpoint_threshold,
                        dtype=float,
                    ),
                    cache_format_version=invalid_version,
                )
                calculator = Mock(return_value=self.histogram_data)

                _, outcome = get_or_create_histogram_data(
                    self.cache_root,
                    self.database_path,
                    key,
                    calculator,
                )

                self.assertEqual(outcome, "cached")
                calculator.assert_called_once_with()
                with np.load(archive_path, allow_pickle=False) as archive:
                    self.assertEqual(
                        int(archive[CACHE_FORMAT_VERSION_FIELD]),
                        CACHE_FORMAT_VERSION,
                    )

    def test_relative_cache_root_falls_back_without_caching(self) -> None:
        calculator = Mock(return_value=self.histogram_data)

        actual, outcome = get_or_create_histogram_data(
            Path("relative-cache"),
            self.database_path,
            self.key,
            calculator,
        )

        self.assertEqual(outcome, "uncached")
        calculator.assert_called_once_with()
        self.assertIs(actual, self.histogram_data)
        self.assertFalse(self.database_path.exists())

    def test_cache_path_that_is_a_file_falls_back_without_caching(self) -> None:
        self.cache_root.write_text("not a directory", encoding="utf-8")
        calculator = Mock(return_value=self.histogram_data)

        _, outcome = get_or_create_histogram_data(
            self.cache_root,
            self.database_path,
            self.key,
            calculator,
        )

        self.assertEqual(outcome, "uncached")
        calculator.assert_called_once_with()

    def test_distribution_conversion_matches_numpy_histogram(self) -> None:
        distribution = np.array(
            [
                [0.1, 0],
                [0.2, 0],
                [0.7, 1],
                [0.9, 1],
            ],
            dtype=float,
        )

        data = histogram_data_from_distribution(distribution)

        expected_same, _ = np.histogram(
            distribution[:2, 0],
            bins=data.bin_edges,
            density=True,
        )
        expected_different, _ = np.histogram(
            distribution[2:, 0],
            bins=data.bin_edges,
            density=True,
        )
        np.testing.assert_allclose(data.different_class_heights, expected_different)
        np.testing.assert_allclose(data.same_class_heights, expected_same)
        self.assertAlmostEqual(data.midpoint_threshold, 0.475)

    def test_checkpoint_version_uses_nanosecond_modification_time(self) -> None:
        checkpoint = self.root / "checkpoint_test.pth"
        checkpoint.write_bytes(b"model")
        expected_time = 1_700_000_000_123_456_700
        os.utime(checkpoint, ns=(expected_time, expected_time))

        self.assertEqual(
            get_model_last_change_time(checkpoint),
            str(checkpoint.stat().st_mtime_ns),
        )


if __name__ == "__main__":
    unittest.main()
