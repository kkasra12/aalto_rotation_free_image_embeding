import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from pydantic import ValidationError

from GUI.settings import (
    DirectoryStatus,
    ENV_PATH,
    GUISettings,
    get_cache_directory_status,
    get_existing_directory_status,
)


@patch.dict(os.environ, {}, clear=True)
class GUISettingsTests(unittest.TestCase):
    def test_defaults_and_blank_paths(self) -> None:
        settings = GUISettings(
            _env_file=None,
            dataset_root=" ",
            checkpoint_root="",
            histogram_cache_root=None,
        )

        self.assertIsNone(settings.dataset_root)
        self.assertIsNone(settings.checkpoint_root)
        self.assertIsNone(settings.histogram_cache_root)
        self.assertEqual(settings.max_images_per_class, 10)
        self.assertEqual(settings.model_device, "auto")

    def test_reads_uppercase_values_from_an_explicit_env_file(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory).resolve()
            env_path = root / "settings.env"
            env_path.write_text(
                "\n".join(
                    (
                        f"DATASET_ROOT={root / 'dataset'}",
                        f"CHECKPOINT_ROOT={root / 'checkpoints'}",
                        f"HISTOGRAM_CACHE_ROOT={root / 'cache'}",
                        "MAX_IMAGES_PER_CLASS=23",
                        "MODEL_DEVICE= CUDA:0 ",
                    )
                ),
                encoding="utf-8",
            )

            settings = GUISettings(_env_file=env_path)

        self.assertEqual(settings.dataset_root, root / "dataset")
        self.assertEqual(settings.checkpoint_root, root / "checkpoints")
        self.assertEqual(settings.histogram_cache_root, root / "cache")
        self.assertEqual(settings.max_images_per_class, 23)
        self.assertEqual(settings.model_device, "cuda:0")

    def test_environment_overrides_env_file_and_unknown_keys_are_ignored(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            env_path = Path(temporary_directory) / "settings.env"
            env_path.write_text(
                "MAX_IMAGES_PER_CLASS=12\nUNRELATED_SETTING=ignored\n",
                encoding="utf-8",
            )

            with patch.dict(
                os.environ,
                {"MAX_IMAGES_PER_CLASS": "31"},
            ):
                settings = GUISettings(_env_file=env_path)

        self.assertEqual(settings.max_images_per_class, 31)

    def test_max_images_per_class_must_be_positive_integer(self) -> None:
        for invalid_value in (0, -1, "not-an-integer"):
            with self.subTest(value=invalid_value):
                with self.assertRaises(ValidationError):
                    GUISettings(
                        _env_file=None,
                        max_images_per_class=invalid_value,
                    )

    def test_model_device_must_not_be_blank(self) -> None:
        with self.assertRaisesRegex(ValidationError, "must not be blank"):
            GUISettings(_env_file=None, model_device="  ")

    def test_model_device_is_normalized(self) -> None:
        settings = GUISettings(_env_file=None, model_device=" CUDA:1 ")

        self.assertEqual(settings.model_device, "cuda:1")

    def test_auto_device_uses_cuda_when_available(self) -> None:
        settings = GUISettings(_env_file=None, model_device="auto")

        with patch("GUI.settings.torch.cuda.is_available", return_value=True):
            self.assertEqual(settings.resolved_model_device, "cuda")
        with patch("GUI.settings.torch.cuda.is_available", return_value=False):
            self.assertEqual(settings.resolved_model_device, "cpu")

    def test_configured_device_is_returned_without_auto_resolution(self) -> None:
        settings = GUISettings(_env_file=None, model_device="mps")

        with patch("GUI.settings.torch.cuda.is_available") as is_available:
            self.assertEqual(settings.resolved_model_device, "mps")
            is_available.assert_not_called()


class DirectoryStatusTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary_directory = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary_directory.name).resolve()

    def tearDown(self) -> None:
        self.temporary_directory.cleanup()

    def test_directory_status_is_immutable(self) -> None:
        status = DirectoryStatus(path=None, message="message")

        with self.assertRaises((AttributeError, TypeError)):
            status.message = "changed"  # type: ignore[misc]

    def test_missing_existing_directory_setting_has_clear_status(self) -> None:
        status = get_existing_directory_status(
            None,
            "DATASET_ROOT",
            "Dataset",
        )

        self.assertIsNone(status.path)
        self.assertEqual(
            status.message,
            f"DATASET_ROOT is not set in {ENV_PATH}",
        )

    def test_existing_directory_must_be_absolute(self) -> None:
        status = get_existing_directory_status(
            Path("relative/dataset"),
            "DATASET_ROOT",
            "Dataset",
        )

        self.assertIsNone(status.path)
        self.assertEqual(
            status.message,
            "DATASET_ROOT must be an absolute path.",
        )

    def test_existing_directory_reports_missing_path_and_file(self) -> None:
        missing = self.root / "missing"
        missing_status = get_existing_directory_status(
            missing,
            "DATASET_ROOT",
            "Dataset",
        )
        self.assertIsNone(missing_status.path)
        self.assertEqual(
            missing_status.message,
            f"Dataset directory does not exist: {missing}",
        )

        configured_file = self.root / "dataset.txt"
        configured_file.write_text("not a directory", encoding="utf-8")
        file_status = get_existing_directory_status(
            configured_file,
            "DATASET_ROOT",
            "Dataset",
        )
        self.assertIsNone(file_status.path)
        self.assertEqual(
            file_status.message,
            f"DATASET_ROOT is not a directory: {configured_file}",
        )

    def test_existing_directory_returns_resolved_path(self) -> None:
        dataset_root = self.root / "dataset"
        dataset_root.mkdir()

        status = get_existing_directory_status(
            dataset_root,
            "DATASET_ROOT",
            "Dataset",
        )

        self.assertEqual(status.path, dataset_root.resolve())
        self.assertEqual(
            status.message,
            f"Dataset connected: {dataset_root.resolve()}",
        )

    def test_cache_directory_is_created(self) -> None:
        cache_root = self.root / "nested" / "cache"

        status = get_cache_directory_status(cache_root)

        self.assertEqual(status.path, cache_root.resolve())
        self.assertTrue(cache_root.is_dir())
        self.assertEqual(
            status.message,
            f"Histogram cache connected: {cache_root.resolve()}",
        )

    def test_cache_directory_requires_absolute_path(self) -> None:
        status = get_cache_directory_status(Path("relative/cache"))

        self.assertIsNone(status.path)
        self.assertEqual(
            status.message,
            "HISTOGRAM_CACHE_ROOT must be an absolute path.",
        )

    def test_cache_directory_reports_file_and_os_errors(self) -> None:
        configured_file = self.root / "cache-file"
        configured_file.write_text("not a directory", encoding="utf-8")

        file_status = get_cache_directory_status(configured_file)
        self.assertIsNone(file_status.path)
        self.assertEqual(
            file_status.message,
            f"HISTOGRAM_CACHE_ROOT is not a directory: {configured_file}",
        )

        cache_root = self.root / "unavailable-cache"
        with patch.object(Path, "mkdir", side_effect=OSError("denied")):
            error_status = get_cache_directory_status(cache_root)
        self.assertIsNone(error_status.path)
        self.assertEqual(
            error_status.message,
            "Histogram cache is unavailable: denied",
        )


if __name__ == "__main__":
    unittest.main()
