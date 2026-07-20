import copy
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock, patch

import numpy as np
import torch
from PIL import Image

from GUI.histogram_cache import HistogramData
from GUI import rotation_free_gui_frame as gui


class RecordingModel:
    def __init__(self, distance: float = 0.375) -> None:
        self.device = torch.device("cpu")
        self.distance = distance
        self.inputs: tuple[torch.Tensor, torch.Tensor] | None = None
        self.inference_mode_enabled = False

    def predict(
        self,
        left: torch.Tensor,
        right: torch.Tensor,
    ) -> torch.Tensor:
        self.inputs = left, right
        self.inference_mode_enabled = torch.is_inference_mode_enabled()
        return torch.tensor([self.distance], dtype=torch.float32)


class PairInferenceTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary_directory = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary_directory.name).resolve()
        self.dataset_root = self.root / "dataset"
        self.image_class = self.dataset_root / "scene"
        self.image_class.mkdir(parents=True)
        self.left_image = self.image_class / "left.png"
        self.right_image = self.image_class / "right.png"
        Image.new("RGB", (19, 11), color=(255, 128, 0)).save(self.left_image)
        Image.new("RGB", (13, 17), color=(0, 64, 255)).save(self.right_image)

        self.histogram_data = HistogramData(
            bin_edges=np.array([0.0, 0.5, 1.0]),
            same_class_heights=np.array([0.5, 1.5]),
            different_class_heights=np.array([1.25, 0.75]),
            midpoint_threshold=0.55,
        )
        with gui.MODEL_LOCK:
            self.original_model_cache = gui.MODEL_CACHE.copy()
            gui.MODEL_CACHE.clear()

    def tearDown(self) -> None:
        with gui.MODEL_LOCK:
            gui.MODEL_CACHE.clear()
            gui.MODEL_CACHE.update(self.original_model_cache)
        self.temporary_directory.cleanup()

    def _create_checkpoint_view(
        self,
    ) -> tuple[Path, Path, gui.HistogramViewState]:
        checkpoint_root = self.root / "checkpoints"
        checkpoint_directory = checkpoint_root / "resnet18_cosine"
        checkpoint_directory.mkdir(parents=True)
        checkpoint = checkpoint_directory / "checkpoint_run-id_run-name.pth"
        checkpoint.write_bytes(b"checkpoint")
        view = gui.HistogramViewState(
            checkpoint_path=str(checkpoint.resolve()),
            checkpoint_last_change_time=gui.get_model_last_change_time(checkpoint),
            backbone_model="resnet18",
            distance_function="cosine",
            title="resnet18 + cosine\nrun-id_run-name",
            histogram_data=self.histogram_data,
        )
        return checkpoint_root, checkpoint, view

    def test_pair_inference_uses_training_preprocessing(self) -> None:
        model = RecordingModel()

        with patch.object(gui, "DATASET_ROOT", self.dataset_root):
            distance = gui.calculate_pair_distance(
                model,
                self.left_image,
                self.right_image,
            )

        self.assertAlmostEqual(distance, 0.375)
        self.assertTrue(model.inference_mode_enabled)
        self.assertIsNotNone(model.inputs)
        assert model.inputs is not None
        for image_tensor in model.inputs:
            self.assertEqual(tuple(image_tensor.shape), (1, 3, 224, 224))
            self.assertEqual(image_tensor.dtype, torch.float32)
            self.assertEqual(image_tensor.device.type, "cpu")
            self.assertGreaterEqual(float(image_tensor.min()), 0.0)
            self.assertLessEqual(float(image_tensor.max()), 1.0)

    def test_pair_inference_rejects_paths_outside_dataset(self) -> None:
        outside_image = self.root / "outside.png"
        Image.new("RGB", (8, 8), color="white").save(outside_image)

        with patch.object(gui, "DATASET_ROOT", self.dataset_root):
            with self.assertRaisesRegex(ValueError, "outside"):
                gui.calculate_pair_distance(
                    RecordingModel(),
                    outside_image,
                    self.right_image,
                )

    def test_pair_inference_rejects_unsupported_image_type(self) -> None:
        unsupported_image = self.image_class / "image.txt"
        unsupported_image.write_text("not an image", encoding="utf-8")

        with patch.object(gui, "DATASET_ROOT", self.dataset_root):
            with self.assertRaisesRegex(ValueError, "Unsupported"):
                gui.calculate_pair_distance(
                    RecordingModel(),
                    unsupported_image,
                    self.right_image,
                )

    def test_pair_inference_rejects_missing_and_converts_non_rgb_images(self) -> None:
        grayscale_image = self.image_class / "gray.png"
        Image.new("L", (8, 8), color=128).save(grayscale_image)

        with patch.object(gui, "DATASET_ROOT", self.dataset_root):
            with self.assertRaisesRegex(ValueError, "does not exist"):
                gui.calculate_pair_distance(
                    RecordingModel(),
                    self.image_class / "missing.png",
                    self.right_image,
                )
            model = RecordingModel()
            gui.calculate_pair_distance(
                model,
                grayscale_image,
                self.right_image,
            )

        assert model.inputs is not None
        self.assertEqual(tuple(model.inputs[0].shape), (1, 3, 224, 224))

    def test_pair_inference_rejects_non_finite_model_output(self) -> None:
        with patch.object(gui, "DATASET_ROOT", self.dataset_root):
            with self.assertRaisesRegex(ValueError, "non-finite"):
                gui.calculate_pair_distance(
                    RecordingModel(float("nan")),
                    self.left_image,
                    self.right_image,
                )

    def test_pair_inference_rejects_multiple_model_outputs(self) -> None:
        model = RecordingModel()
        model.predict = Mock(return_value=torch.tensor([0.1, 0.2]))

        with patch.object(gui, "DATASET_ROOT", self.dataset_root):
            with self.assertRaisesRegex(ValueError, "exactly one"):
                gui.calculate_pair_distance(
                    model,
                    self.left_image,
                    self.right_image,
                )

    def test_histogram_marker_uses_exact_value_and_does_not_accumulate(self) -> None:
        original_data = copy.deepcopy(self.histogram_data)

        first = gui.create_distance_histogram(
            self.histogram_data,
            "first",
            selected_pair_distance=0.25,
        )
        second = gui.create_distance_histogram(
            self.histogram_data,
            "second",
            selected_pair_distance=0.75,
        )

        self.assertIsNot(first, second)
        self.assertEqual(len(first.axes[0].lines), 2)
        self.assertEqual(len(second.axes[0].lines), 2)
        np.testing.assert_allclose(first.axes[0].lines[1].get_xdata(), [0.25, 0.25])
        np.testing.assert_allclose(second.axes[0].lines[1].get_xdata(), [0.75, 0.75])
        self.assertGreater(
            second.axes[0].lines[1].get_zorder(),
            second.axes[0].lines[0].get_zorder(),
        )
        self.assertIn("Selected pair: 0.7500", second.axes[0].lines[1].get_label())
        np.testing.assert_array_equal(
            self.histogram_data.same_class_heights,
            original_data.same_class_heights,
        )
        np.testing.assert_array_equal(
            self.histogram_data.different_class_heights,
            original_data.different_class_heights,
        )

    def test_out_of_range_marker_is_visible_without_clamping(self) -> None:
        figure = gui.create_distance_histogram(
            self.histogram_data,
            "out of range",
            selected_pair_distance=2.5,
        )

        axis = figure.axes[0]
        np.testing.assert_allclose(axis.lines[1].get_xdata(), [2.5, 2.5])
        self.assertLess(axis.get_xlim()[0], 0.0)
        self.assertGreater(axis.get_xlim()[1], 2.5)

    def test_non_finite_marker_is_rejected(self) -> None:
        with self.assertRaisesRegex(ValueError, "finite"):
            gui.create_distance_histogram(
                self.histogram_data,
                "invalid",
                selected_pair_distance=float("inf"),
            )

    def test_pair_callback_uses_exact_cached_model(self) -> None:
        checkpoint_root, checkpoint, view = self._create_checkpoint_view()
        model = object()
        gui.MODEL_CACHE[str(checkpoint.resolve())] = model

        with (
            patch.object(gui, "CHECKPOINT_ROOT", checkpoint_root),
            patch.object(gui, "load_model") as load_model,
            patch.object(gui, "calculate_pair_distance", return_value=0.42) as calculate,
        ):
            distance, figure, status = gui.update_pair_inference(
                str(checkpoint),
                view,
                str(self.left_image),
                str(self.right_image),
            )

        self.assertEqual(distance, 0.42)
        self.assertIsNotNone(figure)
        self.assertIn("purple", status)
        load_model.assert_not_called()
        calculate.assert_called_once_with(
            model,
            self.left_image,
            self.right_image,
        )

    def test_pair_callback_reloads_exact_model_if_global_cache_changed(self) -> None:
        checkpoint_root, checkpoint, view = self._create_checkpoint_view()
        stale_model = object()
        reloaded_model = object()
        gui.MODEL_CACHE["another-checkpoint"] = stale_model

        with (
            patch.object(gui, "CHECKPOINT_ROOT", checkpoint_root),
            patch.object(gui, "load_model", return_value=reloaded_model) as load_model,
            patch.object(gui, "calculate_pair_distance", return_value=0.61) as calculate,
        ):
            distance, _, _ = gui.update_pair_inference(
                str(checkpoint),
                view,
                str(self.left_image),
                str(self.right_image),
            )

        self.assertEqual(distance, 0.61)
        load_model.assert_called_once_with(
            checkpoint.resolve(),
            "resnet18",
            "cosine",
        )
        calculate.assert_called_once_with(
            reloaded_model,
            self.left_image,
            self.right_image,
        )
        self.assertEqual(list(gui.MODEL_CACHE), [str(checkpoint.resolve())])

    def test_pair_callback_refuses_stale_checkpoint_version(self) -> None:
        checkpoint_root, checkpoint, view = self._create_checkpoint_view()
        stale_view = gui.HistogramViewState(
            checkpoint_path=view.checkpoint_path,
            checkpoint_last_change_time="stale-version",
            backbone_model=view.backbone_model,
            distance_function=view.distance_function,
            title=view.title,
            histogram_data=view.histogram_data,
        )

        with (
            patch.object(gui, "CHECKPOINT_ROOT", checkpoint_root),
            patch.object(gui, "calculate_pair_distance") as calculate,
        ):
            distance, figure, status = gui.update_pair_inference(
                str(checkpoint),
                stale_view,
                str(self.left_image),
                str(self.right_image),
            )

        self.assertIsNone(distance)
        self.assertIsNotNone(figure)
        self.assertIn("checkpoint changed", status.casefold())
        calculate.assert_not_called()

    def test_pair_callback_clears_marker_when_prerequisites_are_missing(self) -> None:
        _, checkpoint, view = self._create_checkpoint_view()

        distance, figure, status = gui.update_pair_inference(
            str(checkpoint),
            view,
            None,
            str(self.right_image),
        )

        self.assertIsNone(distance)
        self.assertIsNotNone(figure)
        self.assertEqual(len(figure.axes[0].lines), 1)
        self.assertIn("both", status)

    def test_pair_callback_refuses_loaded_checkpoint_identity_mismatch(self) -> None:
        _, _, view = self._create_checkpoint_view()

        with patch.object(gui, "calculate_pair_distance") as calculate:
            distance, figure, status = gui.update_pair_inference(
                str(self.root / "another-checkpoint.pth"),
                view,
                str(self.left_image),
                str(self.right_image),
            )

        self.assertIsNone(distance)
        self.assertIsNotNone(figure)
        self.assertEqual(len(figure.axes[0].lines), 1)
        self.assertIn("does not match", status)
        calculate.assert_not_called()

    def test_gallery_callbacks_update_preview_and_selected_path_state(self) -> None:
        with patch.object(gui, "DATASET_ROOT", self.dataset_root):
            _, image_paths, preview, selected_state = gui.update_image_gallery(
                "scene"
            )

        self.assertEqual(preview, selected_state)
        self.assertIn(preview, image_paths)
        selected_preview, selected_path = gui.select_gallery_image(
            image_paths,
            SimpleNamespace(index=1),
        )
        self.assertEqual(selected_preview, image_paths[1])
        self.assertEqual(selected_path, image_paths[1])

    def test_gradio_graph_updates_pair_for_histogram_and_both_images(self) -> None:
        pair_dependencies = [
            dependency
            for dependency in gui.demo.config["dependencies"]
            if dependency["api_name"].startswith("update_pair_inference")
        ]

        self.assertEqual(len(pair_dependencies), 1)
        dependency = pair_dependencies[0]
        self.assertEqual(dependency["trigger_mode"], "always_last")
        self.assertEqual(len(dependency["targets"]), 3)
        self.assertEqual(len(dependency["inputs"]), 4)
        self.assertEqual(len(dependency["outputs"]), 3)

        relevant_functions = {
            block_function.fn.__name__: block_function
            for block_function in gui.demo.fns.values()
            if block_function.fn is not None
            and block_function.fn.__name__
            in {
                "update_backbone_selection",
                "update_distance_selection",
                "load_checkpoint_and_histogram",
                "update_pair_inference",
            }
        }
        self.assertEqual(
            set(relevant_functions),
            {
                "update_backbone_selection",
                "update_distance_selection",
                "load_checkpoint_and_histogram",
                "update_pair_inference",
            },
        )
        for block_function in relevant_functions.values():
            self.assertEqual(block_function.concurrency_id, "model-inference")
            self.assertEqual(block_function.concurrency_limit, 1)


if __name__ == "__main__":
    unittest.main()
