"""Gradio interface for the rotation-free image comparison project.

The project root must contain a ``.env`` file with absolute paths::

    DATASET_ROOT=C:/absolute/path/to/dataset
    CHECKPOINT_ROOT=C:/absolute/path/to/checkpoints

Optional runtime settings::

    MODEL_DEVICE=auto
    MAX_IMAGES_PER_CLASS=10

Histogram cache settings::

    HISTOGRAM_CACHE_ROOT=C:/absolute/path/to/histogram/cache

Each direct child directory of ``DATASET_ROOT`` is treated as one image class.
Checkpoint subdirectories must follow ``<backbone>_<distance>`` and contain
files matching ``checkpoint_*.pth``.
"""

import logging
from collections import OrderedDict
from dataclasses import dataclass
from datetime import datetime
from functools import lru_cache
from pathlib import Path
from threading import RLock

import gradio as gr
import numpy as np
import torch
from matplotlib.figure import Figure
from PIL import Image
from torchvision.transforms import v2

from data import ImagePairsDataset
from GUI.histogram_cache import (
    HistogramCacheKey,
    HistogramData,
    get_model_last_change_time,
    get_or_create_histogram_data,
    histogram_data_from_distribution,
)
from GUI.settings import (
    GUISettings,
    get_cache_directory_status,
    get_existing_directory_status,
)
from metric import calculate_full_distance_distribution
from model import ImageEmbeding


GUI_DIRECTORY = Path(__file__).resolve().parent
LOGGER = logging.getLogger(__name__)
HISTOGRAM_CACHE_DATABASE = GUI_DIRECTORY / "histogram_cache.sqlite3"
GUI_SETTINGS = GUISettings()

SUPPORTED_IMAGE_EXTENSIONS = {
    ".bmp",
    ".jpeg",
    ".jpg",
    ".png",
    ".tif",
    ".tiff",
    ".webp",
}

# Only the latest selected model is retained. The Gradio state stores its path,
# not the PyTorch model itself, avoiding a large per-session copy.
MODEL_CACHE: OrderedDict[str, ImageEmbeding] = OrderedDict()
MODEL_LOCK = RLock()


@dataclass(eq=False, frozen=True)
class HistogramViewState:
    """Per-session data needed to redraw one checkpoint's histogram."""

    checkpoint_path: str
    checkpoint_last_change_time: str
    backbone_model: str
    distance_function: str
    title: str
    histogram_data: HistogramData


# ---------------------------------------------------------------------------
# Checkpoint discovery and model evaluation
# ---------------------------------------------------------------------------


def parse_checkpoint_directory(directory: Path) -> tuple[str, str] | None:
    """Parse a ``<backbone>_<distance>`` checkpoint directory name."""
    if not directory.is_dir() or "_" not in directory.name:
        return None
    if not any(directory.glob("checkpoint_*.pth")):
        return None

    backbone, distance_function = directory.name.rsplit("_", maxsplit=1)
    if not backbone or not distance_function:
        return None
    return backbone, distance_function


def discover_backbones(checkpoint_root: Path) -> list[str]:
    """Return all CNN backbones dynamically discovered from checkpoint folders."""
    backbones = {
        parsed[0]
        for directory in checkpoint_root.iterdir()
        if (parsed := parse_checkpoint_directory(directory)) is not None
    }
    return sorted(backbones, key=str.casefold)


def discover_distance_functions(
    checkpoint_root: Path,
    backbone: str,
) -> list[str]:
    """Return available distance functions for a selected backbone."""
    distance_functions = {
        parsed[1]
        for directory in checkpoint_root.iterdir()
        if (parsed := parse_checkpoint_directory(directory)) is not None
        and parsed[0] == backbone
    }
    return sorted(distance_functions, key=str.casefold)


def checkpoint_timestamp(checkpoint_path: Path) -> float:
    """Return creation time when supported, otherwise modification time."""
    file_status = checkpoint_path.stat()
    return float(getattr(file_status, "st_birthtime", file_status.st_mtime))


def discover_checkpoints(
    checkpoint_root: Path,
    backbone: str,
    distance_function: str,
) -> list[Path]:
    """Return matching checkpoints ordered from newest to oldest."""
    checkpoint_directory = checkpoint_root / f"{backbone}_{distance_function}"
    if not checkpoint_directory.is_dir():
        return []

    return sorted(
        (
            checkpoint.resolve()
            for checkpoint in checkpoint_directory.glob("checkpoint_*.pth")
        ),
        key=checkpoint_timestamp,
        reverse=True,
    )


def format_checkpoint_label(checkpoint_path: Path) -> str:
    """Return a checkpoint label containing date and readable model name."""
    created_at = datetime.fromtimestamp(checkpoint_timestamp(checkpoint_path))
    model_name = checkpoint_path.stem.removeprefix("checkpoint_")
    return f"{created_at:%Y-%m-%d %H:%M} — {model_name}"


def checkpoint_dropdown_choices(
    checkpoint_root: Path | None,
    backbone: str | None,
    distance_function: str | None,
) -> list[tuple[str, str]]:
    """Return labelled Gradio choices for the selected checkpoint group."""
    if checkpoint_root is None or backbone is None or distance_function is None:
        return []

    return [
        (format_checkpoint_label(checkpoint), str(checkpoint))
        for checkpoint in discover_checkpoints(
            checkpoint_root,
            backbone,
            distance_function,
        )
    ]


def load_model(
    checkpoint_path: Path,
    backbone: str,
    distance_function: str,
) -> ImageEmbeding:
    """Construct and load the selected image-embedding model."""
    model = ImageEmbeding(
        input_shape=(3, 224, 224),
        cnn_model=backbone,
        distance=distance_function,
        device=GUI_SETTINGS.resolved_model_device,
    ).load(checkpoint_path)
    model.eval()
    return model


def clear_model_cache() -> None:
    """Release the reference to the previously selected model."""
    with MODEL_LOCK:
        MODEL_CACHE.clear()


@lru_cache(maxsize=1)
def create_image_transform() -> v2.Compose:
    """Return the image preprocessing shared by evaluation and pair inference."""
    return v2.Compose(
        [
            v2.Resize((224, 224)),
            v2.ToDtype(torch.float32, scale=True),
        ]
    )


@lru_cache(maxsize=1)
def create_evaluation_dataset(
    dataset_root: Path,
    max_images_per_class: int,
) -> ImagePairsDataset:
    """Create and cache the pair dataset used for model distributions."""
    return ImagePairsDataset(
        str(dataset_root),
        transform=create_image_transform(),
        max_img_per_class=max_images_per_class,
    )


def calculate_distance_histogram_data(
    model: ImageEmbeding,
    dataset_root: Path,
) -> HistogramData:
    """Calculate the numerical values used by a model distance histogram."""
    dataset = create_evaluation_dataset(
        dataset_root,
        GUI_SETTINGS.max_images_per_class,
    )
    with torch.inference_mode():
        distribution = calculate_full_distance_distribution(dataset, model)
    return histogram_data_from_distribution(distribution)


def create_distance_histogram(
    data: HistogramData,
    title: str,
    selected_pair_distance: float | None = None,
) -> Figure:
    """Plot histogram values and optionally mark one selected-pair distance."""
    figure = Figure(figsize=(7.2, 5.2))
    axis = figure.subplots()
    bin_widths = np.diff(data.bin_edges)
    axis.bar(
        data.bin_edges[:-1],
        data.different_class_heights,
        width=bin_widths,
        align="edge",
        color="#ef4444",
        alpha=0.65,
        label="Different class",
    )
    axis.bar(
        data.bin_edges[:-1],
        data.same_class_heights,
        width=bin_widths,
        align="edge",
        color="#3b82f6",
        alpha=0.65,
        label="Same class",
    )

    axis.axvline(
        data.midpoint_threshold,
        color="#111827",
        linewidth=2.2,
        linestyle="--",
        label=f"Midpoint threshold: {data.midpoint_threshold:.3f}",
    )
    if selected_pair_distance is not None:
        selected_pair_distance = float(selected_pair_distance)
        if not np.isfinite(selected_pair_distance):
            raise ValueError("Selected-pair distance must be finite.")
        axis.axvline(
            selected_pair_distance,
            color="#a855f7",
            linewidth=3,
            linestyle="-",
            zorder=6,
            label=f"Selected pair: {selected_pair_distance:.4f}",
        )

        minimum_edge = float(data.bin_edges[0])
        maximum_edge = float(data.bin_edges[-1])
        if not minimum_edge <= selected_pair_distance <= maximum_edge:
            lower_limit = min(minimum_edge, selected_pair_distance)
            upper_limit = max(maximum_edge, selected_pair_distance)
            padding = max((upper_limit - lower_limit) * 0.03, 1e-6)
            axis.set_xlim(lower_limit - padding, upper_limit + padding)

    axis.set_title(title, fontsize=14, fontweight="bold", pad=12)
    axis.set_xlabel("Distance")
    axis.set_ylabel("Density")
    axis.grid(axis="y", alpha=0.18)
    axis.legend(frameon=False, loc="best")
    axis.spines[["top", "right"]].set_visible(False)
    figure.tight_layout()
    return figure


def calculate_pair_distance(
    model: ImageEmbeding,
    left_image: Path,
    right_image: Path,
) -> float:
    """Calculate one model distance using the evaluation preprocessing path."""
    left_path = resolve_dataset_image_path(left_image)
    right_path = resolve_dataset_image_path(right_image)

    def prepare_image(image_path: Path) -> torch.Tensor:
        with Image.open(image_path) as source_image:
            image = v2.ToImage()(source_image.convert("RGB"))
        return (
            create_image_transform()(image)
            .unsqueeze(0)
            .to(model.device, dtype=torch.float32)
        )

    left_tensor = prepare_image(left_path)
    right_tensor = prepare_image(right_path)
    with torch.inference_mode():
        distance_tensor = model.predict(left_tensor, right_tensor)

    if not isinstance(distance_tensor, torch.Tensor) or distance_tensor.numel() != 1:
        raise ValueError("Model prediction must contain exactly one distance.")
    distance = float(distance_tensor.detach().cpu().item())
    if not np.isfinite(distance):
        raise ValueError("Model prediction returned a non-finite distance.")
    return distance


def resolve_dataset_image_path(image_path: str | Path) -> Path:
    """Resolve and validate an image path belonging to the configured dataset."""
    if DATASET_ROOT is None:
        raise ValueError("The dataset directory is not configured.")

    dataset_root = DATASET_ROOT.resolve()
    resolved_path = Path(image_path).resolve()
    try:
        resolved_path.relative_to(dataset_root)
    except ValueError as error:
        raise ValueError("Selected image is outside the configured dataset.") from error

    if not resolved_path.is_file():
        raise ValueError(f"Selected image does not exist: {resolved_path}")
    if resolved_path.suffix.casefold() not in SUPPORTED_IMAGE_EXTENSIONS:
        raise ValueError(f"Unsupported image type: {resolved_path.suffix}")
    return resolved_path


def get_model_for_histogram_view(
    view_state: HistogramViewState,
) -> ImageEmbeding:
    """Return the exact view model, reloading it when another session replaced it."""
    if CHECKPOINT_ROOT is None:
        raise ValueError("The checkpoint directory is not configured.")

    checkpoint_path = Path(view_state.checkpoint_path).resolve()
    valid_checkpoints = discover_checkpoints(
        CHECKPOINT_ROOT,
        view_state.backbone_model,
        view_state.distance_function,
    )
    if checkpoint_path not in valid_checkpoints:
        raise ValueError("The histogram checkpoint is no longer valid.")
    if get_model_last_change_time(checkpoint_path) != (
        view_state.checkpoint_last_change_time
    ):
        raise ValueError(
            "The checkpoint changed; select it again to refresh the histogram."
        )

    cache_key = str(checkpoint_path)
    model = MODEL_CACHE.get(cache_key)
    if model is None:
        model = load_model(
            checkpoint_path,
            view_state.backbone_model,
            view_state.distance_function,
        )
        MODEL_CACHE.clear()
        MODEL_CACHE[cache_key] = model
    else:
        MODEL_CACHE.move_to_end(cache_key)
    return model


def update_pair_inference(
    loaded_checkpoint: str | None,
    view_state: HistogramViewState | None,
    left_image: str | None,
    right_image: str | None,
) -> tuple[float | None, Figure | None, str]:
    """Calculate the selected pair and redraw its marker on the histogram."""
    if view_state is None:
        return None, None, "Choose a checkpoint to compare two images."

    base_histogram = create_distance_histogram(
        view_state.histogram_data,
        view_state.title,
    )
    if loaded_checkpoint is None or (
        str(Path(loaded_checkpoint).resolve()) != view_state.checkpoint_path
    ):
        return None, base_histogram, "The loaded model does not match this histogram."
    if left_image is None or right_image is None:
        return None, base_histogram, "Choose both a left and a right image."

    try:
        with MODEL_LOCK:
            model = get_model_for_histogram_view(view_state)
            distance = calculate_pair_distance(
                model,
                Path(left_image),
                Path(right_image),
            )
        marked_histogram = create_distance_histogram(
            view_state.histogram_data,
            view_state.title,
            selected_pair_distance=distance,
        )
        return distance, marked_histogram, "Selected pair highlighted in purple."
    except (OSError, ValueError) as error:
        LOGGER.warning("Pair inference was skipped: %s", error)
        return None, base_histogram, f"Pair inference failed: `{error}`"
    except Exception as error:  # Keep image browsing usable after model errors.
        LOGGER.exception("Could not calculate selected-pair distance")
        return None, base_histogram, f"Pair inference failed: `{error}`"


def update_backbone_selection(
    backbone: str | None,
) -> tuple[gr.Dropdown, gr.Dropdown, None, None, None, str, None, str]:
    """Update all downstream model controls after a backbone change."""
    clear_model_cache()
    distance_functions = (
        discover_distance_functions(CHECKPOINT_ROOT, backbone)
        if CHECKPOINT_ROOT is not None and backbone is not None
        else []
    )
    selected_distance = distance_functions[0] if distance_functions else None
    checkpoints = checkpoint_dropdown_choices(
        CHECKPOINT_ROOT,
        backbone,
        selected_distance,
    )

    return (
        gr.Dropdown(
            choices=distance_functions,
            value=selected_distance,
            interactive=bool(distance_functions),
        ),
        gr.Dropdown(choices=checkpoints, value=None, interactive=bool(checkpoints)),
        None,
        None,
        None,
        "Choose a checkpoint to load the model and calculate its histogram.",
        None,
        "Choose a checkpoint to compare two images.",
    )


def update_distance_selection(
    backbone: str | None,
    distance_function: str | None,
) -> tuple[gr.Dropdown, None, None, None, str, None, str]:
    """Update checkpoints and clear the loaded model after distance changes."""
    clear_model_cache()
    checkpoints = checkpoint_dropdown_choices(
        CHECKPOINT_ROOT,
        backbone,
        distance_function,
    )
    return (
        gr.Dropdown(choices=checkpoints, value=None, interactive=bool(checkpoints)),
        None,
        None,
        None,
        "Choose a checkpoint to load the model and calculate its histogram.",
        None,
        "Choose a checkpoint to compare two images.",
    )


def load_checkpoint_and_histogram(
    checkpoint_value: str | None,
    backbone: str | None,
    distance_function: str | None,
) -> tuple[
    str | None,
    HistogramViewState | None,
    Figure | None,
    str,
    None,
    str,
]:
    """Load the chosen model and calculate its full distance histogram."""
    if checkpoint_value is None:
        clear_model_cache()
        return (
            None,
            None,
            None,
            "Choose a checkpoint to load the model.",
            None,
            "Choose a checkpoint to compare two images.",
        )

    clear_model_cache()
    if CHECKPOINT_ROOT is None or DATASET_ROOT is None:
        return (
            None,
            None,
            None,
            "Dataset or checkpoint configuration is invalid.",
            None,
            "Pair inference is unavailable.",
        )
    if backbone is None or distance_function is None:
        return (
            None,
            None,
            None,
            "Select a backbone and distance function first.",
            None,
            "Pair inference is unavailable.",
        )

    checkpoint_path = Path(checkpoint_value).resolve()
    valid_checkpoints = discover_checkpoints(
        CHECKPOINT_ROOT,
        backbone,
        distance_function,
    )
    if checkpoint_path not in valid_checkpoints:
        return (
            None,
            None,
            None,
            "The selected checkpoint is not valid.",
            None,
            "Pair inference is unavailable.",
        )

    try:
        with MODEL_LOCK:
            model = load_model(
                checkpoint_path,
                backbone,
                distance_function,
            )
            model_name = checkpoint_path.stem.removeprefix("checkpoint_")
            checkpoint_last_change_time = get_model_last_change_time(checkpoint_path)
            title = f"{backbone} + {distance_function}\n{model_name}"
            if HISTOGRAM_CACHE_ROOT is None:
                histogram_data = calculate_distance_histogram_data(
                    model,
                    DATASET_ROOT,
                )
                cache_outcome = "uncached"
            else:
                cache_key = HistogramCacheKey(
                    backbone_model=backbone,
                    distance_function=distance_function,
                    model_name=model_name,
                    last_change_time=checkpoint_last_change_time,
                )
                assert DATASET_ROOT is not None, (
                    "DATASET_ROOT must be configured to calculate histograms."
                )
                histogram_data, cache_outcome = get_or_create_histogram_data(
                    HISTOGRAM_CACHE_ROOT,
                    HISTOGRAM_CACHE_DATABASE,
                    cache_key,
                    lambda: calculate_distance_histogram_data(
                        model,
                        DATASET_ROOT,
                    ),
                )
            view_state = HistogramViewState(
                checkpoint_path=str(checkpoint_path),
                checkpoint_last_change_time=checkpoint_last_change_time,
                backbone_model=backbone,
                distance_function=distance_function,
                title=title,
                histogram_data=histogram_data,
            )
            histogram = create_distance_histogram(histogram_data, title)

            MODEL_CACHE.clear()
            MODEL_CACHE[str(checkpoint_path)] = model

        cache_status = {
            "loaded": "Histogram loaded from cache.",
            "cached": "Histogram calculated and cached.",
            "uncached": "Histogram calculated without caching.",
        }[cache_outcome]
        if cache_outcome == "uncached" and HISTOGRAM_CACHE_ROOT is None:
            cache_status = f"{cache_status} {HISTOGRAM_CACHE_STATUS}"
        status = (
            f"Model loaded on `{GUI_SETTINGS.resolved_model_device}`: "
            f"**{format_checkpoint_label(checkpoint_path)}**  \n"
            f"{cache_status}"
        )
        return (
            str(checkpoint_path),
            view_state,
            histogram,
            status,
            None,
            "Ready to compare the selected images.",
        )
    except Exception as error:  # Gradio must remain usable after a failed load.
        LOGGER.exception("Could not load checkpoint %s", checkpoint_path)
        MODEL_CACHE.clear()
        return (
            None,
            None,
            None,
            f"Model loading failed: `{error}`",
            None,
            "Pair inference is unavailable.",
        )


# ---------------------------------------------------------------------------
# Dataset image browsing
# ---------------------------------------------------------------------------


def list_dataset_classes(dataset_root: Path) -> list[str]:
    """Return sorted names of direct class directories in the dataset."""
    return sorted(
        (
            child.name
            for child in dataset_root.iterdir()
            if child.is_dir() and not child.name.startswith(".")
        ),
        key=str.casefold,
    )


def list_class_images(dataset_root: Path, class_name: str) -> list[Path]:
    """Return sorted image paths belonging to one direct dataset class."""
    resolved_root = dataset_root.resolve()
    class_directory = (resolved_root / class_name).resolve()

    if class_directory.parent != resolved_root or not class_directory.is_dir():
        raise ValueError(f"Invalid dataset class: {class_name!r}")

    return sorted(
        (
            image_path
            for image_path in class_directory.iterdir()
            if image_path.is_file()
            and image_path.suffix.casefold() in SUPPORTED_IMAGE_EXTENSIONS
        ),
        key=lambda path: path.name.casefold(),
    )


def to_gallery_items(image_paths: list[Path]) -> list[tuple[str, str]]:
    """Convert image paths to Gradio gallery items with filename captions."""
    # TODO: Generate and cache real thumbnail files (or paginate the gallery).
    # Gradio currently receives every original image path in the selected class;
    # CSS displays them at thumbnail size, but the full-resolution files may be
    # transferred to and retained by the browser.
    return [(str(image_path), image_path.name) for image_path in image_paths]


def update_image_gallery(
    class_name: str | None,
) -> tuple[
    list[tuple[str, str]],
    list[str],
    str | None,
    str | None,
]:
    """Update one gallery and select its first image after a class change."""
    if DATASET_ROOT is None or class_name is None:
        return [], [], None, None

    image_paths = list_class_images(DATASET_ROOT, class_name)
    serialized_paths = [str(image_path) for image_path in image_paths]
    selected_image = serialized_paths[0] if serialized_paths else None
    return (
        to_gallery_items(image_paths),
        serialized_paths,
        selected_image,
        selected_image,
    )


def select_gallery_image(
    image_paths: list[str],
    event: gr.SelectData,
) -> tuple[str | None, str | None]:
    """Return the path selected from a Gradio thumbnail gallery."""
    if not image_paths or event.index is None:
        return None, None

    raw_index = (
        event.index[0] if isinstance(event.index, (tuple, list)) else event.index
    )
    selected_index = int(raw_index)
    if selected_index < 0 or selected_index >= len(image_paths):
        return None, None
    selected_path = image_paths[selected_index]
    return selected_path, selected_path


def build_image_panel(side: str, class_names: list[str]) -> gr.State:
    """Render one image panel and return its selected-filepath state."""
    initial_class = class_names[0] if class_names else None
    initial_paths = (
        list_class_images(DATASET_ROOT, initial_class)
        if DATASET_ROOT is not None and initial_class is not None
        else []
    )
    serialized_paths = [str(image_path) for image_path in initial_paths]

    gr.Markdown(f"### {side} image")
    class_dropdown = gr.Dropdown(
        choices=class_names,
        value=initial_class,
        label="Image class",
        interactive=True,
    )
    gallery = gr.Gallery(
        value=to_gallery_items(initial_paths),
        label="Choose an image",
        columns=3,
        rows=2,
        height=250,
        object_fit="cover",
        allow_preview=False,
    )
    image_path_state = gr.State(value=serialized_paths)
    selected_image_path_state = gr.State(
        value=serialized_paths[0] if serialized_paths else None
    )
    selected_image = gr.Image(
        value=serialized_paths[0] if serialized_paths else None,
        label="Selected image",
        interactive=False,
        height=300,
    )

    class_dropdown.change(
        fn=update_image_gallery,
        inputs=class_dropdown,
        outputs=[
            gallery,
            image_path_state,
            selected_image,
            selected_image_path_state,
        ],
    )
    gallery.select(
        fn=select_gallery_image,
        inputs=image_path_state,
        outputs=[selected_image, selected_image_path_state],
    )
    return selected_image_path_state


# ---------------------------------------------------------------------------
# Interface
# ---------------------------------------------------------------------------


CUSTOM_CSS = """
.gradio-container {
    max-width: 1500px !important;
    margin: 0 auto !important;
}

.app-header {
    margin-bottom: 8px;
}

.status-line {
    color: #64748b;
    font-size: 0.92rem;
}

.main-panel {
    border: 1px solid #e2e8f0;
    border-radius: 14px;
    padding: 14px;
    background: #ffffff;
}
"""


DATASET_DIRECTORY_STATUS = get_existing_directory_status(
    GUI_SETTINGS.dataset_root,
    "DATASET_ROOT",
    "Dataset",
)
DATASET_ROOT = DATASET_DIRECTORY_STATUS.path
DATASET_STATUS = DATASET_DIRECTORY_STATUS.message

CHECKPOINT_DIRECTORY_STATUS = get_existing_directory_status(
    GUI_SETTINGS.checkpoint_root,
    "CHECKPOINT_ROOT",
    "Checkpoints",
)
CHECKPOINT_ROOT = CHECKPOINT_DIRECTORY_STATUS.path
CHECKPOINT_STATUS = CHECKPOINT_DIRECTORY_STATUS.message

HISTOGRAM_CACHE_DIRECTORY_STATUS = get_cache_directory_status(
    GUI_SETTINGS.histogram_cache_root,
)
HISTOGRAM_CACHE_ROOT = HISTOGRAM_CACHE_DIRECTORY_STATUS.path
HISTOGRAM_CACHE_STATUS = HISTOGRAM_CACHE_DIRECTORY_STATUS.message

DATASET_CLASSES = list_dataset_classes(DATASET_ROOT) if DATASET_ROOT else []
BACKBONES = discover_backbones(CHECKPOINT_ROOT) if CHECKPOINT_ROOT else []
INITIAL_BACKBONE = BACKBONES[0] if BACKBONES else None
INITIAL_DISTANCES = (
    discover_distance_functions(CHECKPOINT_ROOT, INITIAL_BACKBONE)
    if CHECKPOINT_ROOT is not None and INITIAL_BACKBONE is not None
    else []
)
INITIAL_DISTANCE = INITIAL_DISTANCES[0] if INITIAL_DISTANCES else None
INITIAL_CHECKPOINTS = checkpoint_dropdown_choices(
    CHECKPOINT_ROOT,
    INITIAL_BACKBONE,
    INITIAL_DISTANCE,
)

if DATASET_ROOT is not None and not DATASET_CLASSES:
    DATASET_STATUS = f"No class directories were found in: {DATASET_ROOT}"
if CHECKPOINT_ROOT is not None and not BACKBONES:
    CHECKPOINT_STATUS = f"No checkpoint groups were found in: {CHECKPOINT_ROOT}"


with gr.Blocks(title="Rotation-Free Image Comparison") as demo:
    gr.Markdown(
        f"""
        # Rotation-Free Image Comparison
        <div class="status-line">{DATASET_STATUS}</div>
        <div class="status-line">{CHECKPOINT_STATUS}</div>
        <div class="status-line">{HISTOGRAM_CACHE_STATUS}</div>
        """,
        elem_classes="app-header",
    )

    loaded_checkpoint_state = gr.State(value=None)
    histogram_view_state = gr.State(value=None)

    with gr.Group(elem_classes="main-panel"):
        gr.Markdown("### Model selection")
        with gr.Row(equal_height=True):
            backbone_dropdown = gr.Dropdown(
                choices=BACKBONES,
                value=INITIAL_BACKBONE,
                label="CNN backbone",
                interactive=bool(BACKBONES),
            )
            distance_dropdown = gr.Dropdown(
                choices=INITIAL_DISTANCES,
                value=INITIAL_DISTANCE,
                label="Distance function",
                interactive=bool(INITIAL_DISTANCES),
            )
            checkpoint_dropdown = gr.Dropdown(
                choices=INITIAL_CHECKPOINTS,
                value=None,
                label="Checkpoint",
                interactive=bool(INITIAL_CHECKPOINTS),
            )

    with gr.Row(equal_height=False):
        with gr.Column(scale=3, min_width=320, elem_classes="main-panel"):
            left_selected_image_state = build_image_panel(
                "Left",
                DATASET_CLASSES,
            )

        with gr.Column(scale=4, min_width=420, elem_classes="main-panel"):
            gr.Markdown("### Pair distance")
            pair_distance = gr.Number(
                value=None,
                label="Selected-pair distance",
                precision=4,
                interactive=False,
            )
            pair_status = gr.Markdown("Choose a checkpoint to compare two images.")
            model_status = gr.Markdown(
                "Choose a checkpoint to load the model and calculate its histogram."
            )
            histogram_plot = gr.Plot(value=None, label="Distance distribution")

        with gr.Column(scale=3, min_width=320, elem_classes="main-panel"):
            right_selected_image_state = build_image_panel(
                "Right",
                DATASET_CLASSES,
            )

    backbone_dropdown.change(
        fn=update_backbone_selection,
        inputs=backbone_dropdown,
        outputs=[
            distance_dropdown,
            checkpoint_dropdown,
            loaded_checkpoint_state,
            histogram_view_state,
            histogram_plot,
            model_status,
            pair_distance,
            pair_status,
        ],
        concurrency_limit=1,
        concurrency_id="model-inference",
    )
    distance_dropdown.change(
        fn=update_distance_selection,
        inputs=[backbone_dropdown, distance_dropdown],
        outputs=[
            checkpoint_dropdown,
            loaded_checkpoint_state,
            histogram_view_state,
            histogram_plot,
            model_status,
            pair_distance,
            pair_status,
        ],
        concurrency_limit=1,
        concurrency_id="model-inference",
    )
    checkpoint_dropdown.change(
        fn=load_checkpoint_and_histogram,
        inputs=[
            checkpoint_dropdown,
            backbone_dropdown,
            distance_dropdown,
        ],
        outputs=[
            loaded_checkpoint_state,
            histogram_view_state,
            histogram_plot,
            model_status,
            pair_distance,
            pair_status,
        ],
        show_progress="full",
        concurrency_limit=1,
        concurrency_id="model-inference",
    )

    pair_inference_inputs = [
        loaded_checkpoint_state,
        histogram_view_state,
        left_selected_image_state,
        right_selected_image_state,
    ]
    pair_inference_outputs = [pair_distance, histogram_plot, pair_status]
    gr.on(
        triggers=[
            histogram_view_state.change,
            left_selected_image_state.change,
            right_selected_image_state.change,
        ],
        fn=update_pair_inference,
        inputs=pair_inference_inputs,
        outputs=pair_inference_outputs,
        show_progress="hidden",
        trigger_mode="always_last",
        concurrency_limit=1,
        concurrency_id="model-inference",
    )


def launch_gui() -> None:
    """Launch the Gradio interface from another entrypoint."""
    logging.basicConfig(level=logging.INFO)
    allowed_paths = [str(DATASET_ROOT)] if DATASET_ROOT is not None else None
    demo.launch(
        theme=gr.themes.Soft(primary_hue="blue", neutral_hue="slate"),
        css=CUSTOM_CSS,
        allowed_paths=allowed_paths,
    )
