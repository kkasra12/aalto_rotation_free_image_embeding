"""Gradio interface for the rotation-free image comparison project.

The project root must contain a ``.env`` file with absolute paths::

    DATASET_ROOT=C:/absolute/path/to/dataset
    CHECKPOINT_ROOT=C:/absolute/path/to/checkpoints

Optional runtime settings::

    MODEL_DEVICE=auto
    MAX_IMAGES_PER_CLASS=10

Each direct child directory of ``DATASET_ROOT`` is treated as one image class.
Checkpoint subdirectories must follow ``<backbone>_<distance>`` and contain
files matching ``checkpoint_*.pth``.
"""

import logging
import os
import sys
from collections import OrderedDict
from datetime import datetime
from functools import lru_cache
from pathlib import Path
from threading import RLock
from typing import Any

import gradio as gr
import matplotlib.pyplot as plt
import numpy as np
from dotenv import load_dotenv
from matplotlib.figure import Figure


LOGGER = logging.getLogger(__name__)

GUI_DIRECTORY = Path(__file__).resolve().parent
PROJECT_ROOT = GUI_DIRECTORY.parent
ENV_PATH = PROJECT_ROOT / ".env"

# Running ``python .\GUI\rotation_free_gui_frame.py`` places GUI, rather than
# the project root, first on sys.path. Add the root so data.py, metric.py, and
# model.py can be imported lazily by the callbacks.
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

load_dotenv(ENV_PATH)

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
MODEL_CACHE: OrderedDict[str, Any] = OrderedDict()
MODEL_LOCK = RLock()


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------


def get_configured_directory(
    environment_variable: str,
    display_name: str,
) -> tuple[Path | None, str]:
    """Read and validate an absolute directory configured in ``.env``.

    Args:
        environment_variable: Name of the environment variable to read.
        display_name: Human-readable directory name used in status messages.

    Returns:
        The validated directory, when available, and a status message.
    """
    configured_path = os.getenv(environment_variable, "").strip()
    if not configured_path:
        return None, f"{environment_variable} is not set in {ENV_PATH}"

    directory = Path(configured_path).expanduser()
    if not directory.is_absolute():
        return None, f"{environment_variable} must be an absolute path."
    if not directory.exists():
        return None, f"{display_name} directory does not exist: {directory}"
    if not directory.is_dir():
        return None, f"{environment_variable} is not a directory: {directory}"

    resolved_directory = directory.resolve()
    return resolved_directory, f"{display_name} connected: {resolved_directory}"


def get_max_images_per_class() -> int:
    """Return the positive dataset sampling limit configured in ``.env``."""
    raw_value = os.getenv("MAX_IMAGES_PER_CLASS", "10").strip()
    try:
        value = int(raw_value)
    except ValueError as error:
        raise ValueError("MAX_IMAGES_PER_CLASS must be an integer.") from error

    if value <= 0:
        raise ValueError("MAX_IMAGES_PER_CLASS must be greater than zero.")
    return value


def get_model_device() -> str:
    """Return the configured PyTorch device, resolving ``auto`` dynamically."""
    import torch

    configured_device = os.getenv("MODEL_DEVICE", "auto").strip().casefold()
    if configured_device == "auto":
        return "cuda" if torch.cuda.is_available() else "cpu"
    return configured_device


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
) -> Any:
    """Construct and load the selected image-embedding model."""
    from model import ImageEmbeding

    model = ImageEmbeding(
        input_shape=(3, 224, 224),
        cnn_model=backbone,
        distance=distance_function,
        device=get_model_device(),
    ).load(checkpoint_path)

    if hasattr(model, "eval"):
        model.eval()
    elif hasattr(model, "model") and hasattr(model.model, "eval"):
        model.model.eval()
    return model


def clear_model_cache() -> None:
    """Release the reference to the previously selected model."""
    with MODEL_LOCK:
        MODEL_CACHE.clear()


@lru_cache(maxsize=1)
def create_evaluation_dataset(
    dataset_root: Path,
    max_images_per_class: int,
) -> Any:
    """Create and cache the pair dataset used for model distributions."""
    import torch
    from torchvision.transforms import v2

    from data import ImagePairsDataset

    transform = v2.Compose(
        [
            v2.Resize((224, 224)),
            v2.ToDtype(torch.float32, scale=True),
        ]
    )
    return ImagePairsDataset(
        str(dataset_root),
        transform=transform,
        max_img_per_class=max_images_per_class,
    )


def create_distance_histogram(
    model: Any,
    dataset_root: Path,
    title: str,
) -> Figure:
    """Calculate and plot the model's same/different-class distribution."""
    import torch

    from metric import calculate_full_distance_distribution

    dataset = create_evaluation_dataset(
        dataset_root,
        get_max_images_per_class(),
    )
    with torch.inference_mode():
        distribution = calculate_full_distance_distribution(dataset, model)

    if distribution.ndim != 2 or distribution.shape[1] != 2:
        raise ValueError("Distance distribution must have shape (N, 2).")

    scores = distribution[:, 0]
    labels = distribution[:, 1].astype(int)
    same_class = scores[labels == 1]
    different_class = scores[labels == 0]

    if same_class.size == 0 or different_class.size == 0:
        raise ValueError(
            "The evaluation dataset must contain both same-class and "
            "different-class pairs."
        )

    minimum_score = float(scores.min())
    maximum_score = float(scores.max())
    if np.isclose(minimum_score, maximum_score):
        minimum_score -= 0.5
        maximum_score += 0.5

    figure, axis = plt.subplots(figsize=(7.2, 5.2))
    bins = np.linspace(minimum_score, maximum_score, 25)
    axis.hist(
        different_class,
        bins=bins,
        color="#ef4444",
        alpha=0.65,
        density=True,
        label="Different class",
    )
    axis.hist(
        same_class,
        bins=bins,
        color="#3b82f6",
        alpha=0.65,
        density=True,
        label="Same class",
    )

    threshold = float((same_class.mean() + different_class.mean()) / 2)
    axis.axvline(
        threshold,
        color="#111827",
        linewidth=2.2,
        linestyle="--",
        label=f"Midpoint threshold: {threshold:.3f}",
    )
    axis.set_title(title, fontsize=14, fontweight="bold", pad=12)
    axis.set_xlabel("Distance")
    axis.set_ylabel("Density")
    axis.grid(axis="y", alpha=0.18)
    axis.legend(frameon=False, loc="best")
    axis.spines[["top", "right"]].set_visible(False)
    figure.tight_layout()
    return figure


def calculate_pair_distance(
    model: Any,
    left_image: Path,
    right_image: Path,
) -> float:
    """Calculate the distance between the selected images in a later step."""
    pass


def update_backbone_selection(
    backbone: str | None,
) -> tuple[gr.Dropdown, gr.Dropdown, None, None, str, None]:
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
        "Choose a checkpoint to load the model and calculate its histogram.",
        None,
    )


def update_distance_selection(
    backbone: str | None,
    distance_function: str | None,
) -> tuple[gr.Dropdown, None, None, str, None]:
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
        "Choose a checkpoint to load the model and calculate its histogram.",
        None,
    )


def load_checkpoint_and_histogram(
    checkpoint_value: str | None,
    backbone: str | None,
    distance_function: str | None,
) -> tuple[str | None, Figure | None, str, None]:
    """Load the chosen model and calculate its full distance histogram."""
    if checkpoint_value is None:
        clear_model_cache()
        return None, None, "Choose a checkpoint to load the model.", None

    clear_model_cache()
    if CHECKPOINT_ROOT is None or DATASET_ROOT is None:
        return None, None, "Dataset or checkpoint configuration is invalid.", None
    if backbone is None or distance_function is None:
        return None, None, "Select a backbone and distance function first.", None

    checkpoint_path = Path(checkpoint_value).resolve()
    valid_checkpoints = discover_checkpoints(
        CHECKPOINT_ROOT,
        backbone,
        distance_function,
    )
    if checkpoint_path not in valid_checkpoints:
        return None, None, "The selected checkpoint is not valid.", None

    try:
        with MODEL_LOCK:
            model = load_model(
                checkpoint_path,
                backbone,
                distance_function,
            )
            title = (
                f"{backbone} + {distance_function}\n"
                f"{checkpoint_path.stem.removeprefix('checkpoint_')}"
            )
            histogram = create_distance_histogram(model, DATASET_ROOT, title)

            MODEL_CACHE.clear()
            MODEL_CACHE[str(checkpoint_path)] = model

        status = (
            f"Model loaded on `{get_model_device()}`: "
            f"**{format_checkpoint_label(checkpoint_path)}**"
        )
        return str(checkpoint_path), histogram, status, None
    except Exception as error:  # Gradio must remain usable after a failed load.
        LOGGER.exception("Could not load checkpoint %s", checkpoint_path)
        MODEL_CACHE.clear()
        return None, None, f"Model loading failed: `{error}`", None


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
) -> tuple[list[tuple[str, str]], list[str], str | None]:
    """Update one gallery and select its first image after a class change."""
    if DATASET_ROOT is None or class_name is None:
        return [], [], None

    image_paths = list_class_images(DATASET_ROOT, class_name)
    serialized_paths = [str(image_path) for image_path in image_paths]
    selected_image = serialized_paths[0] if serialized_paths else None
    return to_gallery_items(image_paths), serialized_paths, selected_image


def select_gallery_image(
    image_paths: list[str],
    event: gr.SelectData,
) -> str | None:
    """Return the path selected from a Gradio thumbnail gallery."""
    if not image_paths or event.index is None:
        return None

    raw_index = (
        event.index[0]
        if isinstance(event.index, (tuple, list))
        else event.index
    )
    selected_index = int(raw_index)
    if selected_index < 0 or selected_index >= len(image_paths):
        return None
    return image_paths[selected_index]


def build_image_panel(side: str, class_names: list[str]) -> None:
    """Render and connect one dataset image-selection panel."""
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
    selected_image = gr.Image(
        value=serialized_paths[0] if serialized_paths else None,
        label="Selected image",
        interactive=False,
        height=300,
    )

    class_dropdown.change(
        fn=update_image_gallery,
        inputs=class_dropdown,
        outputs=[gallery, image_path_state, selected_image],
    )
    gallery.select(
        fn=select_gallery_image,
        inputs=image_path_state,
        outputs=selected_image,
    )


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


DATASET_ROOT, DATASET_STATUS = get_configured_directory(
    "DATASET_ROOT",
    "Dataset",
)
CHECKPOINT_ROOT, CHECKPOINT_STATUS = get_configured_directory(
    "CHECKPOINT_ROOT",
    "Checkpoints",
)

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
        """,
        elem_classes="app-header",
    )

    loaded_checkpoint_state = gr.State(value=None)

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
            build_image_panel("Left", DATASET_CLASSES)

        with gr.Column(scale=4, min_width=420, elem_classes="main-panel"):
            gr.Markdown("### Pair distance")
            pair_distance = gr.Number(
                value=None,
                label="Selected-pair distance",
                precision=4,
                interactive=False,
            )
            model_status = gr.Markdown(
                "Choose a checkpoint to load the model and calculate its histogram."
            )
            histogram_plot = gr.Plot(value=None, label="Distance distribution")

        with gr.Column(scale=3, min_width=320, elem_classes="main-panel"):
            build_image_panel("Right", DATASET_CLASSES)

    backbone_dropdown.change(
        fn=update_backbone_selection,
        inputs=backbone_dropdown,
        outputs=[
            distance_dropdown,
            checkpoint_dropdown,
            loaded_checkpoint_state,
            histogram_plot,
            model_status,
            pair_distance,
        ],
    )
    distance_dropdown.change(
        fn=update_distance_selection,
        inputs=[backbone_dropdown, distance_dropdown],
        outputs=[
            checkpoint_dropdown,
            loaded_checkpoint_state,
            histogram_plot,
            model_status,
            pair_distance,
        ],
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
            histogram_plot,
            model_status,
            pair_distance,
        ],
        show_progress="full",
    )


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    allowed_paths = [str(DATASET_ROOT)] if DATASET_ROOT is not None else None
    demo.launch(
        theme=gr.themes.Soft(primary_hue="blue", neutral_hue="slate"),
        css=CUSTOM_CSS,
        allowed_paths=allowed_paths,
    )
