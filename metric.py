from collections.abc import Callable
import json
from pathlib import Path

import torch
from torchvision.transforms import v2
from sklearn.metrics import roc_curve, auc
from torch.utils.data import DataLoader
import numpy as np
import matplotlib.pyplot as plt

from data import ImagePairsDataset
from model import ImageEmbeding


def calculate_accuracy(
    dataset: ImagePairsDataset, model: ImageEmbeding, threshold: float
):
    """

    Args:
        dataset (ImagePairsDataset): _dataset containing pairs of images and labels indicating whether they are from the same scene or not
        model (ImageEmbeding): _model to be evaluated
        threshold (float): threshold for deciding whether two images are from the same scene based on the distance between their embeddings
    """
    correct = 0
    total = 0
    dataloader = DataLoader(dataset, batch_size=8, shuffle=False)
    for img1, img2, label in dataloader:
        img1 = img1.to(model.device, dtype=torch.float32)
        img2 = img2.to(model.device, dtype=torch.float32)
        label = label.to(model.device)
        # if model.predict_is_same_scene(img1, img2, threshold) == label:
        correct += (
            (model.predict_is_same_scene(img1, img2, threshold) == label).sum().item()
        )
        total += label.size(0)
    return correct / total if total > 0 else 0


def calculate_AOC(dataset: ImagePairsDataset, model: ImageEmbeding):
    """
    Calculates the Area Under the Curve (AUC) for the given dataset and model.

    Args:
        dataset (ImagePairsDataset): _dataset containing pairs of images and labels indicating whether they are from the same scene or not
        model (ImageEmbeding): _model to be evaluated
    Returns:
        float: AUC value
        list[float]: list of false positive rates (can be used for plotting the ROC curve)
    """
    distances = []
    labels = []
    dataloader = DataLoader(dataset, batch_size=8, shuffle=False)
    for img1, img2, label in dataloader:
        img1 = img1.to(model.device, dtype=torch.float32)
        img2 = img2.to(model.device, dtype=torch.float32)
        label = label.to(model.device)
        distance = model.predict(img1, img2).cpu().numpy()
        assert distance.shape[0] == label.shape[0], (
            "Distance and label batch sizes do not match"
        )
        distances.append(distance)
        labels.append(label)

    distances = np.concatenate(distances)
    labels = np.concatenate(labels)
    fpr, tpr, _ = roc_curve(labels, distances)
    return auc(fpr, tpr), fpr


def calculate_full_distance_distribution(
    dataset: ImagePairsDataset, model: ImageEmbeding
):
    """
    Calculates the full distance distribution for the given dataset and model.

    Args:
        dataset (ImagePairsDataset): _dataset containing pairs of images and labels indicating whether they are from the same scene or not
        model (ImageEmbeding): _model to be evaluated
    Returns:
        2-column numpy array: the first column contains the distances between image pairs, and the second column contains the corresponding labels (1 for same scene, 0 for different scenes)
    """
    distances = []
    labels = []
    dataloader = DataLoader(dataset, batch_size=8, shuffle=False)
    for img1, img2, label in dataloader:
        img1 = img1.to(model.device, dtype=torch.float32)
        img2 = img2.to(model.device, dtype=torch.float32)
        label = label.to(model.device)
        distance = model.predict(img1, img2).cpu().numpy()
        assert distance.shape[0] == label.shape[0], (
            "Distance and label batch sizes do not match"
        )
        distances.append(distance)
        labels.append(label)

    distances = np.concatenate(distances)
    labels = np.concatenate(labels)

    return np.column_stack((distances, labels))


def plot_distance_distribution(
    data: np.ndarray,
    title: str = "Similarity Score Distribution",
    kind: str = "histogram",
    save_path: Path | None = None,
):
    """
    Plots the distribution of similarity scores separated by label.

    Args:
        data: 2-column array from calculate_full_distance_distribution —
              col 0 = similarity scores, col 1 = binary labels (1=same scene, 0=different)
        title: plot title
        kind: "histogram" or "violin"
        save_path: if provided, saves the figure to this path instead of displaying it
    """
    scores = data[:, 0]
    labels = data[:, 1].astype(int)

    same = scores[labels == 1]
    diff = scores[labels == 0]

    _, ax = plt.subplots(figsize=(8, 5))

    if kind == "histogram":
        bins = np.linspace(scores.min(), scores.max(), 25)
        ax.hist(
            diff,
            bins=bins,
            alpha=0.6,
            color="#e05c5c",
            label="Different scene",
            density=True,
        )
        ax.hist(
            same,
            bins=bins,
            alpha=0.6,
            color="#4c9be8",
            label="Same scene",
            density=True,
        )
        threshold = (same.mean() + diff.mean()) / 2
        ax.axvline(
            threshold,
            color="black",
            linestyle="--",
            linewidth=1.2,
            label=f"Midpoint threshold ({threshold:.2f})",
        )
        ax.set_xlabel("Similarity score", fontsize=12)
        ax.set_ylabel("Density", fontsize=12)
        ax.legend(fontsize=11)
    elif kind == "violin":
        parts = ax.violinplot([diff, same], positions=[0, 1], showmedians=True)
        colors = ["#e05c5c", "#4c9be8"]
        for body, color in zip(parts["bodies"], colors):  # type: ignore[arg-type]
            body.set_facecolor(color)
            body.set_alpha(0.7)
        for key in ("cmedians", "cmins", "cmaxes", "cbars"):
            parts[key].set_color("black")
            parts[key].set_linewidth(1.2)
        ax.set_xticks([0, 1])
        ax.set_xticklabels(["Different scene", "Same scene"], fontsize=12)
        ax.set_ylabel("Similarity score", fontsize=12)
    else:
        raise ValueError(f"kind must be 'histogram' or 'violin', got '{kind}'")

    ax.set_title(title, fontsize=13)
    ax.spines[["top", "right"]].set_visible(False)

    plt.tight_layout()
    if save_path is not None:
        plt.savefig(save_path, dpi=150)
        plt.close()
    else:
        plt.show()


def create_metric_table(
    dataset: ImagePairsDataset,
    checkpoint_dir: Path,
    metric_func: Callable[[ImagePairsDataset, ImageEmbeding], float],
):
    """
    Creates a metric table for the given dataset and checkpoint directory.

    Args:
        dataset (ImagePairsDataset):
        checkpoint_dir (Path): directory containing the model checkpoints. each folder should be in form of <cnn_model>_<distance_metric>/checkpoint<...>.pth
        metric_func (Callable[[ImagePairsDataset, ImageEmbeding], float]): the metric function to be evaluated, e.g. calculate_accuracy or calculate_AOC
    Returns:
        dict: a dictionary containing the pairwise (cnn_model, distance_metric) as keys and their corresponding metric values as values
    """
    metric_table = {}
    for checkpoint_file in checkpoint_dir.rglob("checkpoint_*.pth"):
        cnn_model, distance_metric = checkpoint_file.parent.name.split("_")
        print(
            f"file name is {checkpoint_file} Evaluating {cnn_model} with {distance_metric}..."
        )
        model = ImageEmbeding(
            input_shape=(3, 224, 224), cnn_model="simple", distance=distance_metric
        ).load(checkpoint_file)
        metric_value = metric_func(dataset, model)
        print(f"Metric value for {cnn_model} with {distance_metric}: {metric_value}")
        metric_table[(cnn_model, distance_metric)] = metric_value
    return metric_table


def dict_to_latex_table(table: dict, caption: str = "", label: str = "") -> str:
    """
    Converts a metric dict keyed by (cnn_model, distance_metric) into a LaTeX table string.

    Args:
        table: dict with (cnn_model, distance_metric) keys and float values
        caption: LaTeX table caption
        label: LaTeX table label for \\ref{}
    Returns:
        LaTeX table as a string
    """
    rows: dict[str, dict[str, float]] = {}
    for (cnn_model, distance_metric), value in table.items():
        rows.setdefault(cnn_model, {})[distance_metric] = value

    all_metrics = sorted({dm for (_, dm) in table})
    col_spec = "l" + "c" * len(all_metrics)
    header = " & ".join(["Model"] + all_metrics)

    lines = [
        "\\begin{table}[h]",
        "  \\centering",
        f"  \\begin{{tabular}}{{{col_spec}}}",
        "    \\hline",
        f"    {header} \\\\",
        "    \\hline",
    ]
    for cnn_model, metric_vals in sorted(rows.items()):
        vals = " & ".join(
            f"{metric_vals.get(dm, float('nan')):.4f}" for dm in all_metrics
        )
        lines.append(f"    {cnn_model} & {vals} \\\\")
    lines += [
        "    \\hline",
        "  \\end{tabular}",
        f"  \\caption{{{caption}}}",
        f"  \\label{{{label}}}",
        "\\end{table}",
    ]
    return "\n".join(lines)


def evaluate_all_checkpoints(
    dataset: ImagePairsDataset,
    checkpoint_dir: Path,
    output_dir: Path,
    accuracy_threshold: float = 0.5,
    plot_kind: str = "histogram",
    save_format: str | None = None,
) -> tuple[dict, dict]:
    """
    Evaluates every checkpoint in checkpoint_dir, saves a distribution plot for each,
    and returns accuracy and AUC tables.

    Args:
        dataset: dataset of image pairs with same/different scene labels
        checkpoint_dir: directory of checkpoints; each subfolder named <cnn_model>_<distance_metric>
        output_dir: directory where plots and tables are saved (created if missing)
        accuracy_threshold: cosine/euclidean threshold passed to calculate_accuracy
        plot_kind: "histogram" or "violin"
        save_format: None (no tables saved), "json", or "tex"
    Returns:
        (acc_table, auc_table) — dicts keyed by (cnn_model, distance_metric)
    """
    if save_format not in (None, "json", "tex"):
        raise ValueError(f"save_format must be None, 'json', or 'tex', got '{save_format}'")

    output_dir.mkdir(parents=True, exist_ok=True)
    acc_table: dict = {}
    auc_table: dict = {}

    for checkpoint_file in checkpoint_dir.rglob("checkpoint_*.pth"):
        cnn_model, distance_metric = checkpoint_file.parent.name.split("_")
        key = (cnn_model, distance_metric)
        print(f"Evaluating {cnn_model} + {distance_metric} ...")

        model = ImageEmbeding(
            input_shape=(3, 224, 224), cnn_model=cnn_model, distance=distance_metric
        ).load(checkpoint_file)

        acc_table[key] = calculate_accuracy(dataset, model, accuracy_threshold)
        auc_value, _ = calculate_AOC(dataset, model)
        auc_table[key] = auc_value

        dist_data = calculate_full_distance_distribution(dataset, model)
        plot_path = output_dir / f"{cnn_model}_{distance_metric}.png"
        plot_distance_distribution(
            dist_data,
            title=f"{cnn_model} + {distance_metric}",
            kind=plot_kind,
            save_path=plot_path,
        )
        print(f"  Acc: {acc_table[key]:.4f}  AUC: {auc_table[key]:.4f}  -> {plot_path}")

    if save_format == "json":
        serialisable_acc = {f"{c}_{d}": v for (c, d), v in acc_table.items()}
        serialisable_auc = {f"{c}_{d}": v for (c, d), v in auc_table.items()}
        (output_dir / "accuracy.json").write_text(json.dumps(serialisable_acc, indent=2))
        (output_dir / "auc.json").write_text(json.dumps(serialisable_auc, indent=2))
        print(f"Saved tables -> {output_dir / 'accuracy.json'}, {output_dir / 'auc.json'}")
    elif save_format == "tex":
        acc_tex = dict_to_latex_table(acc_table, caption="Accuracy", label="tab:accuracy")
        auc_tex = dict_to_latex_table(auc_table, caption="AUC", label="tab:auc")
        (output_dir / "accuracy.tex").write_text(acc_tex)
        (output_dir / "auc.tex").write_text(auc_tex)
        print(f"Saved tables -> {output_dir / 'accuracy.tex'}, {output_dir / 'auc.tex'}")

    return acc_table, auc_table


if __name__ == "__main__":
    model = ImageEmbeding(
        input_shape=(3, 224, 224), cnn_model="simple", device="cpu"
    ).load("checkpoints/simple_euclidean/checkpoint_od9ge61m_balmy-hill-70.pth")

    # model = ImageEmbeding(
    #     input_shape=(3, 224, 224), cnn_model="resnet18", device="cpu"
    # ).load("checkpoints/resnet18_cosine/checkpoint_5yggxf8r_prime-water-52.pth")
    transform = v2.Compose(
        [v2.Resize((224, 224)), v2.ToDtype(torch.float32, scale=True)]
    )
    dataset = ImagePairsDataset(
        "C:/Users/kkasr/Downloads/tanks_and_templates/",
        transform=transform,
        max_img_per_class=10,
    )
    acc_table, auc_table = evaluate_all_checkpoints(
        dataset,
        checkpoint_dir=Path("checkpoints"),
        output_dir=Path("plots"),
        plot_kind="histogram",
        save_format="tex",
    )
    print("Accuracy table:", acc_table)
    print("AUC table:", auc_table)
