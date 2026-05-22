from collections.abc import Callable
from pathlib import Path

import torch
from torchvision.transforms import v2
from sklearn.metrics import roc_curve, auc
from torch.utils.data import DataLoader
import numpy as np

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


if __name__ == "__main__":
    model = ImageEmbeding(
        input_shape=(3, 224, 224), cnn_model="simple", device="cpu"
    ).load("checkpoint_zh95kop8_magic-wave-51.pth")

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
    # # accuracy = calculate_accuracy(dataset, model, threshold=0.5)
    # # print(f"Accuracy: {accuracy}")

    # # auc_value, fpr = calculate_AOC(dataset, model)
    # # print(f"AUC: {auc_value}")
    # # print(f"False Positive Rates: {fpr}")

    acc_table = create_metric_table(
        dataset,
        Path("checkpoints"),
        lambda d, m: calculate_accuracy(d, m, threshold=0.5),
    )
    print(f"Accuracy Table: {acc_table}")
