import torch
from torchvision.transforms import v2
from sklearn.metrics import roc_curve, auc
from torch.utils.data import DataLoader
import numpy as np

from data import ImagePairsDataset
from model import ImageEmbeding

def calculate_accuracy(dataset: ImagePairsDataset, model: ImageEmbeding, threshold: float):
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
        correct += ((model.predict_is_same_scene(img1, img2, threshold) == label).sum().item())
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

    for img1, img2, label in dataset:
        distance = model.predict(img1, img2).cpu().numpy()
        assert distance.shape[0] == label.shape[0]
        distances.append(distance.cpu().numpy())
        labels.append(label)

    distances = np.concatenate(distances)
    labels = np.concatenate(labels)



if __name__ == "__main__":
    model = ImageEmbeding(input_shape=(3, 224, 224), cnn_model="simple").load("checkpoint_zh95kop8_magic-wave-51.pth")
    transform = v2.Compose(
            [v2.Resize((224, 224)), v2.ToDtype(torch.float32, scale=True)]
        )
    dataset = ImagePairsDataset("/mnt/c/Users/kkasr/Downloads/tanks_and_templates/", transform=transform, max_img_per_class=10)
    accuracy = calculate_accuracy(dataset,model, threshold=0.5)
    print(f"Accuracy: {accuracy}")
