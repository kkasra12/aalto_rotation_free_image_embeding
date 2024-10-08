"""
Here we define the model for the application.
The model is a regular CNN which takes one image as input and outputs an embedding vector.
The model is trained using a contrastive loss function.

We have two available distance functions:
1. Euclidean distance
2. 1 - Cosine distance

According to the "Chen, Ting, et al. "A simple framework for contrastive learning of visual representations." International conference on machine learning. PMLR, 2020." paper,
we can use cross-entropy loss with a softmax layer instead of the contrastive loss function (denoted as `crossentropy` or `ce` in this code).

Accroding to "Dimensionality Reduction by Learning an Invariant Mapping" paper, eq 4, we can use (denoted as `linear` in this code):
L(W, Y, X1, X2) = (1 - Y) * 1/2 D_W^2 + Y/2 {max(0, m - D_W )}^2

where m > 0 is a margin, D_W is the distance between the embeddings of X1 and X2, and Y is the label (0 or 1). Y is 1 if the images are of the same class, 0 otherwise.
"""

import os
from typing import Optional
from rich.progress import Progress, TimeElapsedColumn
import numpy as np
import torch
from torch import nn
import torch.utils
import torch.utils.data
from torchvision import models
import wandb


def find_output_size(model, input_size):
    return model(torch.rand(1, *input_size)).shape[1:]


class ImageEmbeding(nn.Module):
    def __init__(
        self,
        input_shape: tuple[int],
        distance="euclidean",
        embedding_size=128,
        cnn_model="resnet18",
        loss_function="linear",
        loss_margin=1,
        device=None,
    ):
        super(ImageEmbeding, self).__init__()
        if distance not in ["euclidean", "cosine"]:
            raise ValueError(
                f"Distance function {distance} not supported. Choose from ['euclidean', 'cosine']"
            )
        self.distance = distance
        self.embedding_size = embedding_size

        if cnn_model == "resnet18":
            self.preprocess = models.resnet18(
                weights=models.ResNet18_Weights.IMAGENET1K_V1
            )
            self.preprocess.fc = nn.Identity()
            self.preprocess.conv1 = nn.Conv2d(
                in_channels=input_shape[0],
                out_channels=64,
                kernel_size=7,
                stride=2,
                padding=3,
                bias=False,
            )
        elif cnn_model == "resnet50":
            self.preprocess = models.resnet50(pretrained=True)
            self.preprocess.fc = nn.Identity()
            self.preprocess.conv1 = nn.Conv2d(
                in_channels=input_shape[0],
                out_channels=64,
                kernel_size=7,
                stride=2,
                padding=3,
                bias=False,
            )
        elif cnn_model == "simple":
            self.preprocess = nn.Sequential(
                nn.Conv2d(input_shape[0], 32, kernel_size=3, stride=1, padding=1),
                nn.ReLU(),
                nn.MaxPool2d(kernel_size=2, stride=2),
                nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1),
                nn.ReLU(),
                nn.MaxPool2d(kernel_size=2, stride=2),
                nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1),
                nn.ReLU(),
                nn.MaxPool2d(kernel_size=2, stride=2),
                nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=1),
                nn.ReLU(),
                nn.MaxPool2d(kernel_size=2, stride=2),
                nn.Flatten(),
            )
        else:
            raise ValueError(
                f"Preprocess network {cnn_model} not supported. Choose from ['resnet18', 'resnet50', 'simple']"
            )

        self.embedding = nn.Sequential(
            nn.Linear(np.prod(find_output_size(self.preprocess, input_shape)), 512),
            nn.ReLU(),
            nn.Linear(512, self.embedding_size),
        )
        self.distance_function = (
            nn.PairwiseDistance(p=2)
            if distance == "euclidean"
            else lambda x, y: 1 - nn.functional.cosine_similarity(x, y)
        )
        # the input of distance functions is two matrix of size (batch_size, embedding_size)
        # the output of distance functions is (batch_size,) which is the distance between the two embeddings

        if loss_function == "linear":
            self.loss_function = (
                lambda dx, y: (
                    (1 - y) * dx**2 + y * torch.clamp(loss_margin - dx, min=0) ** 2
                )
                / 2
            )
        # the input of loss function is two matrix of size (batch_size, )
        # the output of loss function is (batch_size, ) which is the loss for each pair

        elif loss_function in ["crossentropy", "ce"]:
            raise NotImplementedError("Cross entropy loss not implemented")
            self.loss_function = nn.CrossEntropyLoss()
            # hmmm!!!
        else:
            raise ValueError(
                f"Loss function {loss_function} not supported. Choose from ['linear', 'crossentropy']"
            )

        if device is None:
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.device = device
        self.to(device)

    # def one_forward(self, img: torch.Tensor) -> torch.Tensor:
    #     return self.embedding(self.preprocess(img))

    def forward(self, img: torch.Tensor) -> torch.Tensor:
        # emb1 = self.one_forward(img1)
        # emb2 = self.one_forward(img2)
        # return self.distance_function(emb1, emb2)
        return self.embedding(self.preprocess(img))

    # def train_step(
    #     self, img1: torch.Tensor, img2: torch.Tensor, label: torch.Tensor
    # ) -> torch.Tensor:
    #     """

    #     Args:
    #         img1 (torch.Tensor): Image 1
    #         img2 (torch.Tensor): Image 2
    #         label (torch.Tensor): 0: same class, 1 different class

    #     Returns:
    #         torch.Tensor: Loss
    #     """
    #     dx = self.forward(img1, img2)
    #     return self.loss_function(dx, label)

    def fit(
        self,
        dataloader: torch.utils.data.DataLoader,
        epochs: int,
        test_dataloader: torch.utils.data.DataLoader = None,
        optimizer: Optional[torch.optim.Optimizer | str] = None,
        use_wandb: bool = False,
        checkpoint_path: Optional[str] = None,
        **kwargs,
    ):
        self.train()
        if optimizer is None:
            optimizer = torch.optim.Adam(self.parameters(), lr=kwargs.get("lr", 1e-3))
        elif isinstance(optimizer, str):
            if optimizer == "adam":
                optimizer = torch.optim.Adam(
                    self.parameters(), lr=kwargs.get("lr", 1e-3)
                )
            elif optimizer == "sgd":
                optimizer = torch.optim.SGD(
                    self.parameters(), lr=kwargs.get("lr", 1e-3)
                )
            else:
                raise ValueError(
                    f"Optimizer {optimizer} not supported. Choose from ['adam', 'sgd']"
                )
        else:
            raise ValueError(
                "Optimizer must be a string or torch.optim.Optimizer object"
            )

        if use_wandb:
            wandb.watch(self)

        progress_bar_message = "[bold cyan]Epoch {current_epoch}/{max_epochs}, loss={loss:.4f}, steps={task.completed}/{task.total}[/]"
        for epoch in range(epochs):
            with Progress(
                *Progress.get_default_columns(), TimeElapsedColumn()
            ) as progress:
                pid = progress.add_task("", total=len(dataloader))
                progress.update(
                    advance=0,
                    description=progress_bar_message.format(
                        current_epoch=epoch + 1,
                        max_epochs=epochs,
                        loss=0,
                        task=progress.tasks[pid],
                    ),
                    task_id=pid,
                )
                train_loss = 0
                count = 0
                for img1, img2, label in dataloader:
                    img1 = img1.to(self.device)
                    img2 = img2.to(self.device)
                    label = label.to(self.device)
                    # TODO: add this part to the dataset

                    optimizer.zero_grad()
                    emb1 = self(img1)
                    emb2 = self(img2)
                    distance = self.distance_function(emb1, emb2)
                    loss = self.loss_function(distance, label).mean()
                    loss.backward()
                    optimizer.step()

                    train_loss += loss.item()
                    count += 1
                    progress.update(
                        advance=1,
                        description=progress_bar_message.format(
                            current_epoch=epoch + 1,
                            max_epochs=epochs,
                            loss=loss.item(),
                            task=progress.tasks[pid],
                        ),
                        task_id=pid,
                    )
                loss = train_loss / count
                if use_wandb:
                    wandb.log({"train_loss": loss})
                print(f"Epoch {epoch+1}/{epochs}, Loss: {loss}")
            if test_dataloader:
                self.eval()
                print("Testing...")
                all_loss = []
                distance_same_scene = []
                distance_diff_scene = []
                for img1, img2, label in test_dataloader:
                    img1 = img1.to(self.device)
                    img2 = img2.to(self.device)
                    label = label.to(self.device)
                    distance = self.predict(img1, img2)
                    loss = self.loss_function(distance, label).mean()
                    all_loss.append(loss.item())
                    if 0 in label:
                        distance_same_scene.append(distance[label == 0].mean().item())
                    if 1 in label:
                        distance_diff_scene.append(distance[label == 1].mean().item())
                print(f"Test loss: {np.mean(all_loss)}")
                if use_wandb:
                    wandb.log(
                        {
                            "test_loss": np.mean(all_loss),
                            "distance_same_scene": wandb.Histogram(distance_same_scene),
                            "distance_diff_scene": wandb.Histogram(distance_diff_scene),
                        }
                    )

                self.train()
            if checkpoint_path:
                self.save(
                    os.path.join(
                        checkpoint_path,
                        f"checkpoint_{wandb.run.id}_{wandb.run.name}.pth",
                    )
                )

        if use_wandb:
            wandb.finish()

        return self

    def predict(self, img1: torch.Tensor, img2: torch.Tensor) -> torch.Tensor:
        with torch.no_grad():
            dx = self.distance_function(self(img1), self(img2))
            return dx

    def predict_one(self, img: torch.Tensor) -> torch.Tensor:
        with torch.no_grad():
            return self(img)

    def save(self, path: str | os.PathLike):
        torch.save(self.state_dict(), path)

    def load(self, path: str | os.PathLike):
        self.load_state_dict(torch.load(path))
        return self
