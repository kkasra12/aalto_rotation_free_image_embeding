"""
This module contains the dataset class for the image pairs dataset.

`Image` class:
    Represents an image file in the dataset.
    the image is loaded when the `image` property is accessed (the image will be cached and the `CACHE_SIZE` most recent images will be kept in memory).

`ImagePairsDataset` class:
    A dataset class for the image pairs dataset.
    The dataset is created by providing the root directory of the dataset.
    The dataset will contain all possible pairs of images in the dataset.
    The `__getitem__` method returns a tuple of two images and a label, where the label is 1 if the images are of the same class, 0 otherwise.
"""

from functools import lru_cache
from itertools import combinations, count
import os
from typing import Optional
import numpy as np
import torch
from torch.utils.data import Dataset
from torchvision.transforms import v2
from torchvision.io import read_image

CACHE_SIZE = 100


class Image:
    def __init__(
        self,
        root_dir: str | os.PathLike,
        class_name: str,
        file_name: str,
        transform: callable = None,
    ):
        # if class_name not in os.listdir(root_dir):
        #     raise FileNotFoundError(f"Class {class_name} not found in {root_dir}")
        # if file_name not in os.listdir(os.path.join(root_dir, class_name)):
        #     raise FileNotFoundError(f"File {file_name} not found in {root_dir}/{class_name}")

        self.class_name = class_name
        self.file_name = file_name
        self.root_dir = root_dir
        self.transform = transform

    @property
    @lru_cache(maxsize=CACHE_SIZE)
    def image(self):
        img = read_image(os.path.join(self.root_dir, self.class_name, self.file_name))
        if self.transform:
            img = self.transform(img)
            # print("loaded img size:", img.shape, img.dtype, type(img))
        return img


class ImagePairsDataset(Dataset):
    files: list[Image]

    def __init__(
        self,
        root_dirs: list[str | os.PathLike] | str | os.PathLike,
        transform=None,
        seed: int = None,
        max_img_per_class: Optional[int] = None,
    ):
        if isinstance(root_dirs, str):
            root_dirs = [root_dirs]
        self.root_dirs = root_dirs
        if seed:
            torch.manual_seed(seed)
        # if any(not os.path.isdir(root_dir) for root_dir in root_dirs):
        #     raise FileNotFoundError(f"One of the directories {root_dirs} not found")
        # classes = {}
        # for root_dir in root_dirs:
        #     if not os.path.isdir(root_dir):
        #         raise FileNotFoundError(f"Directory {root_dir} not found")
        #     classes[root_dir] = os.listdir(root_dir)
        if max_img_per_class is None:
            max_img_per_class = float("inf")
        # if max_img_per_class is not None:
        #     self.files = [
        #         Image(root_dir, class_name, file_name, transform)
        #         for root_dir, class_name in classes.items()
        #         for _, file_name in zip(
        #             range(max_img_per_class),
        #             os.listdir(os.path.join(root_dir, class_name)),
        #         )
        #     ]
        # else:
        self.files = [
            Image(root_dir, class_name, file_name, transform)
            for root_dir in root_dirs
            for class_name in os.listdir(root_dir)
            for i, file_name in zip(
                count(), os.listdir(os.path.join(root_dir, class_name))
            )
            if i < max_img_per_class
        ]
        self.pairse = list(combinations(self.files, 2))

    def __len__(self):
        return len(self.pairse)

    def __getitem__(self, idx: int) -> tuple[tuple[np.ndarray, np.ndarray], int]:
        """
        Returns a tuple of two images and a label,
        where the label is 0 if the images are from same class, 1 otherwise.
        Args:
            idx (int): Index of the pair

        Returns:
            tuple: (image1, image2), label
        """
        img1, img2 = self.pairse[idx]
        return img1.image, img2.image, int(img1.class_name != img2.class_name)


if __name__ == "__main__":
    from torch.utils.data import DataLoader

    dataset = ImagePairsDataset(
        "/home/kasra/datasets/tanks_and_temples/images",
        transform=v2.Compose(
            [
                # v2.ToDtype(torch.float32, scale=True),
                v2.Resize((128, 128)),
                v2.ConvertImageDtype(torch.float32),
                v2.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ]
        ),
    )
    dataloader = DataLoader(dataset, batch_size=10, shuffle=True)
    for i, data in zip(range(5), dataloader):
        print(f"""types: {[type(i) for i in data]}
        shapes: {[i.shape for i in data]}
        labels: {data[2]}""")
