"""
This module contains the dataset class for the image pairs dataset.

`Image` class:
    Represents an image file in the dataset.
    the image is loaded when the `image` property is accessed (the image will be cached and the `CACHE_SIZE` most recent images will be kept in memory).

`ImagePairsDataset` class:
    A dataset class for the image pairs dataset.
    The dataset is created by providing the root directory of the dataset.
    The dataset will contain all possible pairs of images in the dataset.
    The `__getitem__` method returns a tuple of two images and a label, where the label is 0 if the images are of the same class, 1 otherwise.
"""

from collections.abc import Callable
from functools import lru_cache
from itertools import combinations, count
import os
from pathlib import Path
from random import shuffle
import shutil
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
        transform: Optional[Callable[[torch.Tensor], torch.Tensor]] = None,
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
        transform: Optional[Callable] = None,
        seed: Optional[int] = None,
        max_img_per_class: Optional[float] = None,
    ):
        if isinstance(root_dirs, str):
            root_dirs = [root_dirs]
        self.root_dirs = root_dirs
        assert isinstance(root_dirs, list) and all(isinstance(root_dir, (str, os.PathLike)) for root_dir in root_dirs), "root_dirs should be a list of strings or os.PathLike objects"
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
        assert max_img_per_class is not None, "max_img_per_class should be a positive integer or None"
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

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor, int]:
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


def train_test_split(
    folder: str | os.PathLike, test_size: float, transform=None, dest_folder: Optional[str | os.PathLike] = None
):
    """this will copy random `test_size` portion of the image in folder to dest_folder, 
    we assume the folder format is like this:
     folder
     |
     |- class1
     |- class2
     |- class3
        ...

    and each classs has some images in it, the function will copy random `test_size` portion of the images in each class to dest_folder, and the rest of the images will be kept in same place.
    check the description of the `dest_folder` argument to see if function will copy or move the images.
    Args:
        folder (str | os.PathLike): the folder containing the images, the format of the folder should be like this:
        test_size (float): the portion of the images to be copied to dest_folder, should be between 0 and 1.
        transform (_type_, optional): not implemented yet. Defaults to None.
        dest_folder (Optional[str  |  os.PathLike], optional): if none, the images will be moved to and the folder will change in place, otherwise we will copy them in other folder. Defaults to None.
    """
    if transform:
        raise NotImplementedError("transform argument is not implemented yet")
    if not (0 < test_size < 1):
        raise ValueError(f"test_size should be between 0 and 1, not {test_size}")
    if not os.path.isdir(folder):
        raise FileNotFoundError(f"Directory {folder} not found")
    else:
        folder = Path(folder)
    if dest_folder is not None:
        if not os.path.isdir(dest_folder):
            raise FileExistsError(f"Directory {dest_folder} exists")
        else:
            dest_folder = Path(dest_folder)
            dest_folder.mkdir(parents=True, exist_ok=False)
    else:
        dest_folder = folder

    test_folder = folder / "test"
    train_folder = folder / "train"
    if test_folder.exists() or train_folder.exists():
        raise FileExistsError(f"Directory {test_folder} or {train_folder} exists")
    test_folder.mkdir(parents=True, exist_ok=False)
    train_folder.mkdir(parents=True, exist_ok=False)


    for class_name in folder.iterdir():
        if not class_name.is_dir() or class_name.name in ["test", "train"]:
            continue
        print(f"Processing class {class_name.name}...")
        images = list(class_name.iterdir())
        shuffle(images)
        test_images = images[: int(len(images) * test_size)]

        class_train_folder = train_folder / class_name.name
        class_train_folder.mkdir(parents=True, exist_ok=False)
        class_test_folder = test_folder / class_name.name
        class_test_folder.mkdir(parents=True, exist_ok=False)



        if dest_folder is not None:
            copy_or_move = shutil.copy
        else:
            copy_or_move = shutil.move
                
        for img in test_images:
            copy_or_move(img, class_test_folder / img.name)
        for img in images[int(len(images) * test_size) :]:
            copy_or_move(img, class_train_folder / img.name)

        if dest_folder is None:
            class_name.unlink()

    
 

if __name__ == "__main__":
    from torch.utils.data import DataLoader
    from sys import argv
    data_folder = "/home/kasra/datasets/tanks_and_temples/images" if len(argv) < 2 else argv[1]

    dataset = ImagePairsDataset(
        data_folder,
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
