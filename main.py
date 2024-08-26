from ast import parse
import os
import torch
import torch.utils
import torch.utils.data
from torchvision.transforms import v2
import wandb

from data import ImagePairsDataset
from model import ImageEmbeding
import argparse
from rich.console import Console
from rich.table import Table


def create_datasets(
    train_folder: str | os.PathLike,
    test_folder: str | os.PathLike,
    max_img_per_class: int,
    transform=None,
    test_size: float = 0.2,
):
    """
    creates test and train dataset

    Args:
    train_folder: str|os.PathLike: path to the training folder
    test_folder: str|os.PathLike: path to the testing folder
    max_img_per_class: int: max images per class
    transform: torchvision.transforms.Compose: transformation for the images
    test_size: float: portion of the test dataset,
                               if zero, the test dataset will be created from the test folder
                               and the train folder will be used for the train dataset.
                               if more than zero, the test and train folders will be combined
                               and the test_size will be used to split the dataset into test and train datasets.
                               will raise an error if the value
                               is less than zero or more than one.

    """
    if 0 < test_size < 1:
        dataset = ImagePairsDataset(
            root_dirs=[train_folder, test_folder],
            transform=transform,
            max_img_per_class=max_img_per_class,
        )
        train_len = int(len(dataset) * (1 - test_size))
        test_len = len(dataset) - train_len
        train_dataset, test_dataset = torch.utils.data.random_split(
            dataset, [train_len, test_len]
        )
    elif test_size == 0:
        train_dataset = ImagePairsDataset(
            root_dirs=train_folder,
            transform=transform,
            max_img_per_class=max_img_per_class,
        )
        test_dataset = ImagePairsDataset(
            root_dirs=test_folder,
            transform=transform,
            max_img_per_class=max_img_per_class,
        )
    else:
        raise ValueError(
            f"test_size should be between 0 and 1, inclusive, or zero not {test_size}"
        )
    return train_dataset, test_dataset


def main(
    train_folder,
    test_folder,
    test_size,
    max_img_per_class,
    batch_size,
    num_workers,
    epochs,
    use_wandb,
    checkpoint_path=None,
    transform=None,
    model_kwargs={},
    device=None,
):
    if transform is None:
        transform = v2.Compose(
            [v2.Resize((224, 224)), v2.ToDtype(torch.float32, scale=True)]
        )
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if checkpoint_path is not None:
        os.makedirs(checkpoint_path, exist_ok=True)

    # train_dataset = ImagePairsDataset(
    #     train_folder, seed=42, transform=transform, max_img_per_class=max_img_per_class
    # )
    # test_dataset = ImagePairsDataset(
    #     test_folder, seed=42, transform=transform, max_img_per_class=max_img_per_class
    # )
    train_dataset, test_dataset = create_datasets(
        train_folder=train_folder,
        test_folder=test_folder,
        max_img_per_class=max_img_per_class,
        transform=transform,
        test_size=test_size,
    )
    train_dataloader = torch.utils.data.DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers
    )
    test_dataloader = torch.utils.data.DataLoader(
        test_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers
    )

    print("Model created")
    print("train dataset length:", len(train_dataset))
    print("test dataset length:", len(test_dataset))
    print("train dataloader length:", len(train_dataloader))
    print("test dataloader length:", len(test_dataloader))

    if use_wandb:
        wandb.init(
            project="image-embedding",
            config=model_kwargs
            | {
                "max_img_per_class": max_img_per_class,
                "batch_size": batch_size,
                "num_workers": num_workers,
                "epochs": epochs,
                "checkpoint_path": checkpoint_path,
                "device": device,
                "node": os.uname().nodename,
                "test_size": test_size,
            },
        )

    img1, img2, label = next(iter(train_dataloader))
    assert img1.shape == img2.shape
    input_shape = img1.shape[1:]
    print(f"dataset sample shapes: {img1.shape=}, {img2.shape=}, {label=}")
    print(f"{input_shape=}")
    model = ImageEmbeding(input_shape=input_shape, device=device, **model_kwargs)
    model.fit(
        dataloader=train_dataloader,
        test_dataloader=test_dataloader,
        epochs=epochs,
        use_wandb=use_wandb,
        checkpoint_path=checkpoint_path,
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Image Embedding Training")
    parser.add_argument(
        "--train_folder",
        "--train",
        type=str,
        default="/home/kasra/datasets/tanks_and_temples/images/train",
        help="Path to the training folder",
    )
    parser.add_argument(
        "--test_folder",
        "--test",
        type=str,
        default="/home/kasra/datasets/tanks_and_temples/images/test",
        help="Path to the testing folder",
    )
    parser.add_argument(
        "--test_size",
        "-t",
        type=float,
        default=0.2,
        help="Portion of the test dataset, "
        "if zero, the test dataset will be created from the test folder and the train folder will be used for the train dataset. "
        "if more than zero, the test and train folders will be combined "
        "and the test_size will be used to split the dataset into test and train datasets. "
        "will raise an error if the value is less than zero or more than one.",
    )

    parser.add_argument(
        "--max_img_per_class", "-m", type=int, default=20, help="Max images per class"
    )
    parser.add_argument(
        "--batch_size", "-b", type=int, default=32, help="Batch size for training"
    )
    parser.add_argument(
        "--num_workers",
        "-n",
        type=int,
        default=4,
        help="Number of workers for data loading",
    )
    parser.add_argument(
        "--epochs", "-e", type=int, default=10, help="Number of training epochs"
    )
    parser.add_argument(
        "--use_wandb",
        "-w",
        action="store_false",
        help="Whether to use wandb for logging",
    )
    parser.add_argument(
        "--cnn",
        "-c",
        type=str,
        choices=["resnet18", "resnet50", "simple"],
        default="simple",
        help="The CNN part of the model",
    )
    parser.add_argument(
        "--loss_margin",
        "-l",
        type=float,
        default=1.0,
        help="The margin parameter in the loss function",
    )
    parser.add_argument(
        "--distance",
        "-d",
        type=str,
        choices=["cosine", "euclidean"],
        default="cosine",
        help="The distance metric to use",
    )
    parser.add_argument(
        "--json",
        type=str,
        default=None,
        help="Path to the json file containing the model parameters, "
        "if this argument provided, all other arguments will be ignored",
    )

    args = parser.parse_args()
    if args.json is not None:
        import json

        with open(args.json, "r") as f:
            for k, v in json.load(f).items():
                setattr(args, k, v)

    args_table = Table(title="Arguments")
    args_table.add_column("Argument")
    args_table.add_column("Value")
    for i, j in vars(args).items():
        args_table.add_row(i, str(j))
    console = Console()
    console.print(args_table)
    main(
        train_folder=args.train_folder,
        test_folder=args.test_folder,
        test_size=args.test_size,
        max_img_per_class=args.max_img_per_class,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        epochs=args.epochs,
        use_wandb=args.use_wandb,
        transform=None,
        model_kwargs={
            "cnn_model": args.cnn,
            "loss_margin": args.loss_margin,
            "distance": args.distance,
        },
    )
