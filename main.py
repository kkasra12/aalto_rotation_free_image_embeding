import os
import torch
import torch.utils
import torch.utils.data
from torchvision.transforms import v2

from data import ImagePairsDataset
from model import ImageEmbeding
import argparse
from rich.console import Console
from rich.table import Table


def main(
    train_folder,
    test_folder,
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

    train_dataset = ImagePairsDataset(
        train_folder, seed=42, transform=transform, max_img_per_class=max_img_per_class
    )
    test_dataset = ImagePairsDataset(
        test_folder, seed=42, transform=transform, max_img_per_class=max_img_per_class
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
    model.save("model.pth")


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
    args = parser.parse_args()
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
        max_img_per_class=args.max_img_per_class,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        epochs=args.epochs,
        use_wandb=args.use_wandb,
        transform=None,
    )
