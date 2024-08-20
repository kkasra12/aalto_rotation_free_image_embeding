import torch
import torch.utils
import torch.utils.data
from torchvision.transforms import v2

from data import ImagePairsDataset
from model import ImageEmbeding


def main(
    train_folder="/home/kasra/datasets/tanks_and_temples/images/train",
    test_folder="/home/kasra/datasets/tanks_and_temples/images/test",
    batch_size=32,
    num_workers=4,
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

    train_dataset = ImagePairsDataset(train_folder, seed=42, transform=transform)
    test_dataset = ImagePairsDataset(test_folder, seed=42, transform=transform)
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
    model.fit(dataloader=data_loader, epochs=10, use_wandb=True)
    model.save("model.pth")


if __name__ == "__main__":
    main()
