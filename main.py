import torch
import torch.utils
import torch.utils.data
from torchvision.transforms import v2

from data import ImagePairsDataset
from model import ImageEmbeding


def main(
    dataset_folder="/home/kasra/datasets/tanks_and_temples/images/",
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
    dataset = ImagePairsDataset(dataset_folder, seed=42, transform=transform)
    data_loader = torch.utils.data.DataLoader(
        dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers
    )
    print("Model created")
    print("Dataset length:", len(dataset))
    print("DataLoader length:", len(data_loader))
    img1, img2, label = next(iter(data_loader))
    assert img1.shape == img2.shape
    input_shape = img1.shape[1:]
    print(f"dataset sample shapes: {img1.shape=}, {img2.shape=}, {label=}")
    print(f"{input_shape=}")
    model = ImageEmbeding(input_shape=input_shape, device=device, **model_kwargs)
    model.fit(dataloader=data_loader, epochs=10, use_wandb=True)
    model.save("model.pth")


if __name__ == "__main__":
    main()
