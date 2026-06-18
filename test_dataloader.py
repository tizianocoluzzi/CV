"""Smoke test for the RRDataset DataLoader.

Run from the repository root after metadata and splits exist:
    python test_dataloader.py

This script intentionally fetches only one batch. It verifies that RGB images,
high-frequency residuals, and both labels are collated correctly before any
model or training code is introduced.
"""

from data.dataset import RRDataset
from data.transforms import get_train_transforms
from torch.utils.data import DataLoader


def main() -> None:
    """Load the train split and print the structure of one mini-batch."""
    dataset = RRDataset(
        csv_file="data/splits/train.csv",
        transform=get_train_transforms(image_size=224),
        return_residual=True,
        residual_type="gaussian",
    )

    dataloader = DataLoader(
        dataset,
        batch_size=8,
        shuffle=True,
        num_workers=0,
    )

    batch = next(iter(dataloader))

    print(f"image: {batch['image'].shape}")
    print(f"residual: {batch['residual'].shape}")
    print(f"real/fake labels: {batch['real_fake_label']}")
    print(f"transformation labels: {batch['transform_label']}")
    print(f"first image path: {batch['image_path'][0]}")


if __name__ == "__main__":
    main()
