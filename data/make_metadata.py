"""Build the image-level metadata table for RRDataset_final.

Run from the repository root:
    python data/make_metadata.py

The output CSV is the single source of truth for later split generation and
PyTorch loading. It records both tasks used by the project: real/fake
classification and transformation classification.
"""

from pathlib import Path
import pandas as pd
from PIL import Image
from tqdm import tqdm


IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tif", ".tiff"}


# Folder names come from the dataset, while label names match the project text.
# The folder "ai" is stored as the student-facing class name "fake".
REAL_FAKE_LABELS = {
    "real": 0,
    "ai": 1,
}


# Transformation folders are mapped to the exact class names used downstream.
# In particular, "transfer" means internet-transmitted and "redigital" means
# re-digitized.
TRANSFORM_LABELS = {
    "original": 0,
    "transfer": 1,
    "redigital": 2,
}


TRANSFORM_NAMES = {
    "original": "original",
    "transfer": "internet_transmitted",
    "redigital": "redigitized",
}


def is_image_file(path: Path) -> bool:
    """Return True for supported image extensions only."""
    return path.suffix.lower() in IMAGE_EXTENSIONS


def verify_image(path: Path) -> bool:
    """Use PIL to catch corrupted files before they enter metadata.csv."""
    try:
        with Image.open(path) as img:
            img.verify()
        return True
    except Exception:
        return False


def parse_labels_from_path(path: Path, dataset_root: Path):
    """Read task labels from the fixed RRDataset_final folder layout.

    Expected structure:

    RRDataset_final/
    ├── original/
    │   ├── ai/
    │   └── real/
    ├── redigital/
    │   ├── ai/
    │   └── real/
    └── transfer/
        ├── ai/
        └── real/
    """

    relative_parts = path.relative_to(dataset_root).parts

    if len(relative_parts) < 3:
        raise ValueError(f"Unexpected path structure: {path}")

    transform_folder = relative_parts[0].lower()
    real_fake_folder = relative_parts[1].lower()

    if transform_folder not in TRANSFORM_LABELS:
        raise ValueError(f"Unknown transformation folder '{transform_folder}' in path: {path}")

    if real_fake_folder not in REAL_FAKE_LABELS:
        raise ValueError(f"Unknown real/fake folder '{real_fake_folder}' in path: {path}")

    transform_label = TRANSFORM_LABELS[transform_folder]
    transform_name = TRANSFORM_NAMES[transform_folder]

    real_fake_label = REAL_FAKE_LABELS[real_fake_folder]
    real_fake_name = "fake" if real_fake_folder == "ai" else "real"

    return real_fake_label, real_fake_name, transform_label, transform_name


def build_metadata(dataset_root: str, output_csv: str, verify_images: bool = True):
    """Scan the raw dataset and write one row per valid image.

    The resulting CSV has:
    - image_path: path used by RRDataset to open the image later;
    - real_fake_label/name: binary target, 0=real and 1=fake;
    - transform_label/name: transformation target, 0=original,
      1=internet_transmitted, and 2=redigitized.
    """
    dataset_root = Path(dataset_root)

    if not dataset_root.exists():
        raise FileNotFoundError(f"Dataset root does not exist: {dataset_root}")

    image_paths = sorted([p for p in dataset_root.rglob("*") if p.is_file() and is_image_file(p)])

    rows = []
    skipped = 0

    print(f"Found {len(image_paths)} image files.")

    for path in tqdm(image_paths):
        # Verification is optional because it is useful but can be slow on a
        # full dataset. Corrupted images are skipped so loaders do not fail
        # halfway through training.
        if verify_images and not verify_image(path):
            skipped += 1
            print(f"Skipping corrupted image: {path}")
            continue

        real_fake_label, real_fake_name, transform_label, transform_name = parse_labels_from_path(
            path=path,
            dataset_root=dataset_root,
        )

        rows.append({
            "image_path": str(path),
            "real_fake_label": real_fake_label,
            "real_fake_name": real_fake_name,
            "transform_label": transform_label,
            "transform_name": transform_name,
        })

    df = pd.DataFrame(rows)

    output_csv = Path(output_csv)
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_csv, index=False)

    print(f"\nSaved metadata to: {output_csv}")
    print(f"Valid images: {len(df)}")
    print(f"Skipped images: {skipped}")

    # These summaries make label-mapping mistakes obvious before training.
    print("\nReal/Fake distribution:")
    print(df["real_fake_name"].value_counts())

    print("\nTransformation distribution:")
    print(df["transform_name"].value_counts())

    print("\nJoint distribution:")
    print(pd.crosstab(df["real_fake_name"], df["transform_name"]))

    return df


if __name__ == "__main__":
    build_metadata(
        dataset_root="data/raw/RRDataset_final",
        output_csv="data/processed/metadata.csv",
        verify_images=True,
    )
