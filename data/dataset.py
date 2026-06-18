"""PyTorch Dataset for RRDataset split CSV files.

RRDataset reads the metadata rows produced by data/make_splits.py and returns
the two targets needed by the project. It can also return a high-frequency
residual image for the planned trace-aware branch.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
from PIL import Image, ImageFilter
from torch.utils.data import Dataset
from torchvision.transforms import functional as F


REQUIRED_COLUMNS = {
    "image_path",
    "real_fake_label",
    "transform_label",
}


class RRDataset(Dataset):
    """Load RGB images, labels, and optional residuals from a split CSV.

    Each item returns:
    - image: transformed RGB tensor;
    - real_fake_label: 0 for real and 1 for fake;
    - transform_label: 0 original, 1 internet-transmitted, 2 redigitized;
    - image_path: useful for debugging and qualitative inspection.

    When return_residual=True, the sample also includes residual, computed as
    abs(image - blurred(image)). This emphasizes high-frequency traces such as
    compression artifacts, resampling artifacts, and post-processing evidence.
    """

    def __init__(
        self,
        csv_file: str | Path,
        transform: Any | None = None,
        root_dir: str | Path = ".",
        return_residual: bool = False,
        residual_type: str = "gaussian",
        residual_kernel_size: int = 5,
    ) -> None:
        self.csv_file = Path(csv_file)
        self.transform = transform
        self.root_dir = Path(root_dir)
        self.return_residual = return_residual
        self.residual_type = residual_type.lower()
        self.residual_kernel_size = residual_kernel_size

        if not self.csv_file.exists():
            raise FileNotFoundError(f"Split CSV does not exist: {self.csv_file}")

        self.df = pd.read_csv(self.csv_file)
        self._validate_dataframe()
        self._validate_residual_config()

    def _validate_dataframe(self) -> None:
        """Fail early if a split file is not compatible with this Dataset."""
        missing_columns = REQUIRED_COLUMNS.difference(self.df.columns)
        if missing_columns:
            missing = ", ".join(sorted(missing_columns))
            raise ValueError(f"Split CSV is missing required columns: {missing}")

    def _validate_residual_config(self) -> None:
        """Check residual settings before the first image is loaded."""
        if self.residual_type not in {"gaussian", "median"}:
            raise ValueError("residual_type must be either 'gaussian' or 'median'.")

        if self.residual_kernel_size <= 0 or self.residual_kernel_size % 2 == 0:
            raise ValueError("residual_kernel_size must be a positive odd integer.")

    def __len__(self) -> int:
        return len(self.df)

    def _resolve_image_path(self, image_path: str) -> Path:
        """Resolve relative CSV paths from the repository root by default."""
        path = Path(image_path)
        if path.is_absolute():
            return path
        return self.root_dir / path

    def _load_image(self, image_path: str) -> Image.Image:
        """Load every image as RGB so tensors always have three channels."""
        path = self._resolve_image_path(image_path)
        if not path.exists():
            raise FileNotFoundError(f"Image file does not exist: {path}")

        with Image.open(path) as image:
            return image.convert("RGB")

    def _blur_with_cv2(self, array: np.ndarray) -> np.ndarray | None:
        """Use OpenCV for residual blur when it is available."""
        try:
            import cv2
        except ImportError:
            return None

        if self.residual_type == "gaussian":
            return cv2.GaussianBlur(
                array,
                (self.residual_kernel_size, self.residual_kernel_size),
                sigmaX=0,
            )

        return cv2.medianBlur(array, self.residual_kernel_size)

    def _blur_with_pil(self, image: Image.Image) -> np.ndarray:
        """Fallback blur path so residuals still work without OpenCV."""
        if self.residual_type == "gaussian":
            radius = max(self.residual_kernel_size / 6.0, 1.0)
            blurred = image.filter(ImageFilter.GaussianBlur(radius=radius))
        else:
            blurred = image.filter(ImageFilter.MedianFilter(size=self.residual_kernel_size))

        return np.asarray(blurred, dtype=np.float32)

    def _compute_residual(self, image: Image.Image) -> Image.Image:
        """Compute abs(image - blur(image)) as a PIL image.

        The RGB branch captures semantic content. The residual branch is meant
        to expose forensic traces that may help identify generated images or
        post-processing transformations.
        """
        image_array = np.asarray(image, dtype=np.float32)
        blurred = self._blur_with_cv2(image_array)

        if blurred is None:
            blurred = self._blur_with_pil(image)

        residual = np.abs(image_array - blurred)
        residual = np.clip(residual, 0, 255).astype(np.uint8)
        return Image.fromarray(residual, mode="RGB")

    def _apply_transforms(
        self,
        image: Image.Image,
        residual: Image.Image | None = None,
    ):
        """Apply transforms to RGB and residual images.

        RRTransform exposes apply_pair so the random horizontal flip is shared
        by both images. This keeps RGB and residual tensors spatially aligned.
        """
        if self.transform is None:
            image_tensor = F.to_tensor(image)
            residual_tensor = F.to_tensor(residual) if residual is not None else None
            return image_tensor, residual_tensor

        if residual is not None and hasattr(self.transform, "apply_pair"):
            return self.transform.apply_pair(image, residual)

        image_tensor = self.transform(image)
        residual_tensor = self.transform(residual) if residual is not None else None
        return image_tensor, residual_tensor

    def __getitem__(self, index: int) -> dict[str, Any]:
        """Return one training/evaluation sample as a dictionary."""
        row = self.df.iloc[index]
        image_path = str(row["image_path"])

        image = self._load_image(image_path)
        residual = self._compute_residual(image) if self.return_residual else None
        image_tensor, residual_tensor = self._apply_transforms(image, residual)

        sample = {
            "image": image_tensor,
            "real_fake_label": torch.tensor(int(row["real_fake_label"]), dtype=torch.long),
            "transform_label": torch.tensor(int(row["transform_label"]), dtype=torch.long),
            "image_path": image_path,
        }

        if self.return_residual:
            sample["residual"] = residual_tensor

        return sample
