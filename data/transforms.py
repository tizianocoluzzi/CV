"""Mild image transforms for forensic image classification.

The project depends on subtle traces left by generation and post-processing.
For that reason the initial transforms avoid strong color, blur, crop, JPEG,
and rotation augmentations that could destroy the evidence the model should
learn from.
"""

from __future__ import annotations

import random
from dataclasses import dataclass

from PIL import Image
from torchvision.transforms import InterpolationMode
from torchvision.transforms import functional as F


IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]


@dataclass
class RRTransform:
    """Resize, optionally flip, convert to tensor, and normalize.

    The same object can transform a single image or an RGB/residual pair. Pair
    support matters because the residual tensor should remain aligned with the
    RGB tensor when random horizontal flipping is used.
    """

    image_size: int = 224
    train: bool = False
    horizontal_flip_prob: float = 0.5

    def _resize(self, image: Image.Image) -> Image.Image:
        """Use a fixed square size expected by common CNN/Transformer backbones."""
        return F.resize(
            image,
            [self.image_size, self.image_size],
            interpolation=InterpolationMode.BILINEAR,
        )

    def _to_normalized_tensor(self, image: Image.Image):
        """Convert PIL images to tensors with ImageNet normalization."""
        tensor = F.to_tensor(image)
        return F.normalize(tensor, mean=IMAGENET_MEAN, std=IMAGENET_STD)

    def __call__(self, image: Image.Image):
        """Transform one PIL image."""
        image = self._resize(image)

        if self.train and random.random() < self.horizontal_flip_prob:
            image = F.hflip(image)

        return self._to_normalized_tensor(image)

    def apply_pair(self, image: Image.Image, residual: Image.Image):
        """Transform RGB and residual images with shared geometry."""
        image = self._resize(image)
        residual = self._resize(residual)

        if self.train and random.random() < self.horizontal_flip_prob:
            image = F.hflip(image)
            residual = F.hflip(residual)

        return self._to_normalized_tensor(image), self._to_normalized_tensor(residual)


def get_train_transforms(image_size: int = 224):
    """Return mild training transforms that preserve forensic traces."""
    return RRTransform(image_size=image_size, train=True)


def get_eval_transforms(image_size: int = 224):
    """Return deterministic evaluation transforms."""
    return RRTransform(image_size=image_size, train=False)
