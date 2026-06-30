# == IMPORTS ==
from __future__ import annotations

import multiprocessing as mp
import os
import random
import tempfile
from dataclasses import dataclass
from pathlib import Path

_CACHE_ROOT = Path(tempfile.gettempdir()) / "cv_final_pipeline_cache"
(_CACHE_ROOT / "matplotlib").mkdir(parents=True, exist_ok=True)
(_CACHE_ROOT / "xdg").mkdir(parents=True, exist_ok=True)
os.environ.setdefault("MPLCONFIGDIR", str(_CACHE_ROOT / "matplotlib"))
os.environ.setdefault("XDG_CACHE_HOME", str(_CACHE_ROOT / "xdg"))

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image, ImageFile, ImageFilter
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    roc_auc_score,
)
from sklearn.model_selection import GroupShuffleSplit, train_test_split
from torch.utils.data import DataLoader, Dataset
from tqdm.auto import tqdm
from transformers import AutoConfig, AutoImageProcessor, AutoModel, CLIPImageProcessor, CLIPModel


# == GLOBALS ==

DATA_ROOT = Path("data/raw/RRDataset_final")
RESULTS_DIR = Path("results/final_pipeline")
CHECKPOINT_DIR = Path("checkpoints/final_pipeline")

SUBFOLDERS = ["original", "redigital", "transfer"]
CLASS_NAMES = ["ai", "real"]
TRANSFORM_NAMES = SUBFOLDERS

CLASS_TO_IDX = {"ai": 0, "real": 1}
TRANSFORM_TO_IDX = {"original": 0, "redigital": 1, "transfer": 2}
IDX_TO_CLASS = {idx: name for name, idx in CLASS_TO_IDX.items()}
IDX_TO_TRANSFORM = {idx: name for name, idx in TRANSFORM_TO_IDX.items()}

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}

SEED = 42
TEST_SIZE = 0.2
VAL_SIZE = 0.1

SELECTED_BACKBONE = "swin-tiny"
MODEL_CONFIGS = {
    "deit-tiny": {
        "type": "generic",
        "name": "facebook/deit-tiny-patch16-224",
    },
    "deit-small": {
        "type": "generic",
        "name": "facebook/deit-small-patch16-224",
    },
    "swin-tiny": {
        "type": "generic",
        "name": "microsoft/swin-tiny-patch4-window7-224",
    },
    "clip-vit-b": {
        "type": "clip",
        "name": "openai/clip-vit-base-patch32",
    },
}

# Keep this True only for the requested technical smoke test.
# Set FAST_DEV_RUN = False for the real project run.
FAST_DEV_RUN = False

SAMPLES_PER_GROUP = 1000
NUM_EPOCHS = 15
PATIENCE = 4
BATCH_SIZE = 32

FAST_SAMPLES_PER_GROUP = 5
FAST_NUM_EPOCHS = 1
FAST_PATIENCE = 1
FAST_BATCH_SIZE = 4

NUM_WORKERS = min(2, max(1, (os.cpu_count() or 2) // 2))
LEARNING_RATE = 1e-4
WEIGHT_DECAY = 1e-4
MIN_DELTA = 1e-4

# Multi-task loss:
#     total_loss = LAMBDA * binary_loss + (1 - LAMBDA) * transform_loss
# The script evaluates a small sweep of lambda values to compare task weighting.
LAMBDA_VALUES = [0.2, 0.35, 0.65, 0.8]
LAMBDA = 0.5
USE_MULTITASK_TRANSFORM_FOCAL_LOSS = True
MULTITASK_TRANSFORM_FOCAL_GAMMA = 2.0

FREEZE_BACKBONE = True
# Number of final Swin encoder stages to train when FREEZE_BACKBONE is True.
# 0 keeps the previous behavior: the full backbone stays frozen.
SWIN_TRAINABLE_STAGES = 1

USE_RESIDUAL_STATS = True
RESIDUAL_BLUR_RADIUS = 1.0
RESIDUAL_STATS_IMAGE_SIZE = 224
RESIDUAL_HIST_BINS = 8
RESIDUAL_STATS_DIM = 3 * 4 + 2 + 1 + RESIDUAL_HIST_BINS
RESIDUAL_STATS_CSV_PATH = (
    Path("data")
    / f"residual_stats_size{RESIDUAL_STATS_IMAGE_SIZE}_blur{RESIDUAL_BLUR_RADIUS}_bins{RESIDUAL_HIST_BINS}.csv"
)

ImageFile.LOAD_TRUNCATED_IMAGES = True


# == UTILS ==

@dataclass
class RunSettings:
    samples_per_group: int | None
    num_epochs: int
    patience: int
    batch_size: int


@dataclass
class TrainResult:
    model_name: str
    task_type: str
    best_val_score: float
    checkpoint_path: Path
    epochs_ran: int


def get_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


DEVICE = get_device()


def get_run_settings() -> RunSettings:
    if FAST_DEV_RUN:
        return RunSettings(
            samples_per_group=FAST_SAMPLES_PER_GROUP,
            num_epochs=FAST_NUM_EPOCHS,
            patience=FAST_PATIENCE,
            batch_size=FAST_BATCH_SIZE,
        )
    return RunSettings(
        samples_per_group=SAMPLES_PER_GROUP,
        num_epochs=NUM_EPOCHS,
        patience=PATIENCE,
        batch_size=BATCH_SIZE,
    )


def set_seed(seed: int = SEED) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def is_image_file(path: Path) -> bool:
    return path.suffix.lower() in IMAGE_EXTENSIONS


def clear_memory() -> None:
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    if DEVICE.type == "mps":
        try:
            torch.mps.empty_cache()
        except Exception:
            pass


def get_selected_model_config() -> dict:
    if SELECTED_BACKBONE not in MODEL_CONFIGS:
        supported = ", ".join(MODEL_CONFIGS)
        raise ValueError(f"Unknown SELECTED_BACKBONE={SELECTED_BACKBONE!r}. Supported: {supported}")
    return MODEL_CONFIGS[SELECTED_BACKBONE]


def validate_lambda(value: float) -> None:
    if not 0.0 <= value <= 1.0:
        raise ValueError(f"LAMBDA must be in [0, 1], got {value}.")


def get_image_processor(cfg: dict):
    if cfg["type"] == "clip":
        return CLIPImageProcessor.from_pretrained(cfg["name"])
    return AutoImageProcessor.from_pretrained(cfg["name"])


def count_parameters(model: nn.Module) -> tuple[int, int]:
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return total, trainable


def make_cpu_state_dict(model: nn.Module) -> dict[str, torch.Tensor]:
    return {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}


def validate_batch_size(total: int) -> None:
    if total <= 0:
        raise ValueError("Empty dataset split encountered.")


def active_residual_stats_dim() -> int:
    return RESIDUAL_STATS_DIM if USE_RESIDUAL_STATS else 0


def make_run_name(base_name: str) -> str:
    return f"resstats_{base_name}" if USE_RESIDUAL_STATS else base_name


def unpack_batch(
    batch: tuple[torch.Tensor, ...],
) -> tuple[torch.Tensor, torch.Tensor | None, torch.Tensor, torch.Tensor]:
    if len(batch) == 4:
        pixel_values, residual_stats, binary_labels, transform_labels = batch
        return pixel_values, residual_stats, binary_labels, transform_labels
    if len(batch) == 3:
        pixel_values, binary_labels, transform_labels = batch
        return pixel_values, None, binary_labels, transform_labels
    raise ValueError(f"Expected a batch with 3 or 4 elements, got {len(batch)}.")


def compute_residual_stats(image: Image.Image) -> torch.Tensor:
    stats_image = image.convert("RGB").resize(
        (RESIDUAL_STATS_IMAGE_SIZE, RESIDUAL_STATS_IMAGE_SIZE),
        Image.Resampling.BILINEAR,
    )
    blurred = stats_image.filter(ImageFilter.GaussianBlur(radius=RESIDUAL_BLUR_RADIUS))

    image_array = np.asarray(stats_image, dtype=np.float32) / 255.0
    blurred_array = np.asarray(blurred, dtype=np.float32) / 255.0
    image_tensor = torch.from_numpy(image_array)
    blurred_tensor = torch.from_numpy(blurred_array)
    residual = (image_tensor - blurred_tensor).abs()

    flat_residual = residual.reshape(-1, 3)
    channel_mean = flat_residual.mean(dim=0)
    channel_std = flat_residual.std(dim=0, unbiased=False)
    channel_max = flat_residual.max(dim=0).values
    channel_p95 = torch.quantile(flat_residual, 0.95, dim=0)

    residual_values = residual.reshape(-1)
    global_mean = residual_values.mean().view(1)
    global_std = residual_values.std(unbiased=False).view(1)
    hist = torch.histc(residual_values, bins=RESIDUAL_HIST_BINS, min=0.0, max=1.0)
    hist = hist / hist.sum().clamp_min(1.0)

    gray = image_tensor.mean(dim=2)
    if gray.shape[0] > 2 and gray.shape[1] > 2:
        laplacian = (
            -4.0 * gray[1:-1, 1:-1]
            + gray[:-2, 1:-1]
            + gray[2:, 1:-1]
            + gray[1:-1, :-2]
            + gray[1:-1, 2:]
        )
        laplacian_var = laplacian.var(unbiased=False).view(1)
    else:
        laplacian_var = torch.zeros(1, dtype=torch.float32)

    stats = torch.cat(
        [
            channel_mean,
            channel_std,
            channel_max,
            channel_p95,
            global_mean,
            global_std,
            laplacian_var,
            hist,
        ]
    ).to(dtype=torch.float32)

    if stats.numel() != RESIDUAL_STATS_DIM:
        raise RuntimeError(
            f"Residual stats dimension mismatch: expected {RESIDUAL_STATS_DIM}, got {stats.numel()}."
        )
    if not torch.isfinite(stats).all():
        raise RuntimeError("Residual stats contain NaN or Inf values.")
    return stats


def residual_stats_feature_columns() -> list[str]:
    return [f"res_stat_{idx:02d}" for idx in range(RESIDUAL_STATS_DIM)]


def _residual_stats_dataframe_to_cache(
    cached_df: pd.DataFrame,
    feature_columns: list[str],
) -> dict[str, torch.Tensor]:
    values = cached_df[feature_columns].to_numpy(dtype=np.float32)
    return {
        str(path): torch.tensor(row, dtype=torch.float32)
        for path, row in zip(cached_df["path"].astype(str), values)
    }


def _validate_residual_stats_csv(
    cached_df: pd.DataFrame,
    required_paths: set[str],
    feature_columns: list[str],
) -> str | None:
    required_columns = ["path", *feature_columns]
    missing_columns = [column for column in required_columns if column not in cached_df.columns]
    if missing_columns:
        return f"missing columns: {missing_columns}"

    actual_feature_columns = [column for column in cached_df.columns if column.startswith("res_stat_")]
    if len(actual_feature_columns) != RESIDUAL_STATS_DIM:
        return (
            f"expected {RESIDUAL_STATS_DIM} residual-stat columns, "
            f"found {len(actual_feature_columns)}"
        )

    cached_paths = set(cached_df["path"].astype(str))
    missing_paths = required_paths - cached_paths
    if missing_paths:
        return f"missing {len(missing_paths)} required image paths"

    feature_values = cached_df[feature_columns].to_numpy(dtype=np.float32)
    if not np.isfinite(feature_values).all():
        return "non-finite residual-stat values"

    return None


def build_or_load_residual_stats_csv(
    df: pd.DataFrame,
    csv_path: Path,
) -> dict[str, torch.Tensor]:
    feature_columns = residual_stats_feature_columns()
    required_paths = sorted(set(df["path"].astype(str)))

    print("\n== Residual Statistics Cache ==")
    print(f"CSV path: {csv_path}")
    print(f"Required images: {len(required_paths)}")
    print(f"Residual stats dimension: {RESIDUAL_STATS_DIM}")

    if csv_path.exists():
        try:
            cached_df = pd.read_csv(csv_path)
            invalid_reason = _validate_residual_stats_csv(
                cached_df,
                set(required_paths),
                feature_columns,
            )
            if invalid_reason is None:
                valid_df = cached_df[cached_df["path"].astype(str).isin(required_paths)].copy()
                print(
                    "Loaded residual statistics CSV; recomputation skipped. "
                    f"Cached images available for this run: {len(valid_df)}"
                )
                return _residual_stats_dataframe_to_cache(valid_df, feature_columns)
            print(f"[warning] Residual statistics CSV is invalid: {invalid_reason}. Recomputing.")
        except Exception as error:
            print(f"[warning] Could not load residual statistics CSV: {error}. Recomputing.")
    else:
        print("Residual statistics CSV not found. Computing residual statistics once.")

    rows = []
    for image_path in tqdm(required_paths, desc="Computing residual stats", leave=False):
        with Image.open(image_path) as image_file:
            image = image_file.convert("RGB")
        stats = compute_residual_stats(image)
        row = {"path": image_path}
        row.update({column: float(value) for column, value in zip(feature_columns, stats.tolist())})
        rows.append(row)

    residual_df = pd.DataFrame(rows, columns=["path", *feature_columns])
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    residual_df.to_csv(csv_path, index=False)
    print(f"Created residual statistics CSV with {len(residual_df)} images: {csv_path}")

    return _residual_stats_dataframe_to_cache(residual_df, feature_columns)


def combine_with_residual_stats(
    features: torch.Tensor,
    residual_stats: torch.Tensor | None,
    use_residual_stats: bool,
    residual_stats_dim: int,
) -> torch.Tensor:
    if not use_residual_stats:
        return features
    if residual_stats is None:
        raise ValueError("USE_RESIDUAL_STATS=True, but residual_stats were not provided to the model.")
    if residual_stats.ndim == 1:
        residual_stats = residual_stats.unsqueeze(0)
    if residual_stats.shape[0] != features.shape[0]:
        raise ValueError(
            f"Residual stats batch size {residual_stats.shape[0]} does not match "
            f"feature batch size {features.shape[0]}."
        )
    if residual_stats.shape[1] != residual_stats_dim:
        raise ValueError(
            f"Expected residual stats dim {residual_stats_dim}, got {residual_stats.shape[1]}."
        )

    residual_stats = residual_stats.to(device=features.device, dtype=features.dtype)
    return torch.cat([features, residual_stats], dim=1)


# == DATA ==

def build_image_index(data_root: Path = DATA_ROOT) -> pd.DataFrame:
    records = []

    for subfolder in SUBFOLDERS:
        for class_name in CLASS_NAMES:
            class_dir = data_root / subfolder / class_name
            if not class_dir.is_dir():
                print(f"[warning] missing folder: {class_dir}")
                continue

            for img_path in class_dir.rglob("*"):
                if img_path.is_file() and is_image_file(img_path):
                    records.append(
                        {
                            "path": str(img_path.resolve()),
                            "label": class_name,
                            "subfolder": subfolder,
                            "binary_label": CLASS_TO_IDX[class_name],
                            "transform_label": TRANSFORM_TO_IDX[subfolder],
                            "stratify_key": f"{class_name}_{subfolder}",
                            "image_id": img_path.stem,
                        }
                    )

    df = pd.DataFrame(records)
    if df.empty:
        raise RuntimeError(f"No images found under {data_root}. Check the dataset path/structure.")
    return df


def print_dataset_audit(df: pd.DataFrame, title: str) -> None:
    print(f"\n== {title} ==")
    print(f"Total images: {len(df)}")
    print("\nBy binary label:")
    print(df["label"].value_counts().reindex(CLASS_NAMES, fill_value=0))
    print("\nBy transformation:")
    print(df["subfolder"].value_counts().reindex(TRANSFORM_NAMES, fill_value=0))
    print("\nBy label x transformation:")
    joint = (
        df.groupby(["label", "subfolder"])
        .size()
        .reindex(pd.MultiIndex.from_product([CLASS_NAMES, TRANSFORM_NAMES]), fill_value=0)
    )
    print(joint)


def sample_balanced_joint_subset(
    df: pd.DataFrame,
    samples_per_group: int | None,
    seed: int = SEED,
) -> pd.DataFrame:
    group_counts = df.groupby(["label", "subfolder"]).size()
    expected_index = pd.MultiIndex.from_product([CLASS_NAMES, TRANSFORM_NAMES])
    group_counts = group_counts.reindex(expected_index, fill_value=0)

    if (group_counts == 0).any():
        missing = [f"{label}-{subfolder}" for (label, subfolder), count in group_counts.items() if count == 0]
        raise RuntimeError(f"Missing images for required joint groups: {missing}")

    min_available = int(group_counts.min())
    if samples_per_group is None:
        n_take = min_available
    else:
        n_take = min(samples_per_group, min_available)
        if samples_per_group > min_available:
            print(
                f"[warning] requested {samples_per_group} samples per joint group, "
                f"but the smallest group has only {min_available}. Using {n_take}."
            )

    subsets = []
    for class_name in CLASS_NAMES:
        for subfolder in SUBFOLDERS:
            group_df = df[(df["label"] == class_name) & (df["subfolder"] == subfolder)]
            subsets.append(group_df.sample(n=n_take, random_state=seed))

    subset_df = pd.concat(subsets, ignore_index=True)
    return subset_df.sample(frac=1.0, random_state=seed).reset_index(drop=True)


def safe_stratified_split(
    df: pd.DataFrame,
    test_size: float,
    seed: int,
    split_name: str,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    n_rows = len(df)
    n_strata = df["stratify_key"].nunique()
    requested_test_count = int(np.ceil(n_rows * test_size))
    effective_test_size: float | int = test_size

    if requested_test_count < n_strata:
        effective_test_size = n_strata
        print(
            f"[warning] {split_name}: requested test/val count {requested_test_count} is smaller "
            f"than the {n_strata} stratification groups. Using {effective_test_size} rows instead."
        )

    effective_count = effective_test_size if isinstance(effective_test_size, int) else requested_test_count
    if n_rows - effective_count < n_strata:
        raise ValueError(
            f"{split_name}: not enough rows ({n_rows}) to preserve all {n_strata} strata "
            f"in both sides of the split."
        )

    left_df, right_df = train_test_split(
        df,
        test_size=effective_test_size,
        stratify=df["stratify_key"],
        random_state=seed,
    )
    return left_df.reset_index(drop=True), right_df.reset_index(drop=True)


def split_dataframe(df: pd.DataFrame, seed: int = SEED) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    repeated_ids = df["image_id"].duplicated().any()
    val_fraction_of_train_val = VAL_SIZE / (1.0 - TEST_SIZE)

    if repeated_ids:
        print("\nUsing GroupShuffleSplit by image_id to reduce leakage between transformed versions.")
        splitter = GroupShuffleSplit(n_splits=1, test_size=TEST_SIZE, random_state=seed)
        train_val_idx, test_idx = next(splitter.split(df, groups=df["image_id"]))
        train_val_df = df.iloc[train_val_idx].reset_index(drop=True)
        test_df = df.iloc[test_idx].reset_index(drop=True)

        splitter_val = GroupShuffleSplit(n_splits=1, test_size=val_fraction_of_train_val, random_state=seed)
        train_idx, val_idx = next(splitter_val.split(train_val_df, groups=train_val_df["image_id"]))
        train_df = train_val_df.iloc[train_idx].reset_index(drop=True)
        val_df = train_val_df.iloc[val_idx].reset_index(drop=True)

        print("[note] GroupShuffleSplit protects repeated image_id groups but does not guarantee stratification.")
    else:
        print("\nUsing stratified split by label x transformation.")
        train_val_df, test_df = safe_stratified_split(df, TEST_SIZE, seed, "test split")
        train_df, val_df = safe_stratified_split(
            train_val_df,
            val_fraction_of_train_val,
            seed,
            "validation split",
        )

    validate_batch_size(len(train_df))
    validate_batch_size(len(val_df))
    validate_batch_size(len(test_df))
    return train_df, val_df, test_df


class AIRealTransformDataset(Dataset):
    def __init__(
        self,
        dataframe: pd.DataFrame,
        image_processor,
        residual_stats_cache: dict[str, torch.Tensor] | None = None,
    ):
        self.df = dataframe.reset_index(drop=True)
        self.image_processor = image_processor
        self.residual_stats_cache = residual_stats_cache

    def __len__(self) -> int:
        return len(self.df)

    def __getitem__(self, idx: int):
        row = self.df.iloc[idx]
        image = Image.open(row["path"]).convert("RGB")
        encoded = self.image_processor(images=image, return_tensors="pt")
        pixel_values = encoded["pixel_values"][0]
        binary_label = int(row["binary_label"])
        transform_label = int(row["transform_label"])
        if USE_RESIDUAL_STATS:
            image_path = str(row["path"])
            if self.residual_stats_cache is not None:
                if image_path not in self.residual_stats_cache:
                    raise KeyError(f"Missing residual statistics for image path: {image_path}")
                residual_stats = self.residual_stats_cache[image_path].to(dtype=torch.float32)
            else:
                residual_stats = compute_residual_stats(image)
            return pixel_values, residual_stats, binary_label, transform_label
        return pixel_values, binary_label, transform_label


def make_loaders(
    train_df,
    val_df,
    test_df,
    image_processor,
    batch_size: int,
    residual_stats_cache: dict[str, torch.Tensor] | None = None,
):
    train_dataset = AIRealTransformDataset(train_df, image_processor, residual_stats_cache)
    val_dataset = AIRealTransformDataset(val_df, image_processor, residual_stats_cache)
    test_dataset = AIRealTransformDataset(test_df, image_processor, residual_stats_cache)

    pin_memory = DEVICE.type == "cuda"
    persistent_workers = NUM_WORKERS > 0

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=NUM_WORKERS,
        pin_memory=pin_memory,
        persistent_workers=persistent_workers,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=NUM_WORKERS,
        pin_memory=pin_memory,
        persistent_workers=persistent_workers,
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=NUM_WORKERS,
        pin_memory=pin_memory,
        persistent_workers=persistent_workers,
    )

    sample = train_dataset[0]
    sample_pixels, sample_residual_stats, sample_binary, sample_transform = unpack_batch(sample)
    sample_parts = [f"Sample tensor shape: {tuple(sample_pixels.shape)}"]
    if USE_RESIDUAL_STATS:
        if sample_residual_stats is None:
            raise RuntimeError("USE_RESIDUAL_STATS=True but the dataset did not return residual stats.")
        sample_parts.append(f"residual_stats_shape={tuple(sample_residual_stats.shape)}")
    sample_parts.extend(
        [
            f"binary={sample_binary} ({IDX_TO_CLASS[sample_binary]})",
            f"transform={sample_transform} ({IDX_TO_TRANSFORM[sample_transform]})",
        ]
    )
    print(", ".join(sample_parts))
    return train_loader, val_loader, test_loader


# == NETWORK ==

class TransformerFeatureExtractor(nn.Module):
    def __init__(
        self,
        cfg: dict,
        freeze_backbone: bool | None = None,
        swin_trainable_stages: int | None = None,
    ):
        super().__init__()
        if freeze_backbone is None:
            freeze_backbone = FREEZE_BACKBONE
        if swin_trainable_stages is None:
            swin_trainable_stages = SWIN_TRAINABLE_STAGES
        self.cfg = cfg
        self.backbone_type = cfg["type"]
        self.freeze_backbone = freeze_backbone
        self.swin_trainable_stages = swin_trainable_stages
        hf_name = cfg["name"]

        config = AutoConfig.from_pretrained(hf_name)
        print(f"Loading backbone: {hf_name} | HF model_type: {config.model_type}")

        if self.backbone_type == "clip":
            clip_model = CLIPModel.from_pretrained(hf_name)
            self.backbone = clip_model.vision_model
            self.visual_projection = clip_model.visual_projection
            self.feature_dim = clip_model.config.projection_dim
        else:
            self.backbone = AutoModel.from_pretrained(hf_name)
            self.visual_projection = None
            self.feature_dim = self._infer_embed_dim(self.backbone)

        if freeze_backbone:
            for parameter in self.backbone.parameters():
                parameter.requires_grad = False
            if self.visual_projection is not None:
                for parameter in self.visual_projection.parameters():
                    parameter.requires_grad = False
            self._unfreeze_swin_stages(swin_trainable_stages)
            self.backbone_fully_frozen = not any(
                parameter.requires_grad for parameter in self.backbone.parameters()
            )
            self.backbone.eval()
            if self.visual_projection is not None:
                self.visual_projection.eval()
        else:
            self.backbone_fully_frozen = False

    @staticmethod
    def _infer_embed_dim(backbone: nn.Module) -> int:
        cfg = backbone.config
        for attr in ("hidden_size", "embed_dim", "projection_dim"):
            if hasattr(cfg, attr):
                return int(getattr(cfg, attr))
        raise ValueError("Could not infer backbone embedding dimension from config.")

    def _unfreeze_swin_stages(self, swin_trainable_stages: int) -> None:
        if swin_trainable_stages <= 0:
            return
        if getattr(self.backbone.config, "model_type", None) != "swin":
            return

        stages = getattr(getattr(self.backbone, "encoder", None), "layers", None)
        if stages is None:
            raise ValueError("Could not find Swin encoder stages to unfreeze.")
        if swin_trainable_stages > len(stages):
            raise ValueError(
                f"SWIN_TRAINABLE_STAGES={swin_trainable_stages} exceeds "
                f"available Swin stages ({len(stages)})."
            )

        for stage in stages[-swin_trainable_stages:]:
            for parameter in stage.parameters():
                parameter.requires_grad = True
        for module_name in ("layernorm", "pooler"):
            module = getattr(self.backbone, module_name, None)
            if module is not None:
                for parameter in module.parameters():
                    parameter.requires_grad = True

    def _set_swin_trainable_stages_mode(self, mode: bool) -> None:
        if self.swin_trainable_stages <= 0:
            return
        if getattr(self.backbone.config, "model_type", None) != "swin":
            return

        stages = getattr(getattr(self.backbone, "encoder", None), "layers", None)
        for stage in stages[-self.swin_trainable_stages:]:
            stage.train(mode)
        for module_name in ("layernorm", "pooler"):
            module = getattr(self.backbone, module_name, None)
            if module is not None:
                module.train(mode)

    @staticmethod
    def _pool_generic_output(outputs) -> torch.Tensor:
        pooler_output = getattr(outputs, "pooler_output", None)
        if pooler_output is not None:
            return pooler_output

        last_hidden_state = getattr(outputs, "last_hidden_state", None)
        if last_hidden_state is None:
            raise ValueError("Backbone output has neither pooler_output nor last_hidden_state.")

        if last_hidden_state.ndim == 3 and last_hidden_state.size(1) > 0:
            return last_hidden_state[:, 0]

        batch_size = last_hidden_state.shape[0]
        flattened = last_hidden_state.reshape(batch_size, -1, last_hidden_state.shape[-1])
        return flattened.mean(dim=1)

    def forward(self, pixel_values: torch.Tensor) -> torch.Tensor:
        if self.freeze_backbone and self.backbone_fully_frozen:
            with torch.no_grad():
                return self._forward_backbone(pixel_values)
        return self._forward_backbone(pixel_values)

    def _forward_backbone(self, pixel_values: torch.Tensor) -> torch.Tensor:
        if self.backbone_type == "clip":
            outputs = self.backbone(pixel_values=pixel_values)
            pooled = outputs.pooler_output
            return self.visual_projection(pooled)

        outputs = self.backbone(pixel_values=pixel_values)
        return self._pool_generic_output(outputs)

    def train(self, mode: bool = True):
        super().train(mode)
        if self.freeze_backbone:
            self.backbone.eval()
            if self.visual_projection is not None:
                self.visual_projection.eval()
            if not self.backbone_fully_frozen:
                self._set_swin_trainable_stages_mode(mode)
        return self


class SingleTaskTransformerClassifier(nn.Module):
    def __init__(self, cfg: dict, num_classes: int, task: str):
        super().__init__()
        self.task = task
        self.feature_extractor = TransformerFeatureExtractor(cfg)
        self.use_residual_stats = USE_RESIDUAL_STATS
        self.residual_stats_dim = active_residual_stats_dim()
        head_input_dim = self.feature_extractor.feature_dim + self.residual_stats_dim
        self.head = nn.Sequential(
            nn.Linear(head_input_dim, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(0.2),
            nn.Linear(256, num_classes),
        )

    def forward(
        self,
        pixel_values: torch.Tensor,
        residual_stats: torch.Tensor | None = None,
    ) -> torch.Tensor:
        features = self.feature_extractor(pixel_values)
        combined_features = combine_with_residual_stats(
            features,
            residual_stats,
            self.use_residual_stats,
            self.residual_stats_dim,
        )
        return self.head(combined_features)


class MultiTaskTransformerClassifier(nn.Module):
    def __init__(self, cfg: dict):
        super().__init__()
        self.feature_extractor = TransformerFeatureExtractor(cfg)
        self.use_residual_stats = USE_RESIDUAL_STATS
        self.residual_stats_dim = active_residual_stats_dim()
        head_input_dim = self.feature_extractor.feature_dim + self.residual_stats_dim
        self.binary_head = nn.Sequential(
            nn.Linear(head_input_dim, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(0.2),
            nn.Linear(256, len(CLASS_NAMES)),
        )
        self.transform_head = nn.Sequential(
            nn.Linear(head_input_dim, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(0.2),
            nn.Linear(256, len(TRANSFORM_NAMES)),
        )

    def forward(
        self,
        pixel_values: torch.Tensor,
        residual_stats: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        features = self.feature_extractor(pixel_values)
        combined_features = combine_with_residual_stats(
            features,
            residual_stats,
            self.use_residual_stats,
            self.residual_stats_dim,
        )
        binary_logits = self.binary_head(combined_features)
        transform_logits = self.transform_head(combined_features)
        return binary_logits, transform_logits


# == TRAIN ==

class FocalLoss(nn.Module):
    def __init__(self, gamma: float = 2.0, reduction: str = "mean"):
        super().__init__()
        self.gamma = gamma
        self.reduction = reduction

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        ce_loss = F.cross_entropy(logits, targets, reduction="none")
        pt = torch.exp(-ce_loss)
        focal_loss = ((1.0 - pt) ** self.gamma) * ce_loss

        if self.reduction == "mean":
            return focal_loss.mean()
        if self.reduction == "sum":
            return focal_loss.sum()
        if self.reduction == "none":
            return focal_loss
        raise ValueError(f"Unsupported reduction mode: {self.reduction}")


def build_multitask_transform_criterion() -> nn.Module:
    if USE_MULTITASK_TRANSFORM_FOCAL_LOSS:
        print(
            "Using FocalLoss for the multitask transform head "
            f"(gamma={MULTITASK_TRANSFORM_FOCAL_GAMMA})."
        )
        return FocalLoss(gamma=MULTITASK_TRANSFORM_FOCAL_GAMMA)

    print("Using CrossEntropyLoss for the multitask transform head.")
    return nn.CrossEntropyLoss()


def select_task_labels(binary_labels: torch.Tensor, transform_labels: torch.Tensor, task: str) -> torch.Tensor:
    if task == "binary":
        return binary_labels
    if task == "transform":
        return transform_labels
    raise ValueError(f"Unknown task: {task}")


def run_single_task_epoch(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    task: str,
    optimizer: torch.optim.Optimizer | None = None,
) -> tuple[float, float]:
    is_training = optimizer is not None
    model.train(is_training)

    total_loss = 0.0
    correct = 0
    total = 0

    with torch.set_grad_enabled(is_training):
        for batch in tqdm(loader, leave=False):
            pixel_values, residual_stats, binary_labels, transform_labels = unpack_batch(batch)
            pixel_values = pixel_values.to(DEVICE)
            labels = select_task_labels(binary_labels, transform_labels, task).to(DEVICE)

            logits = model(pixel_values, residual_stats)
            loss = criterion(logits, labels)
            if not torch.isfinite(loss):
                raise RuntimeError(f"Non-finite {task} loss encountered: {loss.item()}.")

            if is_training:
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                optimizer.step()

            total_loss += loss.item() * labels.size(0)
            correct += (logits.argmax(dim=1) == labels).sum().item()
            total += labels.size(0)

    return total_loss / total, correct / total


def run_multitask_epoch(
    model: nn.Module,
    loader: DataLoader,
    criterion_binary: nn.Module,
    criterion_transform: nn.Module,
    lambda_value: float,
    optimizer: torch.optim.Optimizer | None = None,
) -> tuple[float, float, float, float]:
    is_training = optimizer is not None
    model.train(is_training)

    total_loss = 0.0
    binary_correct = 0
    transform_correct = 0
    total = 0

    with torch.set_grad_enabled(is_training):
        for batch in tqdm(loader, leave=False):
            pixel_values, residual_stats, binary_labels, transform_labels = unpack_batch(batch)
            pixel_values = pixel_values.to(DEVICE)
            binary_labels = binary_labels.to(DEVICE)
            transform_labels = transform_labels.to(DEVICE)

            binary_logits, transform_logits = model(pixel_values, residual_stats)
            loss_binary = criterion_binary(binary_logits, binary_labels)
            loss_transform = criterion_transform(transform_logits, transform_labels)
            loss_total = lambda_value * loss_binary + (1.0 - lambda_value) * loss_transform
            if not torch.isfinite(loss_total):
                raise RuntimeError(f"Non-finite multitask loss encountered: {loss_total.item()}.")

            if is_training:
                optimizer.zero_grad(set_to_none=True)
                loss_total.backward()
                optimizer.step()

            total_loss += loss_total.item() * binary_labels.size(0)
            binary_correct += (binary_logits.argmax(dim=1) == binary_labels).sum().item()
            transform_correct += (transform_logits.argmax(dim=1) == transform_labels).sum().item()
            total += binary_labels.size(0)

    binary_acc = binary_correct / total
    transform_acc = transform_correct / total
    joint_score = (binary_acc + transform_acc) / 2.0
    return total_loss / total, binary_acc, transform_acc, joint_score


def verify_single_task_forward(model: nn.Module, loader: DataLoader, task: str, num_classes: int) -> None:
    model.eval()
    with torch.no_grad():
        pixel_values, residual_stats, binary_labels, transform_labels = unpack_batch(next(iter(loader)))
        logits = model(pixel_values.to(DEVICE), residual_stats)
        labels = select_task_labels(binary_labels, transform_labels, task)
        residual_shape = None if residual_stats is None else tuple(residual_stats.shape)
        print(
            f"{task} forward check: logits={tuple(logits.shape)}, "
            f"labels={tuple(labels.shape)}, residual_stats={residual_shape}"
        )
        if logits.shape[-1] != num_classes:
            raise RuntimeError(f"Expected {num_classes} logits for {task}, got {logits.shape[-1]}.")


def verify_multitask_forward(model: nn.Module, loader: DataLoader) -> None:
    model.eval()
    with torch.no_grad():
        pixel_values, residual_stats, _, _ = unpack_batch(next(iter(loader)))
        binary_logits, transform_logits = model(pixel_values.to(DEVICE), residual_stats)
        print(
            "multitask forward check: "
            f"binary_logits={tuple(binary_logits.shape)}, "
            f"transform_logits={tuple(transform_logits.shape)}, "
            f"residual_stats={None if residual_stats is None else tuple(residual_stats.shape)}"
        )
        if binary_logits.shape[-1] != len(CLASS_NAMES):
            raise RuntimeError("Invalid binary head output size.")
        if transform_logits.shape[-1] != len(TRANSFORM_NAMES):
            raise RuntimeError("Invalid transform head output size.")


def train_single_task_model(
    model_name: str,
    task: str,
    cfg: dict,
    train_loader: DataLoader,
    val_loader: DataLoader,
    settings: RunSettings,
) -> tuple[TrainResult, nn.Module]:
    num_classes = len(CLASS_NAMES) if task == "binary" else len(TRANSFORM_NAMES)
    effective_model_name = make_run_name(model_name)
    checkpoint_path = CHECKPOINT_DIR / f"{SELECTED_BACKBONE}_{effective_model_name}.pt"

    print("\n" + "=" * 72)
    print(f"Training {effective_model_name} | task={task} | backbone={SELECTED_BACKBONE}")
    print("=" * 72)

    model = SingleTaskTransformerClassifier(cfg, num_classes=num_classes, task=task).to(DEVICE)
    total_params, trainable_params = count_parameters(model)
    print(f"Total parameters: {total_params:,}")
    print(
        f"Trainable parameters: {trainable_params:,} "
        f"(freeze_backbone={FREEZE_BACKBONE}, "
        f"swin_trainable_stages={SWIN_TRAINABLE_STAGES}, "
        f"use_residual_stats={USE_RESIDUAL_STATS}, "
        f"residual_stats_dim={active_residual_stats_dim()})"
    )
    verify_single_task_forward(model, train_loader, task, num_classes)

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(
        filter(lambda parameter: parameter.requires_grad, model.parameters()),
        lr=LEARNING_RATE,
        weight_decay=WEIGHT_DECAY,
    )

    best_val_acc = -np.inf
    best_state_dict = None
    epochs_without_improvement = 0
    epochs_ran = 0

    for epoch in range(1, settings.num_epochs + 1):
        train_loss, train_acc = run_single_task_epoch(model, train_loader, criterion, task, optimizer)
        val_loss, val_acc = run_single_task_epoch(model, val_loader, criterion, task)
        epochs_ran = epoch

        print(
            f"Epoch {epoch:02d}/{settings.num_epochs} | "
            f"train_loss={train_loss:.4f} train_acc={train_acc:.4f} | "
            f"val_loss={val_loss:.4f} val_acc={val_acc:.4f}"
        )

        if val_acc > best_val_acc + MIN_DELTA:
            best_val_acc = val_acc
            best_state_dict = make_cpu_state_dict(model)
            torch.save(best_state_dict, checkpoint_path)
            epochs_without_improvement = 0
            print(f"  -> new best validation accuracy: {best_val_acc:.4f}")
        else:
            epochs_without_improvement += 1
            if epochs_without_improvement >= settings.patience:
                print(f"  -> early stopping after {epoch} epochs")
                break

    if best_state_dict is not None:
        model.load_state_dict(best_state_dict)

    train_result = TrainResult(
        model_name=effective_model_name,
        task_type=task,
        best_val_score=float(best_val_acc),
        checkpoint_path=checkpoint_path,
        epochs_ran=epochs_ran,
    )
    return train_result, model


def train_multitask_model(
    cfg: dict,
    train_loader: DataLoader,
    val_loader: DataLoader,
    settings: RunSettings,
    lambda_value: float,
) -> tuple[TrainResult, nn.Module]:
    model_name = make_run_name(f"multitask_lambda_{lambda_value:.2f}")
    checkpoint_path = CHECKPOINT_DIR / f"{SELECTED_BACKBONE}_{model_name}.pt"

    print("\n" + "=" * 72)
    print(f"Training {model_name} | backbone={SELECTED_BACKBONE} | lambda={lambda_value:.2f}")
    print("=" * 72)

    model = MultiTaskTransformerClassifier(cfg).to(DEVICE)
    total_params, trainable_params = count_parameters(model)
    print(f"Total parameters: {total_params:,}")
    print(
        f"Trainable parameters: {trainable_params:,} "
        f"(freeze_backbone={FREEZE_BACKBONE}, "
        f"swin_trainable_stages={SWIN_TRAINABLE_STAGES}, "
        f"use_residual_stats={USE_RESIDUAL_STATS}, "
        f"residual_stats_dim={active_residual_stats_dim()})"
    )
    verify_multitask_forward(model, train_loader)

    criterion_binary = nn.CrossEntropyLoss()
    criterion_transform = build_multitask_transform_criterion()
    optimizer = torch.optim.AdamW(
        filter(lambda parameter: parameter.requires_grad, model.parameters()),
        lr=LEARNING_RATE,
        weight_decay=WEIGHT_DECAY,
    )

    best_joint_score = -np.inf
    best_state_dict = None
    epochs_without_improvement = 0
    epochs_ran = 0

    for epoch in range(1, settings.num_epochs + 1):
        train_loss, train_binary_acc, train_transform_acc, train_joint = run_multitask_epoch(
            model,
            train_loader,
            criterion_binary,
            criterion_transform,
            lambda_value,
            optimizer,
        )
        val_loss, val_binary_acc, val_transform_acc, val_joint = run_multitask_epoch(
            model,
            val_loader,
            criterion_binary,
            criterion_transform,
            lambda_value,
        )
        epochs_ran = epoch

        print(
            f"Epoch {epoch:02d}/{settings.num_epochs} | "
            f"train_loss={train_loss:.4f} train_bin_acc={train_binary_acc:.4f} "
            f"train_trans_acc={train_transform_acc:.4f} train_joint={train_joint:.4f} | "
            f"val_loss={val_loss:.4f} val_bin_acc={val_binary_acc:.4f} "
            f"val_trans_acc={val_transform_acc:.4f} val_joint={val_joint:.4f}"
        )

        if val_joint > best_joint_score + MIN_DELTA:
            best_joint_score = val_joint
            best_state_dict = make_cpu_state_dict(model)
            torch.save(best_state_dict, checkpoint_path)
            epochs_without_improvement = 0
            print(f"  -> new best validation joint score: {best_joint_score:.4f}")
        else:
            epochs_without_improvement += 1
            if epochs_without_improvement >= settings.patience:
                print(f"  -> early stopping after {epoch} epochs")
                break

    if best_state_dict is not None:
        model.load_state_dict(best_state_dict)

    train_result = TrainResult(
        model_name=model_name,
        task_type="multitask",
        best_val_score=float(best_joint_score),
        checkpoint_path=checkpoint_path,
        epochs_ran=epochs_ran,
    )
    return train_result, model


# == EVALUATION ==

def collect_single_task_predictions(model: nn.Module, loader: DataLoader, task: str) -> dict[str, np.ndarray]:
    model.eval()
    binary_true = []
    transform_true = []
    y_prob = []

    with torch.no_grad():
        for batch in tqdm(loader, leave=False):
            pixel_values, residual_stats, binary_labels, transform_labels = unpack_batch(batch)
            pixel_values = pixel_values.to(DEVICE)
            logits = model(pixel_values, residual_stats)
            probabilities = torch.softmax(logits, dim=1).cpu().numpy()

            binary_true.extend(binary_labels.numpy())
            transform_true.extend(transform_labels.numpy())
            y_prob.extend(probabilities)

    binary_true = np.array(binary_true)
    transform_true = np.array(transform_true)
    y_prob = np.array(y_prob)
    y_true = binary_true if task == "binary" else transform_true
    y_pred = y_prob.argmax(axis=1)

    return {
        "y_true": y_true,
        "y_pred": y_pred,
        "y_prob": y_prob,
        "binary_true": binary_true,
        "transform_true": transform_true,
    }


def collect_multitask_predictions(model: nn.Module, loader: DataLoader) -> dict[str, np.ndarray]:
    model.eval()
    binary_true = []
    transform_true = []
    binary_prob = []
    transform_prob = []

    with torch.no_grad():
        for batch in tqdm(loader, leave=False):
            pixel_values, residual_stats, binary_labels, transform_labels = unpack_batch(batch)
            pixel_values = pixel_values.to(DEVICE)
            binary_logits, transform_logits = model(pixel_values, residual_stats)

            binary_true.extend(binary_labels.numpy())
            transform_true.extend(transform_labels.numpy())
            binary_prob.extend(torch.softmax(binary_logits, dim=1).cpu().numpy())
            transform_prob.extend(torch.softmax(transform_logits, dim=1).cpu().numpy())

    binary_true = np.array(binary_true)
    transform_true = np.array(transform_true)
    binary_prob = np.array(binary_prob)
    transform_prob = np.array(transform_prob)

    return {
        "binary_true": binary_true,
        "binary_pred": binary_prob.argmax(axis=1),
        "binary_prob": binary_prob,
        "transform_true": transform_true,
        "transform_pred": transform_prob.argmax(axis=1),
        "transform_prob": transform_prob,
    }


def compute_binary_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_prob: np.ndarray,
    transform_true: np.ndarray,
) -> dict[str, float]:
    metrics = {
        "binary_test_accuracy": accuracy_score(y_true, y_pred),
        "binary_test_macro_f1": f1_score(y_true, y_pred, average="macro", zero_division=0),
    }

    try:
        real_idx = CLASS_TO_IDX["real"]
        metrics["binary_test_roc_auc"] = roc_auc_score(y_true, y_prob[:, real_idx])
    except Exception:
        metrics["binary_test_roc_auc"] = np.nan

    for transform_idx, transform_name in IDX_TO_TRANSFORM.items():
        mask = transform_true == transform_idx
        key = f"binary_acc_{transform_name}"
        metrics[key] = accuracy_score(y_true[mask], y_pred[mask]) if mask.sum() else np.nan

    for class_idx, class_name in IDX_TO_CLASS.items():
        for transform_idx, transform_name in IDX_TO_TRANSFORM.items():
            mask = (y_true == class_idx) & (transform_true == transform_idx)
            key = f"binary_acc_{class_name}_{transform_name}"
            metrics[key] = accuracy_score(y_true[mask], y_pred[mask]) if mask.sum() else np.nan

    return metrics


def compute_transform_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float]:
    return {
        "transform_test_accuracy": accuracy_score(y_true, y_pred),
        "transform_test_macro_f1": f1_score(y_true, y_pred, average="macro", zero_division=0),
    }


def save_confusion_matrix(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    labels: list[int],
    target_names: list[str],
    title: str,
    out_path: Path,
) -> None:
    cm = confusion_matrix(y_true, y_pred, labels=labels)
    pd.DataFrame(cm, index=target_names, columns=target_names).to_csv(out_path.with_suffix(".csv"))

    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=target_names)
    disp.plot(cmap="Blues", values_format="d", xticks_rotation=30)
    plt.title(title)
    plt.tight_layout()
    plt.savefig(out_path, dpi=200)
    plt.close()


def write_report(
    out_path: Path,
    title: str,
    metrics: dict[str, float],
    y_true: np.ndarray,
    y_pred: np.ndarray,
    labels: list[int],
    target_names: list[str],
) -> None:
    report = classification_report(
        y_true,
        y_pred,
        labels=labels,
        target_names=target_names,
        zero_division=0,
    )

    lines = [title, "=" * len(title), ""]
    for key, value in metrics.items():
        if isinstance(value, float):
            lines.append(f"{key}: {value:.6f}" if not np.isnan(value) else f"{key}: NaN")
        else:
            lines.append(f"{key}: {value}")
    lines.extend(["", "Classification report:", report])

    with open(out_path, "w", encoding="utf-8") as file:
        file.write("\n".join(lines))


def evaluate_binary_outputs(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_prob: np.ndarray,
    transform_true: np.ndarray,
    run_name: str,
) -> dict[str, float]:
    metrics = compute_binary_metrics(y_true, y_pred, y_prob, transform_true)
    write_report(
        RESULTS_DIR / f"report_{run_name}.txt",
        f"{run_name} binary evaluation",
        metrics,
        y_true,
        y_pred,
        labels=list(range(len(CLASS_NAMES))),
        target_names=CLASS_NAMES,
    )
    save_confusion_matrix(
        y_true,
        y_pred,
        labels=list(range(len(CLASS_NAMES))),
        target_names=CLASS_NAMES,
        title=f"{run_name} binary confusion matrix",
        out_path=RESULTS_DIR / f"cm_{run_name}.png",
    )
    return metrics


def evaluate_transform_outputs(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    run_name: str,
) -> dict[str, float]:
    metrics = compute_transform_metrics(y_true, y_pred)
    write_report(
        RESULTS_DIR / f"report_{run_name}.txt",
        f"{run_name} transform evaluation",
        metrics,
        y_true,
        y_pred,
        labels=list(range(len(TRANSFORM_NAMES))),
        target_names=TRANSFORM_NAMES,
    )
    save_confusion_matrix(
        y_true,
        y_pred,
        labels=list(range(len(TRANSFORM_NAMES))),
        target_names=TRANSFORM_NAMES,
        title=f"{run_name} transform confusion matrix",
        out_path=RESULTS_DIR / f"cm_{run_name}.png",
    )
    return metrics


def evaluate_single_task_model(
    model: nn.Module,
    loader: DataLoader,
    task: str,
    run_name: str,
) -> dict[str, float]:
    predictions = collect_single_task_predictions(model, loader, task)

    if task == "binary":
        return evaluate_binary_outputs(
            predictions["y_true"],
            predictions["y_pred"],
            predictions["y_prob"],
            predictions["transform_true"],
            run_name,
        )

    return evaluate_transform_outputs(
        predictions["y_true"],
        predictions["y_pred"],
        run_name,
    )


def evaluate_multitask_model(
    model: nn.Module,
    loader: DataLoader,
    run_name: str,
) -> tuple[dict[str, float], dict[str, float]]:
    predictions = collect_multitask_predictions(model, loader)
    binary_metrics = evaluate_binary_outputs(
        predictions["binary_true"],
        predictions["binary_pred"],
        predictions["binary_prob"],
        predictions["transform_true"],
        f"{run_name}_binary",
    )
    transform_metrics = evaluate_transform_outputs(
        predictions["transform_true"],
        predictions["transform_pred"],
        f"{run_name}_transform",
    )
    return binary_metrics, transform_metrics


def build_comparison_table(
    binary_train: TrainResult,
    binary_metrics: dict[str, float],
    transform_train: TrainResult,
    transform_metrics: dict[str, float],
    multitask_results: list[dict[str, object]],
) -> pd.DataFrame:
    rows = [
        {
            "model_name": binary_train.model_name,
            "selected_backbone": SELECTED_BACKBONE,
            "task_type": "binary",
            "freeze_backbone": FREEZE_BACKBONE,
            "swin_trainable_stages": SWIN_TRAINABLE_STAGES,
            "use_residual_stats": USE_RESIDUAL_STATS,
            "residual_stats_dim": active_residual_stats_dim(),
            "multitask_transform_loss": np.nan,
            "multitask_transform_focal_gamma": np.nan,
            "lambda": np.nan,
            "binary_loss_weight": 1.0,
            "transform_loss_weight": np.nan,
            "best_val_score": binary_train.best_val_score,
            "epochs_ran": binary_train.epochs_ran,
            "binary_test_accuracy": binary_metrics.get("binary_test_accuracy", np.nan),
            "binary_test_macro_f1": binary_metrics.get("binary_test_macro_f1", np.nan),
            "binary_test_roc_auc": binary_metrics.get("binary_test_roc_auc", np.nan),
            "binary_acc_original": binary_metrics.get("binary_acc_original", np.nan),
            "binary_acc_redigital": binary_metrics.get("binary_acc_redigital", np.nan),
            "binary_acc_transfer": binary_metrics.get("binary_acc_transfer", np.nan),
            "transform_test_accuracy": np.nan,
            "transform_test_macro_f1": np.nan,
            "checkpoint_path": str(binary_train.checkpoint_path),
        },
        {
            "model_name": transform_train.model_name,
            "selected_backbone": SELECTED_BACKBONE,
            "task_type": "transform",
            "freeze_backbone": FREEZE_BACKBONE,
            "swin_trainable_stages": SWIN_TRAINABLE_STAGES,
            "use_residual_stats": USE_RESIDUAL_STATS,
            "residual_stats_dim": active_residual_stats_dim(),
            "multitask_transform_loss": np.nan,
            "multitask_transform_focal_gamma": np.nan,
            "lambda": np.nan,
            "binary_loss_weight": np.nan,
            "transform_loss_weight": 1.0,
            "best_val_score": transform_train.best_val_score,
            "epochs_ran": transform_train.epochs_ran,
            "binary_test_accuracy": np.nan,
            "binary_test_macro_f1": np.nan,
            "binary_test_roc_auc": np.nan,
            "binary_acc_original": np.nan,
            "binary_acc_redigital": np.nan,
            "binary_acc_transfer": np.nan,
            "transform_test_accuracy": transform_metrics.get("transform_test_accuracy", np.nan),
            "transform_test_macro_f1": transform_metrics.get("transform_test_macro_f1", np.nan),
            "checkpoint_path": str(transform_train.checkpoint_path),
        },
    ]
    rows.extend(multitask_results)
    return pd.DataFrame(rows)


# == MAIN ==

def main() -> None:
    set_seed(SEED)
    settings = get_run_settings()
    cfg = get_selected_model_config()

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)

    lambda_values = [LAMBDA] if FAST_DEV_RUN else LAMBDA_VALUES
    for lambda_value in lambda_values:
        validate_lambda(lambda_value)

    print(f"Device: {DEVICE}")
    print(f"FAST_DEV_RUN: {FAST_DEV_RUN}")
    print(f"Selected backbone: {SELECTED_BACKBONE} ({cfg['name']})")
    print(
        "Run settings: "
        f"samples_per_group={settings.samples_per_group}, "
        f"epochs={settings.num_epochs}, "
        f"patience={settings.patience}, "
        f"batch_size={settings.batch_size}, "
        f"num_workers={NUM_WORKERS}, "
        f"freeze_backbone={FREEZE_BACKBONE}, "
        f"swin_trainable_stages={SWIN_TRAINABLE_STAGES}, "
        f"use_residual_stats={USE_RESIDUAL_STATS}, "
        f"residual_stats_dim={active_residual_stats_dim()}, "
        f"multitask_transform_focal_loss={USE_MULTITASK_TRANSFORM_FOCAL_LOSS}, "
        f"multitask_transform_focal_gamma={MULTITASK_TRANSFORM_FOCAL_GAMMA}, "
        f"lambda_values={', '.join(f'{value:.2f}' for value in lambda_values)}"
    )

    full_index = build_image_index(DATA_ROOT)
    print_dataset_audit(full_index, "FULL DATASET")

    balanced_subset = sample_balanced_joint_subset(full_index, settings.samples_per_group, SEED)
    print_dataset_audit(balanced_subset, "BALANCED SUBSET")

    train_df, val_df, test_df = split_dataframe(balanced_subset, SEED)
    print_dataset_audit(train_df, "TRAIN SPLIT")
    print_dataset_audit(val_df, "VALIDATION SPLIT")
    print_dataset_audit(test_df, "TEST SPLIT")

    train_df.to_csv(RESULTS_DIR / "split_train.csv", index=False)
    val_df.to_csv(RESULTS_DIR / "split_val.csv", index=False)
    test_df.to_csv(RESULTS_DIR / "split_test.csv", index=False)

    if USE_RESIDUAL_STATS:
        cache_df = pd.concat([train_df, val_df, test_df], ignore_index=True)
        residual_stats_cache = build_or_load_residual_stats_csv(
            cache_df,
            RESIDUAL_STATS_CSV_PATH,
        )
    else:
        residual_stats_cache = None

    image_processor = get_image_processor(cfg)
    print(f"\nImage processor loaded for {SELECTED_BACKBONE}: {image_processor.__class__.__name__}")

    train_loader, val_loader, test_loader = make_loaders(
        train_df,
        val_df,
        test_df,
        image_processor,
        settings.batch_size,
        residual_stats_cache,
    )

    binary_train, binary_model = train_single_task_model(
        model_name="binary_baseline",
        task="binary",
        cfg=cfg,
        train_loader=train_loader,
        val_loader=val_loader,
        settings=settings,
    )
    binary_metrics = evaluate_single_task_model(binary_model, test_loader, "binary", binary_train.model_name)
    del binary_model
    clear_memory()

    transform_train, transform_model = train_single_task_model(
        model_name="transform_baseline",
        task="transform",
        cfg=cfg,
        train_loader=train_loader,
        val_loader=val_loader,
        settings=settings,
    )
    transform_metrics = evaluate_single_task_model(
        transform_model,
        test_loader,
        "transform",
        transform_train.model_name,
    )
    del transform_model
    clear_memory()

    multitask_rows = []
    for lambda_value in lambda_values:
        multitask_train, multitask_model = train_multitask_model(
            cfg=cfg,
            train_loader=train_loader,
            val_loader=val_loader,
            settings=settings,
            lambda_value=lambda_value,
        )
        multitask_binary_metrics, multitask_transform_metrics = evaluate_multitask_model(
            multitask_model,
            test_loader,
            multitask_train.model_name,
        )
        del multitask_model
        clear_memory()
        multitask_rows.append(
            {
                "model_name": multitask_train.model_name,
                "selected_backbone": SELECTED_BACKBONE,
                "task_type": "multitask",
                "freeze_backbone": FREEZE_BACKBONE,
                "swin_trainable_stages": SWIN_TRAINABLE_STAGES,
                "use_residual_stats": USE_RESIDUAL_STATS,
                "residual_stats_dim": active_residual_stats_dim(),
                "multitask_transform_loss": (
                    "focal" if USE_MULTITASK_TRANSFORM_FOCAL_LOSS else "cross_entropy"
                ),
                "multitask_transform_focal_gamma": (
                    MULTITASK_TRANSFORM_FOCAL_GAMMA
                    if USE_MULTITASK_TRANSFORM_FOCAL_LOSS
                    else np.nan
                ),
                "lambda": lambda_value,
                "binary_loss_weight": lambda_value,
                "transform_loss_weight": 1.0 - lambda_value,
                "best_val_score": multitask_train.best_val_score,
                "epochs_ran": multitask_train.epochs_ran,
                "binary_test_accuracy": multitask_binary_metrics.get("binary_test_accuracy", np.nan),
                "binary_test_macro_f1": multitask_binary_metrics.get("binary_test_macro_f1", np.nan),
                "binary_test_roc_auc": multitask_binary_metrics.get("binary_test_roc_auc", np.nan),
                "binary_acc_original": multitask_binary_metrics.get("binary_acc_original", np.nan),
                "binary_acc_redigital": multitask_binary_metrics.get("binary_acc_redigital", np.nan),
                "binary_acc_transfer": multitask_binary_metrics.get("binary_acc_transfer", np.nan),
                "transform_test_accuracy": multitask_transform_metrics.get("transform_test_accuracy", np.nan),
                "transform_test_macro_f1": multitask_transform_metrics.get("transform_test_macro_f1", np.nan),
                "checkpoint_path": str(multitask_train.checkpoint_path),
            }
        )

    comparison_df = build_comparison_table(
        binary_train,
        binary_metrics,
        transform_train,
        transform_metrics,
        multitask_rows,
    )
    comparison_path = RESULTS_DIR / "final_model_comparison.csv"
    comparison_df.to_csv(comparison_path, index=False)

    print("\n" + "=" * 72)
    print("FINAL MODEL COMPARISON")
    print("=" * 72)
    with pd.option_context("display.max_columns", None, "display.width", 180):
        print(comparison_df.to_string(index=False))

    print(f"\nSaved final comparison to: {comparison_path}")
    print(f"Saved reports and confusion matrices under: {RESULTS_DIR}")
    print(f"Saved checkpoints under: {CHECKPOINT_DIR}")


if __name__ == "__main__":
    mp.freeze_support()
    main()
