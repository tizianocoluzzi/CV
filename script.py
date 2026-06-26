# == IMPORTS ==
import random
from pathlib import Path
import copy
import os
import multiprocessing as mp

import numpy as np
import pandas as pd
from PIL import Image
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

from transformers import (
    CLIPModel,
    CLIPImageProcessor,
    AutoImageProcessor,
    AutoModel,
    AutoConfig,
)

from sklearn.model_selection import train_test_split, GroupShuffleSplit
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    classification_report,
    confusion_matrix,
    ConfusionMatrixDisplay,
    roc_auc_score,
)

from tqdm.auto import tqdm


# == GLOBALS ==

# --- Dataset structure -------------------------------------------------
DATA_ROOT = Path("./data/raw/RRDataset_final")
SUBFOLDERS = ["original", "redigital", "transfer"]
CLASS_NAMES = ["ai", "real"]
TRANSFORM_NAMES = SUBFOLDERS

CLASS_TO_IDX = {name: idx for idx, name in enumerate(CLASS_NAMES)}
TRANSFORM_TO_IDX = {name: idx for idx, name in enumerate(TRANSFORM_NAMES)}

IDX_TO_CLASS = {idx: name for name, idx in CLASS_TO_IDX.items()}
IDX_TO_TRANSFORM = {idx: name for name, idx in TRANSFORM_TO_IDX.items()}

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}

# --- Preliminary benchmark settings -----------------------------------
SEED = 42

# Use the same balanced subset for every backbone.
# If None, the script uses the minimum available count among the 6 groups:
# ai/original, ai/redigital, ai/transfer, real/original, real/redigital, real/transfer.
SAMPLES_PER_GROUP = 150

# First run these models. Keep clip-vit-l disabled unless you have a strong GPU.
BACKBONES_TO_TEST = [
    "deit-tiny",
    "deit-small",
    "swin-tiny",
    "clip-vit-b",
]

# The preliminary phase evaluates the backbone on the two single-task baselines.
# The final project can then reuse the selected backbone for:
# 1) binary baseline, 2) transformation baseline, 3) two-head multi-task model.
TASKS_TO_TEST = ["binary", "transform"]

# --- Model configs ------------------------------------------------------
MODEL_CONFIGS = {
    "clip-vit-l": {
        "type": "clip",
        "name": "openai/clip-vit-large-patch14",
    },
    "clip-vit-b": {
        "type": "clip",
        "name": "openai/clip-vit-base-patch32",
    },
    "swin-tiny": {
        "type": "generic",
        "name": "microsoft/swin-tiny-patch4-window7-224",
    },
    "deit-tiny": {
        "type": "generic",
        "name": "facebook/deit-tiny-patch16-224",
    },
    "deit-small": {
        "type": "generic",
        "name": "facebook/deit-small-patch16-224",
    },
}

FREEZE_BACKBONE = True  # preliminary search only evaluates the head, not the backbone

# --- Hardware ----------------------------------------------------------
DEVICE = torch.device(
    "cuda" if torch.cuda.is_available()
    else "mps" if torch.backends.mps.is_available()
    else "cpu"
)

# --- Dataloader --------------------------------------------------------
BATCH_SIZE = 16
NUM_WORKERS = min(4, max(1, (os.cpu_count() or 2) // 2))

# --- Train / validation / test split ----------------------------------
TEST_SIZE = 0.2
VAL_SIZE = 0.1
USE_GROUP_SPLIT = True

# --- Training ----------------------------------------------------------
NUM_EPOCHS = 15
LEARNING_RATE = 1e-4
WEIGHT_DECAY = 1e-4
PATIENCE = 4
MIN_DELTA = 1e-4

CHECKPOINT_DIR = Path("./checkpoints/preliminary_backbone_search")
RESULTS_DIR = Path("./results/preliminary_backbone_search")

print(f"Device: {DEVICE}")


# == UTILS ==

def set_seed(seed: int = SEED) -> None:
    """Make sampling and shuffling reproducible."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def is_image_file(path: Path) -> bool:
    return path.suffix.lower() in IMAGE_EXTENSIONS


def build_image_index(data_root: Path = DATA_ROOT) -> pd.DataFrame:
    """
    Walk the dataset folder structure:

        data_root / <subfolder> / <class> / *.jpg

    Returns one row per image:
        path             absolute image path
        label            ai / real
        subfolder        original / redigital / transfer
        binary_label     integer label for ai / real
        transform_label  integer label for original / redigital / transfer
        stratify_key     joint key used for balanced sampling and stratification
        image_id         filename stem, used to reduce leakage between transformed versions
    """
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
    print(df["label"].value_counts().sort_index())
    print("\nBy transformation:")
    print(df["subfolder"].value_counts().sort_index())
    print("\nBy label x transformation:")
    print(df.groupby(["label", "subfolder"]).size())


def sample_balanced_joint_subset(
    df: pd.DataFrame,
    samples_per_group: int | None = SAMPLES_PER_GROUP,
    seed: int = SEED,
) -> pd.DataFrame:
    """
    Balance the dataset on the joint label:

        ai/original, ai/redigital, ai/transfer,
        real/original, real/redigital, real/transfer.

    This is better than balancing only ai vs real because the project has two tasks:
    binary source classification and transformation classification.
    """
    group_counts = df.groupby(["label", "subfolder"]).size()
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
            sampled = group_df.sample(n=n_take, random_state=seed)
            subsets.append(sampled)

    subset_df = pd.concat(subsets, ignore_index=True)
    subset_df = subset_df.sample(frac=1.0, random_state=seed).reset_index(drop=True)
    return subset_df


def split_dataframe(df: pd.DataFrame, seed: int = SEED):
    """
    Create one fixed train/val/test split for all backbone comparisons.

    If USE_GROUP_SPLIT is True and image_id repeats across rows, the split keeps all
    versions of the same image_id in the same split to reduce data leakage.
    Otherwise it falls back to stratified splitting on label x transformation.
    """
    repeated_ids = df["image_id"].duplicated().any()

    if USE_GROUP_SPLIT and repeated_ids:
        print("\nUsing group split by image_id to reduce leakage between transformed versions.")

        splitter = GroupShuffleSplit(n_splits=1, test_size=TEST_SIZE, random_state=seed)
        train_val_idx, test_idx = next(splitter.split(df, groups=df["image_id"]))
        train_val_df = df.iloc[train_val_idx].reset_index(drop=True)
        test_df = df.iloc[test_idx].reset_index(drop=True)

        val_fraction_of_train_val = VAL_SIZE
        splitter_val = GroupShuffleSplit(n_splits=1, test_size=val_fraction_of_train_val, random_state=seed)
        train_idx, val_idx = next(splitter_val.split(train_val_df, groups=train_val_df["image_id"]))
        train_df = train_val_df.iloc[train_idx].reset_index(drop=True)
        val_df = train_val_df.iloc[val_idx].reset_index(drop=True)

        print("[note] GroupShuffleSplit does not guarantee perfect class stratification.")
        print("[note] Check the split distributions printed below.")

    else:
        print("\nUsing stratified split by label x transformation.")

        train_val_df, test_df = train_test_split(
            df,
            test_size=TEST_SIZE,
            stratify=df["stratify_key"],
            random_state=seed,
        )
        train_df, val_df = train_test_split(
            train_val_df,
            test_size=VAL_SIZE,
            stratify=train_val_df["stratify_key"],
            random_state=seed,
        )

        train_df = train_df.reset_index(drop=True)
        val_df = val_df.reset_index(drop=True)
        test_df = test_df.reset_index(drop=True)

    return train_df, val_df, test_df


def get_image_processor(cfg: dict):
    if cfg["type"] == "clip":
        return CLIPImageProcessor.from_pretrained(cfg["name"])
    return AutoImageProcessor.from_pretrained(cfg["name"])


def clear_memory() -> None:
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    if DEVICE.type == "mps":
        try:
            torch.mps.empty_cache()
        except Exception:
            pass


# == DATASET ==

class AIRealTransformDataset(Dataset):
    """
    Loads images and returns:
        pixel_values, binary_label, transform_label
    """

    def __init__(self, dataframe: pd.DataFrame, image_processor):
        self.df = dataframe.reset_index(drop=True)
        self.image_processor = image_processor

    def __len__(self) -> int:
        return len(self.df)

    def __getitem__(self, idx: int):
        row = self.df.iloc[idx]
        image = Image.open(row["path"]).convert("RGB")
        pixel_values = self.image_processor(images=image, return_tensors="pt")["pixel_values"][0]

        binary_label = int(row["binary_label"])
        transform_label = int(row["transform_label"])

        return pixel_values, binary_label, transform_label


# == MODEL ==

class BackboneClassifier(nn.Module):
    """
    Unified single-head classifier for preliminary backbone comparison.

    It supports:
        - CLIP vision encoder
        - DeiT / Swin / other image backbones through AutoModel
    """

    def __init__(self, cfg: dict, num_classes: int, freeze_backbone: bool = FREEZE_BACKBONE):
        super().__init__()
        self.cfg = cfg
        self.backbone_type = cfg["type"]
        hf_name = cfg["name"]

        config = AutoConfig.from_pretrained(hf_name)
        print(f"Model: {hf_name} | HF model_type: {config.model_type}")

        if self.backbone_type == "clip":
            clip_model = CLIPModel.from_pretrained(hf_name)
            self.backbone = clip_model.vision_model
            self.projection = clip_model.visual_projection
            embed_dim = clip_model.config.projection_dim
        else:
            self.backbone = AutoModel.from_pretrained(hf_name)
            self.projection = None
            embed_dim = self._infer_embed_dim(self.backbone)

        if freeze_backbone:
            for p in self.backbone.parameters():
                p.requires_grad = False
            if self.projection is not None:
                for p in self.projection.parameters():
                    p.requires_grad = False

        self.head = nn.Sequential(
            nn.Linear(embed_dim, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(0.2),
            nn.Linear(256, num_classes),
        )

    @staticmethod
    def _infer_embed_dim(backbone) -> int:
        cfg = backbone.config
        if hasattr(cfg, "hidden_size"):
            return cfg.hidden_size
        if hasattr(cfg, "embed_dim"):
            return cfg.embed_dim
        if hasattr(cfg, "projection_dim"):
            return cfg.projection_dim
        raise ValueError("Could not infer backbone embedding dimension from config.")

    def _pool_generic_output(self, outputs):
        if hasattr(outputs, "pooler_output") and outputs.pooler_output is not None:
            return outputs.pooler_output
        if hasattr(outputs, "last_hidden_state") and outputs.last_hidden_state is not None:
            return outputs.last_hidden_state[:, 0]
        raise ValueError("Backbone output has neither pooler_output nor last_hidden_state.")

    def forward_features(self, x):
        if self.backbone_type == "clip":
            out = self.backbone(pixel_values=x).pooler_output
            return self.projection(out)

        outputs = self.backbone(pixel_values=x)
        return self._pool_generic_output(outputs)

    def forward(self, x):
        features = self.forward_features(x)
        return self.head(features)


# == TRAINING / EVALUATION ==

def select_labels(binary_labels, transform_labels, task: str):
    if task == "binary":
        return binary_labels
    if task == "transform":
        return transform_labels
    raise ValueError(f"Unknown task: {task}")


def run_epoch(model, loader, criterion, task: str, optimizer=None):
    is_training = optimizer is not None
    model.train(is_training)

    total_loss, correct, total = 0.0, 0, 0

    with torch.set_grad_enabled(is_training):
        for pixel_values, binary_labels, transform_labels in tqdm(loader, leave=False):
            pixel_values = pixel_values.to(DEVICE)
            labels = select_labels(binary_labels, transform_labels, task).to(DEVICE)

            logits = model(pixel_values)
            loss = criterion(logits, labels)

            if is_training:
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                optimizer.step()

            total_loss += loss.item() * labels.size(0)
            correct += (logits.argmax(dim=1) == labels).sum().item()
            total += labels.size(0)

    clear_memory()
    return total_loss / total, correct / total


def predict(model, loader, task: str):
    model.eval()
    all_labels = []
    all_preds = []
    all_probs = []
    all_binary_labels = []
    all_transform_labels = []

    with torch.no_grad():
        for pixel_values, binary_labels, transform_labels in tqdm(loader, leave=False):
            pixel_values = pixel_values.to(DEVICE)
            labels = select_labels(binary_labels, transform_labels, task)

            logits = model(pixel_values)
            probs = torch.softmax(logits, dim=1).cpu().numpy()

            all_labels.extend(labels.numpy())
            all_preds.extend(probs.argmax(axis=1))
            all_probs.extend(probs)
            all_binary_labels.extend(binary_labels.numpy())
            all_transform_labels.extend(transform_labels.numpy())

    return {
        "y_true": np.array(all_labels),
        "y_pred": np.array(all_preds),
        "y_prob": np.array(all_probs),
        "binary_true": np.array(all_binary_labels),
        "transform_true": np.array(all_transform_labels),
    }


def compute_metrics(prediction: dict, task: str) -> dict:
    y_true = prediction["y_true"]
    y_pred = prediction["y_pred"]
    y_prob = prediction["y_prob"]

    metrics = {
        "accuracy": accuracy_score(y_true, y_pred),
        "macro_f1": f1_score(y_true, y_pred, average="macro"),
    }

    if task == "binary":
        try:
            real_idx = CLASS_TO_IDX["real"]
            metrics["roc_auc"] = roc_auc_score(y_true, y_prob[:, real_idx])
        except Exception:
            metrics["roc_auc"] = np.nan

        # Robustness view: binary accuracy separately on original/redigital/transfer.
        for transform_idx, transform_name in IDX_TO_TRANSFORM.items():
            mask = prediction["transform_true"] == transform_idx
            if mask.sum() > 0:
                metrics[f"acc_on_{transform_name}"] = accuracy_score(y_true[mask], y_pred[mask])
            else:
                metrics[f"acc_on_{transform_name}"] = np.nan
    else:
        metrics["roc_auc"] = np.nan
        for transform_name in TRANSFORM_NAMES:
            metrics[f"acc_on_{transform_name}"] = np.nan

    return metrics


def plot_and_save_confusion_matrix(prediction: dict, task: str, backbone_name: str, out_dir: Path) -> None:
    labels = CLASS_NAMES if task == "binary" else TRANSFORM_NAMES
    cm = confusion_matrix(prediction["y_true"], prediction["y_pred"])

    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=labels)
    disp.plot(cmap="Blues", values_format="d", xticks_rotation=30)
    plt.title(f"{backbone_name} — {task} confusion matrix")
    plt.tight_layout()

    out_path = out_dir / f"cm_{backbone_name}_{task}.png"
    plt.savefig(out_path, dpi=200)
    plt.close()


def make_loaders(train_df, val_df, test_df, image_processor):
    train_dataset = AIRealTransformDataset(train_df, image_processor)
    val_dataset = AIRealTransformDataset(val_df, image_processor)
    test_dataset = AIRealTransformDataset(test_df, image_processor)

    pin_memory = DEVICE.type == "cuda"
    persistent_workers = NUM_WORKERS > 0

    train_loader = DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=NUM_WORKERS,
        pin_memory=pin_memory,
        persistent_workers=persistent_workers,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=NUM_WORKERS,
        pin_memory=pin_memory,
        persistent_workers=persistent_workers,
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=NUM_WORKERS,
        pin_memory=pin_memory,
        persistent_workers=persistent_workers,
    )

    sample_pixels, sample_binary, sample_transform = train_dataset[0]
    print(
        f"Sample tensor shape: {sample_pixels.shape}, "
        f"binary={sample_binary} ({IDX_TO_CLASS[sample_binary]}), "
        f"transform={sample_transform} ({IDX_TO_TRANSFORM[sample_transform]})"
    )

    return train_loader, val_loader, test_loader


def train_one_model(backbone_name: str, task: str, train_loader, val_loader, test_loader):
    cfg = MODEL_CONFIGS[backbone_name]
    num_classes = len(CLASS_NAMES) if task == "binary" else len(TRANSFORM_NAMES)

    print(f"\n============================================================")
    print(f"Backbone: {backbone_name} | Task: {task} | Classes: {num_classes}")
    print(f"============================================================")

    model = BackboneClassifier(cfg, num_classes=num_classes, freeze_backbone=FREEZE_BACKBONE).to(DEVICE)

    n_total_params = sum(p.numel() for p in model.parameters())
    n_trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Total parameters:     {n_total_params:,}")
    print(f"Trainable parameters: {n_trainable_params:,}  (backbone frozen: {FREEZE_BACKBONE})")

    model.eval()
    with torch.no_grad():
        pixel_values, binary_labels, transform_labels = next(iter(train_loader))
        pixel_values = pixel_values.to(DEVICE)
        logits = model(pixel_values)
        print(f"Logits shape check: {logits.shape}")

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=LEARNING_RATE,
        weight_decay=WEIGHT_DECAY,
    )

    checkpoint_path = CHECKPOINT_DIR / f"{backbone_name}_{task}.pt"
    history = {"train_loss": [], "train_acc": [], "val_loss": [], "val_acc": []}

    best_val_acc = -np.inf
    best_state_dict = None
    epochs_without_improvement = 0

    for epoch in range(1, NUM_EPOCHS + 1):
        train_loss, train_acc = run_epoch(model, train_loader, criterion, task, optimizer)
        val_loss, val_acc = run_epoch(model, val_loader, criterion, task, optimizer=None)

        history["train_loss"].append(train_loss)
        history["train_acc"].append(train_acc)
        history["val_loss"].append(val_loss)
        history["val_acc"].append(val_acc)

        print(
            f"Epoch {epoch:02d}/{NUM_EPOCHS} | "
            f"train_loss={train_loss:.4f} train_acc={train_acc:.4f} | "
            f"val_loss={val_loss:.4f} val_acc={val_acc:.4f}"
        )

        if val_acc > best_val_acc + MIN_DELTA:
            best_val_acc = val_acc
            best_state_dict = copy.deepcopy(model.state_dict())
            torch.save(best_state_dict, checkpoint_path)
            epochs_without_improvement = 0
            print(f"  -> new best val_acc={best_val_acc:.4f}, checkpoint saved to {checkpoint_path}")
        else:
            epochs_without_improvement += 1
            if epochs_without_improvement >= PATIENCE:
                print(f"  -> early stopping after {epoch} epochs")
                break

    if best_state_dict is not None:
        model.load_state_dict(best_state_dict)

    prediction = predict(model, test_loader, task)
    metrics = compute_metrics(prediction, task)

    print(f"\nBest validation accuracy: {best_val_acc:.4f}")
    print(f"Test accuracy: {metrics['accuracy']:.4f}")
    print(f"Test macro F1: {metrics['macro_f1']:.4f}")
    if task == "binary":
        print(f"Test ROC-AUC: {metrics['roc_auc']:.4f}")
        for transform_name in TRANSFORM_NAMES:
            print(f"Binary accuracy on {transform_name}: {metrics[f'acc_on_{transform_name}']:.4f}")

    target_names = CLASS_NAMES if task == "binary" else TRANSFORM_NAMES
    report = classification_report(prediction["y_true"], prediction["y_pred"], target_names=target_names)
    print("\nClassification report:")
    print(report)

    report_path = RESULTS_DIR / f"report_{backbone_name}_{task}.txt"
    with open(report_path, "w", encoding="utf-8") as f:
        f.write(report)

    plot_and_save_confusion_matrix(prediction, task, backbone_name, RESULTS_DIR)

    result_row = {
        "backbone": backbone_name,
        "task": task,
        "best_val_acc": best_val_acc,
        "test_accuracy": metrics["accuracy"],
        "test_macro_f1": metrics["macro_f1"],
        "test_roc_auc": metrics["roc_auc"],
        "binary_acc_original": metrics["acc_on_original"],
        "binary_acc_redigital": metrics["acc_on_redigital"],
        "binary_acc_transfer": metrics["acc_on_transfer"],
        "total_params": n_total_params,
        "trainable_params": n_trainable_params,
        "epochs_ran": len(history["train_loss"]),
        "checkpoint": str(checkpoint_path),
    }

    del model
    clear_memory()

    return result_row, history


def main():
    set_seed(SEED)
    CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    full_index = build_image_index()
    print_dataset_audit(full_index, "FULL DATASET")

    data_subset = sample_balanced_joint_subset(full_index, SAMPLES_PER_GROUP, SEED)
    print_dataset_audit(data_subset, "BALANCED SUBSET USED FOR BACKBONE SEARCH")

    train_df, val_df, test_df = split_dataframe(data_subset, SEED)

    print_dataset_audit(train_df, "TRAIN SPLIT")
    print_dataset_audit(val_df, "VALIDATION SPLIT")
    print_dataset_audit(test_df, "TEST SPLIT")

    # Save the exact splits so every future experiment can reuse them.
    train_df.to_csv(RESULTS_DIR / "split_train.csv", index=False)
    val_df.to_csv(RESULTS_DIR / "split_val.csv", index=False)
    test_df.to_csv(RESULTS_DIR / "split_test.csv", index=False)

    all_results = []

    for backbone_name in BACKBONES_TO_TEST:
        cfg = MODEL_CONFIGS[backbone_name]
        image_processor = get_image_processor(cfg)
        print(f"\nImage processor for {backbone_name}:\n{image_processor}")

        train_loader, val_loader, test_loader = make_loaders(train_df, val_df, test_df, image_processor)

        for task in TASKS_TO_TEST:
            result_row, _ = train_one_model(
                backbone_name=backbone_name,
                task=task,
                train_loader=train_loader,
                val_loader=val_loader,
                test_loader=test_loader,
            )
            all_results.append(result_row)

            results_df = pd.DataFrame(all_results)
            results_df.to_csv(RESULTS_DIR / "backbone_preliminary_results.csv", index=False)
            print("\nCurrent results table:")
            print(results_df.sort_values(["task", "test_macro_f1"], ascending=[True, False]))

        del train_loader, val_loader, test_loader
        clear_memory()

    results_df = pd.DataFrame(all_results)
    results_path = RESULTS_DIR / "backbone_preliminary_results.csv"
    results_df.to_csv(results_path, index=False)

    print("\n============================================================")
    print("FINAL PRELIMINARY BACKBONE RESULTS")
    print("============================================================")
    print(results_df.sort_values(["task", "test_macro_f1"], ascending=[True, False]))
    print(f"\nSaved results to: {results_path}")
    print(f"Saved reports/confusion matrices to: {RESULTS_DIR}")

    # Compact pivot table useful for choosing the final backbone.
    pivot = results_df.pivot_table(
        index="backbone",
        columns="task",
        values=["test_accuracy", "test_macro_f1", "test_roc_auc"],
        aggfunc="first",
    )
    pivot_path = RESULTS_DIR / "backbone_preliminary_pivot.csv"
    pivot.to_csv(pivot_path)
    print(f"Saved pivot table to: {pivot_path}")


if __name__ == "__main__":
    mp.freeze_support()
    main()
