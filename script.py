# == IMPORT ==
import random
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

from transformers import (
    CLIPModel, CLIPImageProcessor,
    SwinModel, AutoImageProcessor,
    DeiTModel, DeiTImageProcessor,
    ViTModel, ViTImageProcessor,
)
from transformers import AutoConfig
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    ConfusionMatrixDisplay,
    roc_auc_score,
    roc_curve,
)

from tqdm.auto import tqdm
import copy

# == GLOBALS ==

# --- Dataset structure -------------------------------------------------
DATA_ROOT = Path("./data/raw/RRDataset_final")
SUBFOLDERS = ["original", "redigital", "transfer"]   # processing type, kept as metadata only
CLASS_NAMES = ["ai", "real"]                          # classification target
CLASS_TO_IDX = {name: idx for idx, name in enumerate(CLASS_NAMES)}
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}

# --- Sampling ------------------------------------------------------------
SAMPLES_PER_CLASS = 1000   # images per class (ai / real) to load; set to None to use ALL images
SEED = 42

# --- Model -----------------------------------------------------------------
MODEL_CONFIGS = {
    "clip-vit-l": {
        "type":       "clip",
        "name":       "openai/clip-vit-large-patch14",
    },
    "clip-vit-b": {
        "type":       "clip",
        "name":       "openai/clip-vit-base-patch32",
    },
    "swin-tiny": {
        "type":       "swin",
        "name":       "microsoft/swin-tiny-patch4-window7-224",
    },
    "deit-tiny": {
        "type":       "deit",
        "name":       "facebook/deit-tiny-patch16-224",
    },
    "deit-small": {
        "type":       "deit",
        "name":       "facebook/deit-small-patch16-224",
    },
}

NUM_CLASSES = len(CLASS_NAMES)
CHOSEN_MODEL = "deit-tiny"
CLIP_MODEL_NAME = MODEL_CONFIGS[CHOSEN_MODEL]["name"]
FREEZE_BACKBONE = False   # freeze CLIP weights, train only the classification head

# --- Hardware ----------------------------------------------------------
DEVICE = torch.device(
    "cuda" if torch.cuda.is_available()
    else "mps" if torch.backends.mps.is_available()
    else "cpu"
)

# --- Dataloader ----------------------------------------------------------
BATCH_SIZE = 16
NUM_WORKERS = 8

# --- Train / validation / test split -------------------------------------
TEST_SIZE = 0.2     # fraction of the balanced subset held out as the final test set
VAL_SIZE = 0.1       # fraction of the remaining train+val data used for validation during training

# --- Training --------------------------------------------------------------
NUM_EPOCHS = 200
LEARNING_RATE = 1e-4
WEIGHT_DECAY = 1e-4
CHECKPOINT_DIR = Path("./checkpoints")
CHECKPOINT_PATH = (CHECKPOINT_DIR / f"{CHOSEN_MODEL}.pt")

print(f"Device: {DEVICE}")

# == UTILS ==

def set_seed(seed: int = SEED) -> None:
    """Make sampling / shuffling reproducible."""
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

    and return a DataFrame with one row per image:
        path        - absolute path to the image file
        label       - "ai" or "real" (the classification target)
        subfolder   - "original" / "redigital" / "transfer" (metadata only)
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
                            "path": str(img_path),
                            "label": class_name,
                            "subfolder": subfolder,
                        }
                    )

    df = pd.DataFrame(records)
    if df.empty:
        raise RuntimeError(f"No images found under {data_root}. Check the dataset path/structure.")
    return df


full_index = build_image_index()
print(f"Total images found: {len(full_index)}")
print(full_index.groupby("label").size())
print(full_index.groupby(["subfolder", "label"]).size())


def sample_balanced_subset(
    df: pd.DataFrame,
    samples_per_class: "int | None" = SAMPLES_PER_CLASS,
    seed: int = SEED,
) -> pd.DataFrame:
    """Take an equally-sized, class-balanced subset of the full index."""
    subsets = []

    for class_name in CLASS_NAMES:
        class_df = df[df["label"] == class_name]
        n_available = len(class_df)
        n_take = n_available if samples_per_class is None else min(samples_per_class, n_available)

        sampled = class_df.sample(n=n_take, random_state=seed)
        subsets.append(sampled)

        if samples_per_class is not None and n_available < samples_per_class:
            print(f"[warning] class '{class_name}' has only {n_available} images, "
                  f"fewer than the requested {samples_per_class}.")

    subset_df = pd.concat(subsets, ignore_index=True)
    subset_df = subset_df.sample(frac=1.0, random_state=seed).reset_index(drop=True)  # shuffle
    return subset_df


set_seed(SEED)
data_subset = sample_balanced_subset(full_index, SAMPLES_PER_CLASS, SEED)
print(f"Subset size: {len(data_subset)}")
print(data_subset.groupby("label").size())
print(data_subset.groupby(["subfolder", "label"]).size())



train_val_df, test_df = train_test_split(
    data_subset,
    test_size=TEST_SIZE,
    stratify=data_subset["label"],
    random_state=SEED,
)
train_df, val_df = train_test_split(
    train_val_df,
    test_size=VAL_SIZE,
    stratify=train_val_df["label"],
    random_state=SEED,
)

train_df = train_df.reset_index(drop=True)
val_df = val_df.reset_index(drop=True)
test_df = test_df.reset_index(drop=True)

print(f"Train: {len(train_df)}  |  Val: {len(val_df)}  |  Test: {len(test_df)}")
for name, split_df in [("train", train_df), ("val", val_df), ("test", test_df)]:
    print(f"  {name}:", dict(split_df["label"].value_counts()))


class AIRealDataset(Dataset):
    """Loads images and returns CLIP-preprocessed pixel tensors + integer labels."""

    def __init__(self, dataframe: pd.DataFrame, image_processor: CLIPImageProcessor):
        self.df = dataframe.reset_index(drop=True)
        self.image_processor = image_processor

    def __len__(self) -> int:
        return len(self.df)

    def __getitem__(self, idx: int):
        row = self.df.iloc[idx]
        image = Image.open(row["path"]).convert("RGB")

        pixel_values = self.image_processor(images=image, return_tensors="pt")["pixel_values"][0]
        label = CLASS_TO_IDX[row["label"]]

        return pixel_values, label

t = MODEL_CONFIGS[CHOSEN_MODEL]["type"]
print(t)
if t == "clip":
    clip_image_processor = CLIPImageProcessor.from_pretrained(CLIP_MODEL_NAME)
elif t == "swin":
    clip_image_processor = AutoImageProcessor.from_pretrained(CLIP_MODEL_NAME)
elif t == "deit":
    clip_image_processor = ViTImageProcessor.from_pretrained(CLIP_MODEL_NAME)
else:
    clip_image_processor = CLIPImageProcessor.from_pretrained(CLIP_MODEL_NAME)

print(f"model info:\n{clip_image_processor}")
train_dataset = AIRealDataset(train_df, clip_image_processor)
val_dataset = AIRealDataset(val_df, clip_image_processor)
test_dataset = AIRealDataset(test_df, clip_image_processor)

train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True,
                           num_workers=NUM_WORKERS, pin_memory=True, persistent_workers=True)
val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False,
                         num_workers=NUM_WORKERS, pin_memory=True, persistent_workers=True)
test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False,
                          num_workers=NUM_WORKERS, pin_memory=True, persistent_workers=True)

# quick sanity check
sample_pixels, sample_label = train_dataset[0]
print(f"Sample tensor shape: {sample_pixels.shape}, label: {sample_label} ({CLASS_NAMES[sample_label]})")


class TransformerClassifier(nn.Module):
    """
    Unified classifier supporting CLIP ViT-B, Swin-Tiny, DeiT-Tiny/Small.
    backbone_type: 'clip' | 'swin' | 'deit'
    """

    def __init__(self, cfg: dict, num_classes: int = NUM_CLASSES,
                 freeze_backbone: bool = FREEZE_BACKBONE):
        super().__init__()
        btype = cfg["type"]
        hf_name = cfg["name"]
        
        c = AutoConfig.from_pretrained(hf_name)
        print(f"model type: {c.model_type} ")
        if btype == "clip":
            _clip = CLIPModel.from_pretrained(hf_name)
            self.backbone        = _clip.vision_model
            self.projection      = _clip.visual_projection
            embed_dim            = _clip.config.projection_dim          # 512 for ViT-B/32
            self._forward_fn     = self._clip_forward

        elif btype == "swin":
            self.backbone        = SwinModel.from_pretrained(hf_name)
            self.projection      = None
            embed_dim            = self.backbone.config.hidden_size     # 768 for Swin-Tiny
            self._forward_fn     = self._swin_forward

        elif btype == "deit":
            self.backbone        = ViTModel.from_pretrained(
                                       hf_name, add_pooling_layer=True)
            self.projection      = None
            embed_dim            = self.backbone.config.hidden_size     # 192/384
            self._forward_fn     = self._deit_forward

        else:
            raise ValueError(f"Unknown backbone type: {btype!r}")

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

    # --- per-backbone pooling helpers ---
    def _clip_forward(self, x):
        out = self.backbone(pixel_values=x).pooler_output
        return self.projection(out)                     # [B, embed_dim]

    def _swin_forward(self, x):
        return self.backbone(pixel_values=x).pooler_output   # [B, embed_dim]

    def _deit_forward(self, x):
        return self.backbone(pixel_values=x).pooler_output   # [B, embed_dim]

    def forward(self, x):
        features = self._forward_fn(x)
        return self.head(features)


model = TransformerClassifier(MODEL_CONFIGS[CHOSEN_MODEL]).to(DEVICE)

n_total_params = sum(p.numel() for p in model.parameters())
n_trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
print(f"Total parameters:     {n_total_params:,}")
print(f"Trainable parameters: {n_trainable_params:,}  (backbone frozen: {FREEZE_BACKBONE})")


model.eval()
with torch.no_grad():
    pixel_values, labels = next(iter(train_loader))
    pixel_values = pixel_values.to(DEVICE)
    logits = model(pixel_values)
    print(f"Logits shape: {logits.shape}")  # [BATCH_SIZE, NUM_CLASSES]


criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.AdamW(
    filter(lambda p: p.requires_grad, model.parameters()),
    lr=LEARNING_RATE,
    weight_decay=WEIGHT_DECAY,
)


def run_epoch(model, loader, criterion, optimizer=None):
    """
    Run one pass over `loader`.

    If `optimizer` is given, the model is set to train mode and weights are updated
    (a training epoch). Otherwise the model is set to eval mode and no gradients are
    computed (a validation / test epoch).

    Returns: (average_loss, accuracy)
    """
    is_training = optimizer is not None
    model.train(is_training)

    total_loss, correct, total = 0.0, 0, 0

    with torch.set_grad_enabled(is_training):
        for pixel_values, labels in tqdm(loader, leave=False):
            pixel_values = pixel_values.to(DEVICE)
            labels = labels.to(DEVICE)

            logits = model(pixel_values)
            loss = criterion(logits, labels)

            if is_training:
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

            total_loss += loss.item() * labels.size(0)
            correct += (logits.argmax(dim=1) == labels).sum().item()
            total += labels.size(0)
    
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    return total_loss / total, correct / total


set_seed(SEED)
CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)

history = {"train_loss": [], "train_acc": [], "val_loss": [], "val_acc": []}
best_val_acc = 0.0
best_state_dict = None

for epoch in range(1, NUM_EPOCHS + 1):
    train_loss, train_acc = run_epoch(model, train_loader, criterion, optimizer)
    torch.cuda.empty_cache()  # <-- add this
    val_loss, val_acc = run_epoch(model, val_loader, criterion, optimizer=None)

    history["train_loss"].append(train_loss)
    history["train_acc"].append(train_acc)
    history["val_loss"].append(val_loss)
    history["val_acc"].append(val_acc)

    print(
        f"Epoch {epoch:02d}/{NUM_EPOCHS} | "
        f"train_loss={train_loss:.4f} train_acc={train_acc:.4f} | "
        f"val_loss={val_loss:.4f} val_acc={val_acc:.4f}"
    )

    if val_acc > best_val_acc:
        best_val_acc = val_acc
        best_state_dict = copy.deepcopy(model.state_dict())
        torch.save(best_state_dict, CHECKPOINT_PATH)
        print(f"  -> new best val_acc={best_val_acc:.4f}, checkpoint saved to {CHECKPOINT_PATH}")

# reload the best-performing weights before evaluation
if best_state_dict is not None:
    model.load_state_dict(best_state_dict)
print(f"\nBest validation accuracy: {best_val_acc:.4f}")



fig, axes = plt.subplots(1, 2, figsize=(12, 4))

axes[0].plot(history["train_loss"], label="train")
axes[0].plot(history["val_loss"], label="val")
axes[0].set_title("Loss")
axes[0].set_xlabel("epoch")
axes[0].legend()

axes[1].plot(history["train_acc"], label="train")
axes[1].plot(history["val_acc"], label="val")
axes[1].set_title("Accuracy")
axes[1].set_xlabel("epoch")
axes[1].legend()

plt.tight_layout()
plt.show()


def predict(model, loader):
    """Run inference and return (true_labels, predicted_labels, prob_of_'real')."""
    model.eval()
    all_labels, all_preds, all_probs = [], [], []

    with torch.no_grad():
        for pixel_values, labels in tqdm(loader, leave=False):
            pixel_values = pixel_values.to(DEVICE)
            logits = model(pixel_values)
            probs = torch.softmax(logits, dim=1)

            all_labels.extend(labels.numpy())
            all_preds.extend(probs.argmax(dim=1).cpu().numpy())
            all_probs.extend(probs[:, CLASS_TO_IDX["real"]].cpu().numpy())

    return np.array(all_labels), np.array(all_preds), np.array(all_probs)


y_true, y_pred, y_prob_real = predict(model, test_loader)
test_acc = accuracy_score(y_true, y_pred)
print(f"Test accuracy: {test_acc:.4f}")


print(classification_report(y_true, y_pred, target_names=CLASS_NAMES))

cm = confusion_matrix(y_true, y_pred)
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=CLASS_NAMES)
disp.plot(cmap="Blues", values_format="d")
plt.title("Confusion matrix — test set")
plt.show()
