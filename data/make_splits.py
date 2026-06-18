"""Create train/validation/test CSVs with joint-label stratification.

Run from the repository root:
    python data/make_splits.py

The model predicts two labels per image, so every split must preserve the joint
distribution of real/fake and transformation labels. By default this script
balances all six joint groups down to the smallest available group before
splitting, which handles the local 50999-image dataset cleanly.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd


REQUIRED_COLUMNS = {
    "image_path",
    "real_fake_label",
    "real_fake_name",
    "transform_label",
    "transform_name",
}

EXPECTED_REAL_FAKE = {"real", "fake"}
EXPECTED_TRANSFORMS = {"original", "internet_transmitted", "redigitized"}


def _validate_metadata(df: pd.DataFrame) -> None:
    """Check that metadata.csv has the columns and class names used later."""
    missing_columns = REQUIRED_COLUMNS.difference(df.columns)
    if missing_columns:
        missing = ", ".join(sorted(missing_columns))
        raise ValueError(f"Metadata CSV is missing required columns: {missing}")

    unknown_real_fake = set(df["real_fake_name"].unique()).difference(EXPECTED_REAL_FAKE)
    if unknown_real_fake:
        unknown = ", ".join(sorted(unknown_real_fake))
        raise ValueError(f"Unknown real/fake names found in metadata: {unknown}")

    unknown_transforms = set(df["transform_name"].unique()).difference(EXPECTED_TRANSFORMS)
    if unknown_transforms:
        unknown = ", ".join(sorted(unknown_transforms))
        raise ValueError(f"Unknown transformation names found in metadata: {unknown}")


def _joint_label(df: pd.DataFrame) -> pd.Series:
    """Combine both targets so stratification preserves the two-task balance."""
    return df["real_fake_name"].astype(str) + "__" + df["transform_name"].astype(str)


def _validate_max_samples(max_samples_per_group: int | None) -> None:
    if max_samples_per_group is not None and max_samples_per_group <= 0:
        raise ValueError("max_samples_per_group must be a positive integer or None.")


def _selected_group_sizes(
    original_sizes: pd.Series,
    balance_to_min_group: bool,
    max_samples_per_group: int | None,
) -> dict[str, int]:
    """Decide how many images to keep from each joint group.

    Balanced selection prevents a one-image raw-dataset mismatch from leaking
    into the final train/validation/test distributions.
    """
    min_group_size = int(original_sizes.min())
    selected_sizes: dict[str, int] = {}

    for joint_name, original_size in original_sizes.items():
        selected_size = min_group_size if balance_to_min_group else int(original_size)

        if max_samples_per_group is not None:
            selected_size = min(selected_size, max_samples_per_group)

        selected_sizes[str(joint_name)] = selected_size

    return selected_sizes


def _print_selection_summary(
    original_sizes: pd.Series,
    selected_sizes: dict[str, int],
    balance_to_min_group: bool,
) -> None:
    """Show exactly what was kept and discarded before splitting."""
    print("\nOriginal joint group sizes:")
    for joint_name, original_size in original_sizes.items():
        print(f"{joint_name}: {original_size}")

    if balance_to_min_group:
        print(f"\nBalancing to minimum joint group size: {int(original_sizes.min())}")
    else:
        print("\nBalancing to minimum joint group size: disabled")

    total_selected = 0
    total_discarded = 0

    print("\nSelected joint group sizes:")
    for joint_name, original_size in original_sizes.items():
        selected_size = selected_sizes[str(joint_name)]
        discarded = int(original_size) - selected_size
        total_selected += selected_size
        total_discarded += discarded
        print(
            f"{joint_name}: original={original_size}, "
            f"selected={selected_size}, discarded={discarded}"
        )

    print(f"\nTotal selected images: {total_selected}")
    print(f"Total discarded images: {total_discarded}")


def _shuffle_and_select_group(
    group: pd.DataFrame,
    rng_seed: int,
    selected_size: int,
) -> pd.DataFrame:
    """Shuffle deterministically, then keep the selected number of rows."""
    shuffled = group.sort_values("image_path").sample(frac=1.0, random_state=rng_seed)
    return shuffled.head(selected_size)


def _split_group(
    group: pd.DataFrame,
    train_ratio: float,
    val_ratio: float,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Split one already-balanced joint group into train/val/test slices."""
    n_items = len(group)
    n_train = int(n_items * train_ratio)
    n_val = int(n_items * val_ratio)
    n_test = n_items - n_train - n_val

    train = group.iloc[:n_train]
    val = group.iloc[n_train : n_train + n_val]
    test = group.iloc[n_train + n_val : n_train + n_val + n_test]

    return train, val, test


def _print_split_distribution(name: str, df: pd.DataFrame) -> None:
    """Print the two-task distribution of one generated split."""
    print(f"\n{name} split: {len(df)} images")
    print(pd.crosstab(df["real_fake_name"], df["transform_name"]))


def make_splits(
    metadata_csv: str | Path = "data/processed/metadata.csv",
    output_dir: str | Path = "data/splits",
    train_ratio: float = 0.70,
    val_ratio: float = 0.15,
    seed: int = 42,
    max_samples_per_group: int | None = None,
    balance_to_min_group: bool = True,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Create deterministic, joint-stratified train/val/test CSV files."""
    metadata_csv = Path(metadata_csv)
    output_dir = Path(output_dir)

    if not metadata_csv.exists():
        raise FileNotFoundError(f"Metadata CSV does not exist: {metadata_csv}")

    if train_ratio <= 0 or val_ratio <= 0 or train_ratio + val_ratio >= 1:
        raise ValueError("train_ratio and val_ratio must be positive and sum to less than 1.")

    _validate_max_samples(max_samples_per_group)

    df = pd.read_csv(metadata_csv)
    _validate_metadata(df)
    df = df.copy()
    df["_joint_label"] = _joint_label(df)

    # The joint label is the key stratification unit: for example,
    # fake__redigitized and real__original must remain balanced separately.
    original_sizes = df.groupby("_joint_label").size().sort_index()
    selected_sizes = _selected_group_sizes(
        original_sizes=original_sizes,
        balance_to_min_group=balance_to_min_group,
        max_samples_per_group=max_samples_per_group,
    )
    _print_selection_summary(original_sizes, selected_sizes, balance_to_min_group)

    train_parts = []
    val_parts = []
    test_parts = []

    for group_idx, (joint_name, group) in enumerate(sorted(df.groupby("_joint_label"))):
        group_seed = seed + group_idx
        group = _shuffle_and_select_group(group, group_seed, selected_sizes[str(joint_name)])
        train, val, test = _split_group(group, train_ratio, val_ratio)

        print(
            f"{joint_name}: total={len(group)}, "
            f"train={len(train)}, val={len(val)}, test={len(test)}"
        )

        train_parts.append(train)
        val_parts.append(val)
        test_parts.append(test)

    train_df = pd.concat(train_parts).sample(frac=1.0, random_state=seed).reset_index(drop=True)
    val_df = pd.concat(val_parts).sample(frac=1.0, random_state=seed).reset_index(drop=True)
    test_df = pd.concat(test_parts).sample(frac=1.0, random_state=seed).reset_index(drop=True)

    train_df = train_df.drop(columns=["_joint_label"])
    val_df = val_df.drop(columns=["_joint_label"])
    test_df = test_df.drop(columns=["_joint_label"])

    output_dir.mkdir(parents=True, exist_ok=True)
    train_df.to_csv(output_dir / "train.csv", index=False)
    val_df.to_csv(output_dir / "val.csv", index=False)
    test_df.to_csv(output_dir / "test.csv", index=False)

    print(f"\nSaved splits to: {output_dir}")
    _print_split_distribution("Train", train_df)
    _print_split_distribution("Validation", val_df)
    _print_split_distribution("Test", test_df)

    return train_df, val_df, test_df


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Create joint-stratified RRDataset splits.")
    parser.add_argument("--metadata-csv", default="data/processed/metadata.csv")
    parser.add_argument("--output-dir", default="data/splits")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max-samples-per-group", type=int, default=None)
    parser.add_argument(
        "--balance-to-min-group",
        dest="balance_to_min_group",
        action="store_true",
        default=True,
        help="Balance all joint groups down to the smallest group before splitting.",
    )
    parser.add_argument(
        "--no-balance-to-min-group",
        dest="balance_to_min_group",
        action="store_false",
        help="Keep original joint group sizes unless --max-samples-per-group is used.",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    make_splits(
        metadata_csv=args.metadata_csv,
        output_dir=args.output_dir,
        seed=args.seed,
        max_samples_per_group=args.max_samples_per_group,
        balance_to_min_group=args.balance_to_min_group,
    )
