"""
01_data_processing.py
Ankle Alignment Data Processing Pipeline

Parses Label Studio JSON consensus annotations and creates train/val/test splits.
"""

import json
import os
import re
from pathlib import Path
from sklearn.model_selection import train_test_split
import pandas as pd

from config import (
    DATA_DIR,
    OUTPUT_DIR,
    CONSENSUS_DIR,
    LABEL_MAP,
    CLASS_NAMES,
    TRAIN_RATIO,
    VAL_RATIO,
    TEST_RATIO,
    RANDOM_STATE,
    print_config,
)


def parse_consensus_json(json_path: Path) -> list[dict]:
    """
    Parse a Label Studio JSON export file and extract image paths with labels.

    Returns list of dicts: [{"image_path": str, "label": int, "label_name": str}, ...]
    """
    with open(json_path, "r", encoding="utf-8") as f:
        try:
            data = json.load(f)
        except json.JSONDecodeError:
            print(f"Warning: Could not parse {json_path}")
            return []

    if not isinstance(data, list):
        print(f"Warning: Expected list in {json_path}, got {type(data)}")
        return []

    results = []
    for item in data:
        annotations = item.get("annotations", [])
        if not annotations:
            continue

        # Get the first annotation result
        for annotation in annotations:
            result_list = annotation.get("result", [])
            for result in result_list:
                if result.get("type") == "choices":
                    choices = result.get("value", {}).get("choices", [])
                    if choices:
                        label_name = choices[0]
                        if label_name in LABEL_MAP:
                            # Extract image filename from file_upload field
                            file_upload = item.get("file_upload", "")
                            # file_upload format: "uuid-filename.jpg"
                            # We need to find the actual file
                            image_info = {
                                "file_upload": file_upload,
                                "label": LABEL_MAP[label_name],
                                "label_name": label_name,
                            }
                            results.append(image_info)
                            break
            break  # Only use first annotation

    return results


def find_image_files(data_dir: Path) -> dict[str, Path]:
    """
    Scan data directory and build a mapping of filename -> full path.
    Handles both internet_* and sajat_* images.
    """
    image_map = {}
    image_extensions = {".jpg", ".jpeg", ".png", ".webp"}

    for root, dirs, files in os.walk(data_dir):
        for file in files:
            if Path(file).suffix.lower() in image_extensions:
                full_path = Path(root) / file
                # Store by filename (without uuid prefix if present)
                image_map[file.lower()] = full_path
                # Also try without uuid prefix (format: uuid-filename.jpg)
                if "-" in file:
                    parts = file.split("-", 1)
                    if len(parts) == 2:
                        image_map[parts[1].lower()] = full_path

    return image_map


def resolve_image_path(file_upload: str, image_map: dict[str, Path]) -> Path | None:
    """
    Resolve the actual image path from file_upload field.
    file_upload format: "uuid-filename.jpg"

    Label Studio adds folder identifiers to filenames in various patterns:
    - sajat_reszvevo01_01_D6AE9F.jpg (suffix)
    - D6AE9F_sajat_reszvevo01_01.jpg (prefix)

    Actual file on disk: sajat_reszvevo01_01.jpg (in D6AE9F/ folder)
    """
    if not file_upload:
        return None

    # Try exact match first
    if file_upload.lower() in image_map:
        return image_map[file_upload.lower()]

    # Try without uuid prefix (format: uuid-filename.jpg)
    filename = file_upload
    if "-" in file_upload:
        parts = file_upload.split("-", 1)
        if len(parts) == 2:
            filename = parts[1]
            if filename.lower() in image_map:
                return image_map[filename.lower()]

    # Try stripping folder suffix (e.g., sajat_reszvevo01_01_D6AE9F.jpg -> sajat_reszvevo01_01.jpg)
    # Pattern: name_FOLDERID.ext where FOLDERID is 6 uppercase alphanumeric chars
    base, ext = os.path.splitext(filename)
    # Remove folder suffix like _D6AE9F, _ECSGGY, etc.
    stripped = re.sub(r'_[A-Z0-9]{6}$', '', base)
    if stripped != base:
        candidate = (stripped + ext).lower()
        if candidate in image_map:
            return image_map[candidate]

    # Try stripping folder prefix (e.g., D6AE9F_sajat_reszvevo01_01.jpg -> sajat_reszvevo01_01.jpg)
    stripped = re.sub(r'^[A-Z0-9]{6}_', '', base)
    if stripped != base:
        candidate = (stripped + ext).lower()
        if candidate in image_map:
            return image_map[candidate]

    return None


def create_dataset_manifest(consensus_dir: Path, data_dir: Path) -> pd.DataFrame:
    """
    Process all consensus JSON files and create dataset manifest.
    """
    print(f"Scanning for images in {data_dir}...")
    image_map = find_image_files(data_dir)
    print(f"Found {len(image_map)} image files")

    all_samples = []
    json_files = list(consensus_dir.glob("*.json"))
    print(f"Found {len(json_files)} consensus JSON files")

    for json_file in json_files:
        samples = parse_consensus_json(json_file)
        for sample in samples:
            image_path = resolve_image_path(sample["file_upload"], image_map)
            if image_path and image_path.exists():
                all_samples.append({
                    "image_path": str(image_path),
                    "label": sample["label"],
                    "label_name": sample["label_name"],
                    "source_json": json_file.name,
                })
            else:
                print(f"Warning: Could not find image for {sample['file_upload']}")

    df = pd.DataFrame(all_samples)
    print(f"\nTotal samples with resolved images: {len(df)}")

    if len(df) > 0:
        print("\nClass distribution:")
        print(df["label_name"].value_counts())

    return df


def create_splits(df: pd.DataFrame, train_ratio: float = TRAIN_RATIO,
                  val_ratio: float = VAL_RATIO, test_ratio: float = TEST_RATIO,
                  random_state: int = RANDOM_STATE) -> pd.DataFrame:
    """
    Create stratified train/validation/test splits.
    """
    if len(df) == 0:
        return df

    # First split: train vs (val + test)
    train_df, temp_df = train_test_split(
        df,
        train_size=train_ratio,
        stratify=df["label"],
        random_state=random_state
    )

    # Second split: val vs test (from remaining data)
    relative_val_ratio = val_ratio / (val_ratio + test_ratio)
    val_df, test_df = train_test_split(
        temp_df,
        train_size=relative_val_ratio,
        stratify=temp_df["label"],
        random_state=random_state
    )

    # Assign split labels
    train_df = train_df.copy()
    val_df = val_df.copy()
    test_df = test_df.copy()

    train_df["split"] = "train"
    val_df["split"] = "val"
    test_df["split"] = "test"

    result_df = pd.concat([train_df, val_df, test_df], ignore_index=True)

    print(f"\nSplit sizes:")
    print(f"  Train: {len(train_df)} ({len(train_df)/len(df)*100:.1f}%)")
    print(f"  Val:   {len(val_df)} ({len(val_df)/len(df)*100:.1f}%)")
    print(f"  Test:  {len(test_df)} ({len(test_df)/len(df)*100:.1f}%)")

    print(f"\nClass distribution per split:")
    print(result_df.groupby(["split", "label_name"]).size().unstack(fill_value=0))

    return result_df


def main():
    print("=" * 60)
    print("AnkleAlign Data Processing")
    print("=" * 60)

    # Print configuration
    print_config()

    # Create output directory
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # Check if consensus directory exists
    if not CONSENSUS_DIR.exists():
        print(f"Error: Consensus directory not found at {CONSENSUS_DIR}")
        print("Make sure data is mounted correctly at /data")
        return

    # Create dataset manifest
    df = create_dataset_manifest(CONSENSUS_DIR, DATA_DIR)

    if len(df) == 0:
        print("Error: No valid samples found!")
        return

    # Create splits
    df = create_splits(df)

    # Save manifest
    manifest_path = OUTPUT_DIR / "dataset_manifest.csv"
    df.to_csv(manifest_path, index=False)
    print(f"\nDataset manifest saved to {manifest_path}")

    # Save class names for reference
    class_info = {
        "class_names": CLASS_NAMES,
        "label_map": LABEL_MAP,
        "num_classes": len(CLASS_NAMES),
    }
    class_info_path = OUTPUT_DIR / "class_info.json"
    with open(class_info_path, "w") as f:
        json.dump(class_info, f, indent=2)
    print(f"Class info saved to {class_info_path}")

    print("\n" + "=" * 60)
    print("Data processing completed successfully!")
    print("=" * 60)


if __name__ == "__main__":
    main()
