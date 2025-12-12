"""
config.py
Configuration file containing hyperparameters and paths for AnkleAlign project.
"""

from pathlib import Path

# =============================================================================
# Paths (inside container)
# =============================================================================
DATA_DIR = Path("/data")
OUTPUT_DIR = Path("/app/output")
MODELS_DIR = OUTPUT_DIR / "models"
MANIFEST_PATH = OUTPUT_DIR / "dataset_manifest.csv"
MODEL_PATH = MODELS_DIR / "best_model.pth"

# Data paths
CONSENSUS_DIR = DATA_DIR / "anklealign" / "anklealign" / "consensus"

# =============================================================================
# Training Hyperparameters
# =============================================================================
NUM_EPOCHS = 50
BATCH_SIZE = 32
LEARNING_RATE = 1e-4
PATIENCE = 10  # Early stopping patience
NUM_WORKERS = 0  # Set to 0 to avoid shared memory issues in Docker
IMAGE_SIZE = 224

# =============================================================================
# Model Configuration
# =============================================================================
MODEL_NAME = "resnet18"
PRETRAINED = True
NUM_CLASSES = 3

# =============================================================================
# Data Split Ratios
# =============================================================================
TRAIN_RATIO = 0.70
VAL_RATIO = 0.15
TEST_RATIO = 0.15
RANDOM_STATE = 42

# =============================================================================
# Class Configuration
# =============================================================================
LABEL_MAP = {
    "1_Pronacio": 0,
    "2_Neutralis": 1,
    "3_Szupinacio": 2,
}

CLASS_NAMES = ["Pronacio", "Neutralis", "Szupinacio"]

# =============================================================================
# ImageNet Normalization (for pretrained models)
# =============================================================================
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]


def print_config():
    """Print all configuration parameters."""
    print("=" * 60)
    print("CONFIGURATION")
    print("=" * 60)
    print(f"\n[Paths]")
    print(f"  Data directory: {DATA_DIR}")
    print(f"  Output directory: {OUTPUT_DIR}")
    print(f"  Models directory: {MODELS_DIR}")
    print(f"  Consensus directory: {CONSENSUS_DIR}")

    print(f"\n[Training Hyperparameters]")
    print(f"  Number of epochs: {NUM_EPOCHS}")
    print(f"  Batch size: {BATCH_SIZE}")
    print(f"  Learning rate: {LEARNING_RATE}")
    print(f"  Early stopping patience: {PATIENCE}")
    print(f"  Number of workers: {NUM_WORKERS}")
    print(f"  Image size: {IMAGE_SIZE}x{IMAGE_SIZE}")

    print(f"\n[Model Configuration]")
    print(f"  Model: {MODEL_NAME}")
    print(f"  Pretrained: {PRETRAINED}")
    print(f"  Number of classes: {NUM_CLASSES}")

    print(f"\n[Data Split Ratios]")
    print(f"  Train: {TRAIN_RATIO * 100:.0f}%")
    print(f"  Validation: {VAL_RATIO * 100:.0f}%")
    print(f"  Test: {TEST_RATIO * 100:.0f}%")
    print(f"  Random state: {RANDOM_STATE}")

    print(f"\n[Classes]")
    for label_name, idx in LABEL_MAP.items():
        print(f"  {idx}: {label_name} -> {CLASS_NAMES[idx]}")

    print("=" * 60)
