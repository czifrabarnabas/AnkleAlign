"""
03_evaluation.py
Ankle Alignment Model Evaluation Pipeline

Evaluates the trained model on the test set and generates metrics and visualizations.
"""

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, models
from PIL import Image
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    classification_report,
)
from tqdm import tqdm

from config import (
    OUTPUT_DIR,
    MODELS_DIR,
    MANIFEST_PATH,
    MODEL_PATH,
    IMAGE_SIZE,
    BATCH_SIZE,
    NUM_WORKERS,
    NUM_CLASSES,
    CLASS_NAMES,
    IMAGENET_MEAN,
    IMAGENET_STD,
)
from utils import print_evaluation_header, get_device


class AnkleDataset(Dataset):
    """Custom Dataset for ankle alignment images."""

    def __init__(self, df: pd.DataFrame, transform=None):
        self.df = df.reset_index(drop=True)
        self.transform = transform

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        image_path = row["image_path"]
        label = row["label"]

        image = Image.open(image_path).convert("RGB")

        if self.transform:
            image = self.transform(image)

        return image, label


def get_eval_transform():
    """Get image transforms for evaluation."""
    return transforms.Compose([
        transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
    ])


def create_model(num_classes: int = NUM_CLASSES):
    """Create ResNet-18 model architecture."""
    model = models.resnet18(weights=None)
    num_features = model.fc.in_features
    model.fc = nn.Linear(num_features, num_classes)
    return model


def load_model(model_path: Path, device: torch.device):
    """Load trained model from checkpoint."""
    model = create_model(num_classes=NUM_CLASSES)

    checkpoint = torch.load(model_path, map_location=device)
    model.load_state_dict(checkpoint["model_state_dict"])

    model = model.to(device)
    model.eval()

    print(f"Model loaded from {model_path}")
    print(f"  Checkpoint epoch: {checkpoint.get('epoch', 'N/A')}")
    print(f"  Checkpoint val_acc: {checkpoint.get('val_acc', 'N/A'):.2f}%")

    return model


def run_inference(model, dataloader, device):
    """Run inference on dataloader and return predictions and labels."""
    all_predictions = []
    all_labels = []
    all_probabilities = []

    with torch.no_grad():
        for images, labels in tqdm(dataloader, desc="Running inference"):
            images = images.to(device)

            outputs = model(images)
            probabilities = torch.softmax(outputs, dim=1)
            _, predictions = outputs.max(1)

            all_predictions.extend(predictions.cpu().numpy())
            all_labels.extend(labels.numpy())
            all_probabilities.extend(probabilities.cpu().numpy())

    return np.array(all_predictions), np.array(all_labels), np.array(all_probabilities)


def compute_metrics(y_true, y_pred, class_names):
    """Compute classification metrics."""
    metrics = {
        "accuracy": accuracy_score(y_true, y_pred),
        "precision_macro": precision_score(y_true, y_pred, average="macro", zero_division=0),
        "recall_macro": recall_score(y_true, y_pred, average="macro", zero_division=0),
        "f1_macro": f1_score(y_true, y_pred, average="macro", zero_division=0),
        "precision_weighted": precision_score(y_true, y_pred, average="weighted", zero_division=0),
        "recall_weighted": recall_score(y_true, y_pred, average="weighted", zero_division=0),
        "f1_weighted": f1_score(y_true, y_pred, average="weighted", zero_division=0),
    }

    # Per-class metrics
    precision_per_class = precision_score(y_true, y_pred, average=None, zero_division=0)
    recall_per_class = recall_score(y_true, y_pred, average=None, zero_division=0)
    f1_per_class = f1_score(y_true, y_pred, average=None, zero_division=0)

    for i, class_name in enumerate(class_names):
        metrics[f"precision_{class_name}"] = precision_per_class[i]
        metrics[f"recall_{class_name}"] = recall_per_class[i]
        metrics[f"f1_{class_name}"] = f1_per_class[i]

    return metrics


def plot_confusion_matrix(y_true, y_pred, class_names, output_path):
    """Generate and save confusion matrix visualization."""
    cm = confusion_matrix(y_true, y_pred)

    plt.figure(figsize=(10, 8))
    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=class_names,
        yticklabels=class_names,
    )
    plt.title("Confusion Matrix - Ankle Alignment Classification")
    plt.xlabel("Predicted Label")
    plt.ylabel("True Label")
    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()

    print(f"Confusion matrix saved to {output_path}")

    return cm


def plot_confusion_matrix_normalized(y_true, y_pred, class_names, output_path):
    """Generate and save normalized confusion matrix."""
    cm = confusion_matrix(y_true, y_pred, normalize="true")

    plt.figure(figsize=(10, 8))
    sns.heatmap(
        cm,
        annot=True,
        fmt=".2%",
        cmap="Blues",
        xticklabels=class_names,
        yticklabels=class_names,
    )
    plt.title("Normalized Confusion Matrix - Ankle Alignment Classification")
    plt.xlabel("Predicted Label")
    plt.ylabel("True Label")
    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()

    print(f"Normalized confusion matrix saved to {output_path}")


def main():
    print_evaluation_header()

    # Check if required files exist
    if not MANIFEST_PATH.exists():
        print(f"Error: Dataset manifest not found at {MANIFEST_PATH}")
        return

    if not MODEL_PATH.exists():
        print(f"Error: Model checkpoint not found at {MODEL_PATH}")
        return

    # Setup device
    device = get_device()

    # Load model
    print("\nLoading model...")
    model = load_model(MODEL_PATH, device)

    # Load test data
    df = pd.read_csv(MANIFEST_PATH)
    test_df = df[df["split"] == "test"]
    print(f"\nTest set size: {len(test_df)}")

    # Create dataset and dataloader
    test_dataset = AnkleDataset(test_df, transform=get_eval_transform())
    test_loader = DataLoader(
        test_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=NUM_WORKERS,
        pin_memory=True
    )

    # Run inference
    print("\nRunning inference on test set...")
    predictions, labels, probabilities = run_inference(model, test_loader, device)

    # Compute metrics
    print("\nComputing metrics...")
    metrics = compute_metrics(labels, predictions, CLASS_NAMES)

    # Print metrics
    print("\n" + "=" * 60)
    print("EVALUATION RESULTS")
    print("=" * 60)
    print(f"\nOverall Accuracy: {metrics['accuracy'] * 100:.2f}%")
    print(f"\nMacro-averaged Metrics:")
    print(f"  Precision: {metrics['precision_macro'] * 100:.2f}%")
    print(f"  Recall: {metrics['recall_macro'] * 100:.2f}%")
    print(f"  F1-Score: {metrics['f1_macro'] * 100:.2f}%")

    print(f"\nPer-class Metrics:")
    for class_name in CLASS_NAMES:
        print(f"\n  {class_name}:")
        print(f"    Precision: {metrics[f'precision_{class_name}'] * 100:.2f}%")
        print(f"    Recall: {metrics[f'recall_{class_name}'] * 100:.2f}%")
        print(f"    F1-Score: {metrics[f'f1_{class_name}'] * 100:.2f}%")

    # Generate classification report
    print("\n" + "=" * 60)
    print("Classification Report:")
    print(classification_report(labels, predictions, target_names=CLASS_NAMES, zero_division=0))

    # Plot confusion matrices
    cm = plot_confusion_matrix(
        labels, predictions, CLASS_NAMES,
        OUTPUT_DIR / "confusion_matrix.png"
    )
    plot_confusion_matrix_normalized(
        labels, predictions, CLASS_NAMES,
        OUTPUT_DIR / "confusion_matrix_normalized.png"
    )

    # Save evaluation report
    report = {
        "test_set_size": len(test_df),
        "class_names": CLASS_NAMES,
        "metrics": metrics,
        "confusion_matrix": cm.tolist(),
        "class_distribution": test_df["label_name"].value_counts().to_dict(),
    }

    report_path = OUTPUT_DIR / "evaluation_report.json"
    with open(report_path, "w") as f:
        json.dump(report, f, indent=2)
    print(f"\nEvaluation report saved to {report_path}")

    # Save predictions for analysis
    predictions_df = test_df.copy()
    predictions_df["predicted_label"] = predictions
    predictions_df["predicted_name"] = [CLASS_NAMES[p] for p in predictions]
    predictions_df["correct"] = predictions_df["label"] == predictions_df["predicted_label"]

    for i, class_name in enumerate(CLASS_NAMES):
        predictions_df[f"prob_{class_name}"] = probabilities[:, i]

    predictions_path = OUTPUT_DIR / "test_predictions.csv"
    predictions_df.to_csv(predictions_path, index=False)
    print(f"Test predictions saved to {predictions_path}")

    print("\n" + "=" * 60)
    print("Evaluation completed successfully!")
    print("=" * 60)


if __name__ == "__main__":
    main()
