"""
02_train.py
Ankle Alignment Model Training Pipeline

Trains a ResNet-18 model for 3-class ankle alignment classification.
"""

import json
from pathlib import Path

import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, models
from PIL import Image
from tqdm import tqdm

from config import (
    OUTPUT_DIR,
    MODELS_DIR,
    MANIFEST_PATH,
    BATCH_SIZE,
    NUM_EPOCHS,
    LEARNING_RATE,
    PATIENCE,
    NUM_WORKERS,
    IMAGE_SIZE,
    NUM_CLASSES,
    PRETRAINED,
    MODEL_NAME,
    IMAGENET_MEAN,
    IMAGENET_STD,
    print_config,
)
from utils import (
    print_model_summary,
    print_training_header,
    print_data_summary,
    get_device,
    format_time,
)


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

        # Load image
        image = Image.open(image_path).convert("RGB")

        if self.transform:
            image = self.transform(image)

        return image, label


def get_transforms(is_training: bool):
    """Get image transforms for training or validation/test."""
    if is_training:
        return transforms.Compose([
            transforms.Resize((IMAGE_SIZE + 32, IMAGE_SIZE + 32)),
            transforms.RandomCrop(IMAGE_SIZE),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomRotation(degrees=15),
            transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
            transforms.ToTensor(),
            transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
        ])
    else:
        return transforms.Compose([
            transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
            transforms.ToTensor(),
            transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
        ])


def create_model(num_classes: int = NUM_CLASSES, pretrained: bool = PRETRAINED):
    """Create ResNet-18 model with custom classifier head."""
    model = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1 if pretrained else None)

    # Replace the final fully connected layer
    num_features = model.fc.in_features
    model.fc = nn.Linear(num_features, num_classes)

    return model


def train_epoch(model, dataloader, criterion, optimizer, device):
    """Train for one epoch."""
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0

    pbar = tqdm(dataloader, desc="Training", leave=False)
    for images, labels in pbar:
        images = images.to(device)
        labels = labels.to(device)

        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item() * images.size(0)
        _, predicted = outputs.max(1)
        total += labels.size(0)
        correct += predicted.eq(labels).sum().item()

        pbar.set_postfix({"loss": loss.item(), "acc": 100.0 * correct / total})

    epoch_loss = running_loss / total
    epoch_acc = 100.0 * correct / total
    return epoch_loss, epoch_acc


def validate(model, dataloader, criterion, device):
    """Validate the model."""
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0

    with torch.no_grad():
        for images, labels in tqdm(dataloader, desc="Validating", leave=False):
            images = images.to(device)
            labels = labels.to(device)

            outputs = model(images)
            loss = criterion(outputs, labels)

            running_loss += loss.item() * images.size(0)
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()

    epoch_loss = running_loss / total
    epoch_acc = 100.0 * correct / total
    return epoch_loss, epoch_acc


def main():
    print_training_header()

    # Print configuration
    print_config()

    # Create output directories
    MODELS_DIR.mkdir(parents=True, exist_ok=True)

    # Check if manifest exists
    if not MANIFEST_PATH.exists():
        print(f"Error: Dataset manifest not found at {MANIFEST_PATH}")
        print("Run 01_data_processing.py first")
        return

    # Load dataset manifest
    df = pd.read_csv(MANIFEST_PATH)
    print(f"\nLoaded manifest with {len(df)} samples")

    # Split data
    train_df = df[df["split"] == "train"]
    val_df = df[df["split"] == "val"]
    test_df = df[df["split"] == "test"]

    # Print data summary
    class_dist = {
        "train": train_df["label_name"].value_counts().to_dict(),
        "validation": val_df["label_name"].value_counts().to_dict(),
        "test": test_df["label_name"].value_counts().to_dict(),
    }
    print_data_summary(len(train_df), len(val_df), len(test_df), class_dist)

    # Create datasets
    train_dataset = AnkleDataset(train_df, transform=get_transforms(is_training=True))
    val_dataset = AnkleDataset(val_df, transform=get_transforms(is_training=False))

    # Create dataloaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=NUM_WORKERS,
        pin_memory=True
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=NUM_WORKERS,
        pin_memory=True
    )

    # Setup device
    device = get_device()

    # Create model
    model = create_model(num_classes=NUM_CLASSES, pretrained=PRETRAINED)

    # Print model summary with parameter counts
    print_model_summary(model, f"{MODEL_NAME} (pretrained={PRETRAINED})")

    model = model.to(device)

    # Loss and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="max", factor=0.5, patience=5, verbose=True
    )

    # Training loop
    best_val_acc = 0.0
    epochs_without_improvement = 0
    training_history = []

    print(f"\nStarting training for {NUM_EPOCHS} epochs...")
    print("-" * 60)

    for epoch in range(NUM_EPOCHS):
        print(f"\nEpoch {epoch + 1}/{NUM_EPOCHS}")

        # Train
        train_loss, train_acc = train_epoch(model, train_loader, criterion, optimizer, device)

        # Validate
        val_loss, val_acc = validate(model, val_loader, criterion, device)

        # Update scheduler
        scheduler.step(val_acc)

        # Log metrics
        print(f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.2f}%")
        print(f"Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.2f}%")

        training_history.append({
            "epoch": epoch + 1,
            "train_loss": train_loss,
            "train_acc": train_acc,
            "val_loss": val_loss,
            "val_acc": val_acc,
        })

        # Save best model
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            epochs_without_improvement = 0
            checkpoint_path = MODELS_DIR / "best_model.pth"
            torch.save({
                "epoch": epoch + 1,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "val_acc": val_acc,
                "val_loss": val_loss,
            }, checkpoint_path)
            print(f"New best model saved! Val Acc: {val_acc:.2f}%")
        else:
            epochs_without_improvement += 1
            print(f"No improvement for {epochs_without_improvement} epochs")

        # Early stopping
        if epochs_without_improvement >= PATIENCE:
            print(f"\nEarly stopping triggered after {epoch + 1} epochs")
            break

    # Save training history
    history_df = pd.DataFrame(training_history)
    history_path = OUTPUT_DIR / "training_history.csv"
    history_df.to_csv(history_path, index=False)
    print(f"\nTraining history saved to {history_path}")

    # Save training config
    config = {
        "model": MODEL_NAME,
        "pretrained": PRETRAINED,
        "num_classes": NUM_CLASSES,
        "batch_size": BATCH_SIZE,
        "learning_rate": LEARNING_RATE,
        "num_epochs": NUM_EPOCHS,
        "patience": PATIENCE,
        "image_size": IMAGE_SIZE,
        "best_val_acc": best_val_acc,
        "final_epoch": epoch + 1,
    }
    config_path = OUTPUT_DIR / "training_config.json"
    with open(config_path, "w") as f:
        json.dump(config, f, indent=2)
    print(f"Training config saved to {config_path}")

    print(f"\nTraining completed!")
    print(f"Best validation accuracy: {best_val_acc:.2f}%")
    print(f"Model saved to {MODELS_DIR / 'best_model.pth'}")


if __name__ == "__main__":
    main()
