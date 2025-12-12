"""
04_inference.py
Ankle Alignment Model Inference Pipeline

Runs inference on new, unseen images to generate predictions.
"""

import json
import os
from pathlib import Path

import torch
import torch.nn as nn
from torchvision import transforms, models
from PIL import Image
import pandas as pd

from config import (
    OUTPUT_DIR,
    MODEL_PATH,
    IMAGE_SIZE,
    NUM_CLASSES,
    CLASS_NAMES,
    IMAGENET_MEAN,
    IMAGENET_STD,
)
from utils import print_inference_header, print_model_summary, get_device


def get_inference_transform():
    """Get image transforms for inference."""
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


def predict_single_image(model, image_path: Path, transform, device) -> dict:
    """
    Run inference on a single image.

    Args:
        model: Trained PyTorch model
        image_path: Path to the image file
        transform: Image transforms
        device: Device to run inference on

    Returns:
        Dictionary with prediction results
    """
    # Load and preprocess image
    image = Image.open(image_path).convert("RGB")
    image_tensor = transform(image).unsqueeze(0).to(device)

    # Run inference
    with torch.no_grad():
        outputs = model(image_tensor)
        probabilities = torch.softmax(outputs, dim=1)
        predicted_class = outputs.argmax(dim=1).item()
        confidence = probabilities[0, predicted_class].item()

    # Get all class probabilities
    class_probs = {
        CLASS_NAMES[i]: probabilities[0, i].item()
        for i in range(NUM_CLASSES)
    }

    return {
        "image_path": str(image_path),
        "predicted_class": predicted_class,
        "predicted_label": CLASS_NAMES[predicted_class],
        "confidence": confidence,
        "class_probabilities": class_probs,
    }


def predict_batch(model, image_paths: list[Path], transform, device) -> list[dict]:
    """
    Run inference on a batch of images.

    Args:
        model: Trained PyTorch model
        image_paths: List of paths to image files
        transform: Image transforms
        device: Device to run inference on

    Returns:
        List of dictionaries with prediction results
    """
    results = []
    for image_path in image_paths:
        try:
            result = predict_single_image(model, image_path, transform, device)
            results.append(result)
        except Exception as e:
            print(f"Error processing {image_path}: {e}")
            results.append({
                "image_path": str(image_path),
                "error": str(e),
            })
    return results


def find_inference_images(data_dir: Path) -> list[Path]:
    """
    Find images for inference in the data directory.
    Looks for images in an 'inference' subdirectory.

    Args:
        data_dir: Base data directory

    Returns:
        List of image paths
    """
    inference_dir = data_dir / "inference"
    image_extensions = {".jpg", ".jpeg", ".png", ".webp"}

    if not inference_dir.exists():
        return []

    images = []
    for ext in image_extensions:
        images.extend(inference_dir.glob(f"*{ext}"))
        images.extend(inference_dir.glob(f"*{ext.upper()}"))

    return sorted(images)


def main():
    print_inference_header()

    # Check if model exists
    if not MODEL_PATH.exists():
        print(f"Error: Model checkpoint not found at {MODEL_PATH}")
        print("Run training first (02_train.py)")
        return

    # Setup device
    device = get_device()

    # Load model
    print("\nLoading model...")
    model = load_model(MODEL_PATH, device)

    # Get transforms
    transform = get_inference_transform()

    # Find inference images
    data_dir = Path("/data")
    inference_images = find_inference_images(data_dir)

    if not inference_images:
        print(f"\nNo inference images found in {data_dir / 'inference'}")
        print("To run inference on new images:")
        print(f"  1. Create a directory: {data_dir / 'inference'}")
        print("  2. Place your images in that directory")
        print("  3. Run this script again")

        # Demo inference using a test image from the manifest if available
        manifest_path = OUTPUT_DIR / "dataset_manifest.csv"
        if manifest_path.exists():
            print("\nRunning demo inference on sample test images...")
            df = pd.read_csv(manifest_path)
            test_df = df[df["split"] == "test"].head(5)

            if len(test_df) > 0:
                demo_images = [Path(p) for p in test_df["image_path"].tolist()]
                demo_images = [p for p in demo_images if p.exists()]

                if demo_images:
                    results = predict_batch(model, demo_images, transform, device)
                    print("\n[Demo Inference Results]")
                    print("-" * 60)
                    for result in results:
                        if "error" not in result:
                            print(f"\nImage: {Path(result['image_path']).name}")
                            print(f"  Predicted: {result['predicted_label']}")
                            print(f"  Confidence: {result['confidence']*100:.1f}%")
                            print("  All probabilities:")
                            for cls, prob in result['class_probabilities'].items():
                                print(f"    {cls}: {prob*100:.1f}%")
                        else:
                            print(f"\nImage: {result['image_path']}")
                            print(f"  Error: {result['error']}")
        return

    # Run inference on found images
    print(f"\nFound {len(inference_images)} images for inference")
    print("-" * 60)

    results = predict_batch(model, inference_images, transform, device)

    # Print results
    print("\n[Inference Results]")
    print("-" * 60)
    for result in results:
        if "error" not in result:
            print(f"\nImage: {Path(result['image_path']).name}")
            print(f"  Predicted: {result['predicted_label']}")
            print(f"  Confidence: {result['confidence']*100:.1f}%")
            print("  All probabilities:")
            for cls, prob in result['class_probabilities'].items():
                print(f"    {cls}: {prob*100:.1f}%")
        else:
            print(f"\nImage: {result['image_path']}")
            print(f"  Error: {result['error']}")

    # Save results to file
    inference_results_path = OUTPUT_DIR / "inference_results.json"
    with open(inference_results_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nInference results saved to {inference_results_path}")

    # Also save as CSV for easy viewing
    results_df = pd.DataFrame([
        {
            "image": Path(r.get("image_path", "")).name,
            "predicted_label": r.get("predicted_label", ""),
            "confidence": r.get("confidence", 0),
            "prob_Pronacio": r.get("class_probabilities", {}).get("Pronacio", 0),
            "prob_Neutralis": r.get("class_probabilities", {}).get("Neutralis", 0),
            "prob_Szupinacio": r.get("class_probabilities", {}).get("Szupinacio", 0),
        }
        for r in results if "error" not in r
    ])

    if len(results_df) > 0:
        csv_path = OUTPUT_DIR / "inference_results.csv"
        results_df.to_csv(csv_path, index=False)
        print(f"Inference results CSV saved to {csv_path}")

    print("\n" + "=" * 60)
    print("Inference completed successfully!")
    print("=" * 60)


if __name__ == "__main__":
    main()
