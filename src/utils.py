"""
utils.py
Utility functions and logging configuration for AnkleAlign project.
"""

import logging
import sys
from datetime import datetime

import torch
import torch.nn as nn


def setup_logging(name: str = "AnkleAlign") -> logging.Logger:
    """
    Configure logging to output to stdout (which Docker captures).

    Args:
        name: Logger name

    Returns:
        Configured logger instance
    """
    logger = logging.getLogger(name)
    logger.setLevel(logging.INFO)

    # Remove existing handlers to avoid duplicates
    logger.handlers.clear()

    # Create stdout handler
    handler = logging.StreamHandler(sys.stdout)
    handler.setLevel(logging.INFO)

    # Create formatter
    formatter = logging.Formatter(
        fmt='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    handler.setFormatter(formatter)

    # Add handler to logger
    logger.addHandler(handler)

    return logger


def count_parameters(model: nn.Module) -> tuple[int, int]:
    """
    Count the number of trainable and non-trainable parameters in a model.

    Args:
        model: PyTorch model

    Returns:
        Tuple of (trainable_params, non_trainable_params)
    """
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    non_trainable = sum(p.numel() for p in model.parameters() if not p.requires_grad)
    return trainable, non_trainable


def print_model_summary(model: nn.Module, model_name: str = "Model"):
    """
    Print a summary of the model architecture and parameter counts.

    Args:
        model: PyTorch model
        model_name: Name to display in the summary
    """
    trainable, non_trainable = count_parameters(model)
    total = trainable + non_trainable

    print("\n" + "=" * 60)
    print("MODEL ARCHITECTURE SUMMARY")
    print("=" * 60)
    print(f"\nModel: {model_name}")
    print(f"\n[Architecture]")
    print(model)
    print(f"\n[Parameter Counts]")
    print(f"  Trainable parameters:     {trainable:,}")
    print(f"  Non-trainable parameters: {non_trainable:,}")
    print(f"  Total parameters:         {total:,}")
    print(f"\n  Model size (approx):      {total * 4 / (1024 * 1024):.2f} MB (float32)")
    print("=" * 60 + "\n")


def get_device() -> torch.device:
    """
    Get the best available device (CUDA if available, else CPU).

    Returns:
        torch.device
    """
    if torch.cuda.is_available():
        device = torch.device("cuda")
        print(f"\n[Device Information]")
        print(f"  Using: CUDA (GPU)")
        print(f"  GPU Name: {torch.cuda.get_device_name(0)}")
        print(f"  GPU Memory: {torch.cuda.get_device_properties(0).total_memory / (1024**3):.1f} GB")
    else:
        device = torch.device("cpu")
        print(f"\n[Device Information]")
        print(f"  Using: CPU")

    return device


def print_data_summary(train_size: int, val_size: int, test_size: int, class_distribution: dict = None):
    """
    Print a summary of the dataset splits.

    Args:
        train_size: Number of training samples
        val_size: Number of validation samples
        test_size: Number of test samples
        class_distribution: Optional dict with class distribution per split
    """
    total = train_size + val_size + test_size

    print("\n" + "=" * 60)
    print("DATA SUMMARY")
    print("=" * 60)
    print(f"\n[Dataset Splits]")
    print(f"  Training samples:   {train_size:,} ({train_size/total*100:.1f}%)")
    print(f"  Validation samples: {val_size:,} ({val_size/total*100:.1f}%)")
    print(f"  Test samples:       {test_size:,} ({test_size/total*100:.1f}%)")
    print(f"  Total samples:      {total:,}")

    if class_distribution:
        print(f"\n[Class Distribution]")
        for split, dist in class_distribution.items():
            print(f"\n  {split.capitalize()}:")
            for class_name, count in dist.items():
                print(f"    {class_name}: {count}")

    print("=" * 60 + "\n")


def print_training_header():
    """Print header for training section."""
    print("\n" + "=" * 60)
    print("TRAINING")
    print("=" * 60)


def print_evaluation_header():
    """Print header for evaluation section."""
    print("\n" + "=" * 60)
    print("EVALUATION")
    print("=" * 60)


def print_inference_header():
    """Print header for inference section."""
    print("\n" + "=" * 60)
    print("INFERENCE")
    print("=" * 60)


def format_time(seconds: float) -> str:
    """
    Format seconds into a human-readable string.

    Args:
        seconds: Time in seconds

    Returns:
        Formatted time string
    """
    if seconds < 60:
        return f"{seconds:.1f}s"
    elif seconds < 3600:
        minutes = int(seconds // 60)
        secs = seconds % 60
        return f"{minutes}m {secs:.1f}s"
    else:
        hours = int(seconds // 3600)
        minutes = int((seconds % 3600) // 60)
        return f"{hours}h {minutes}m"
