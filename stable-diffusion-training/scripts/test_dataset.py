#!/usr/bin/env python
"""
Script to test the dataset loading.
"""
import os
import argparse
import logging
import sys
from pathlib import Path
import matplotlib.pyplot as plt
import torch
import numpy as np

# Add the parent directory to the Python path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from src.data.laion_dataset import LAION400MDataset

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)

def parse_args():
    parser = argparse.ArgumentParser(description="Test dataset loading")

    parser.add_argument(
        "--dataset_name",
        type=str,
        default="laion/laion400m",
        help="Name of the dataset on Hugging Face Hub",
    )
    parser.add_argument(
        "--subset_name",
        type=str,
        default="default",
        help="Name of the subset",
    )
    parser.add_argument(
        "--hf_token",
        type=str,
        required=True,
        help="Hugging Face API token for accessing gated datasets",
    )
    parser.add_argument(
        "--split",
        type=str,
        default="train",
        help="Dataset split (train, validation, test)",
    )
    parser.add_argument(
        "--num_samples",
        type=int,
        default=5,
        help="Number of samples to display",
    )
    parser.add_argument(
        "--resolution",
        type=int,
        default=512,
        help="Image resolution",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="outputs/dataset_test",
        help="Directory to save sample images",
    )

    return parser.parse_args()

def main():
    args = parse_args()

    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)

    # Load dataset
    logger.info(f"Loading dataset {args.dataset_name}/{args.subset_name} (split: {args.split})")
    dataset = LAION400MDataset(
        dataset_name=args.dataset_name,
        subset_name=args.subset_name,
        split=args.split,
        max_samples=args.num_samples,
        resolution=args.resolution,
        hf_token=args.hf_token,
    )

    logger.info(f"Dataset loaded with {len(dataset)} samples")

    # Display samples
    for i in range(min(args.num_samples, len(dataset))):
        sample = dataset[i]

        # Convert image tensor to numpy array
        image = sample["pixel_values"]
        image = (image + 1.0) / 2.0  # Convert from [-1, 1] to [0, 1]
        image = image.permute(1, 2, 0).numpy()  # CHW -> HWC
        image = np.clip(image, 0, 1)

        # Get caption
        caption = sample["text"]

        # Display image and caption
        plt.figure(figsize=(10, 8))
        plt.imshow(image)
        plt.title(caption)
        plt.axis("off")

        # Save image
        output_path = os.path.join(args.output_dir, f"sample_{i:02d}.png")
        plt.savefig(output_path)
        logger.info(f"Sample {i} saved to {output_path}")

        plt.close()

    logger.info(f"All samples saved to {args.output_dir}")

if __name__ == "__main__":
    main()
