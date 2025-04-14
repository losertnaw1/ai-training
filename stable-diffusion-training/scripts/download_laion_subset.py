#!/usr/bin/env python
"""
Script to download a subset of the LAION-400M dataset.
"""
import os
import argparse
import logging
from pathlib import Path
from typing import Optional

import torch
from datasets import load_dataset
from tqdm import tqdm
import requests
from PIL import Image
from concurrent.futures import ThreadPoolExecutor, as_completed

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)

def download_image(url: str, save_path: Path) -> Optional[Path]:
    """
    Download an image from a URL and save it to disk.
    
    Args:
        url: URL of the image
        save_path: Path to save the image
        
    Returns:
        Path to the saved image if successful, None otherwise
    """
    try:
        response = requests.get(url, timeout=10, stream=True)
        response.raise_for_status()
        
        with open(save_path, "wb") as f:
            for chunk in response.iter_content(chunk_size=8192):
                f.write(chunk)
        
        # Verify the image can be opened
        Image.open(save_path).verify()
        
        return save_path
    except Exception as e:
        logger.warning(f"Failed to download image from {url}: {e}")
        if save_path.exists():
            save_path.unlink()  # Remove the file if it exists
        return None

def main():
    parser = argparse.ArgumentParser(description="Download a subset of the LAION-400M dataset")
    parser.add_argument(
        "--output_dir",
        type=str,
        default="data/laion400m",
        help="Directory to save the downloaded images",
    )
    parser.add_argument(
        "--num_samples",
        type=int,
        default=1000,
        help="Number of samples to download",
    )
    parser.add_argument(
        "--num_workers",
        type=int,
        default=8,
        help="Number of workers for parallel downloading",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for reproducibility",
    )
    parser.add_argument(
        "--min_width",
        type=int,
        default=256,
        help="Minimum image width",
    )
    parser.add_argument(
        "--min_height",
        type=int,
        default=256,
        help="Minimum image height",
    )
    parser.add_argument(
        "--aesthetic_score",
        type=float,
        default=5.0,
        help="Minimum aesthetic score",
    )
    
    args = parser.parse_args()
    
    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Create subdirectories
    images_dir = output_dir / "images"
    images_dir.mkdir(exist_ok=True)
    
    # Load dataset
    logger.info("Loading LAION-400M dataset")
    dataset = load_dataset(
        "laion/laion400m",
        "laion400m-data",
        split="train",
        streaming=True,
    )
    
    # Set random seed for reproducibility
    torch.manual_seed(args.seed)
    
    # Filter dataset
    logger.info("Filtering dataset")
    filtered_dataset = dataset.filter(
        lambda example: (
            example.get("width", 0) >= args.min_width
            and example.get("height", 0) >= args.min_height
            and example.get("aesthetic_score", 0) >= args.aesthetic_score
        )
    )
    
    # Take a subset
    subset = list(filtered_dataset.take(args.num_samples))
    
    # Prepare metadata
    metadata = []
    
    # Download images in parallel
    logger.info(f"Downloading {len(subset)} images with {args.num_workers} workers")
    with ThreadPoolExecutor(max_workers=args.num_workers) as executor:
        futures = []
        
        for i, example in enumerate(subset):
            url = example["url"]
            image_id = f"{i:08d}"
            save_path = images_dir / f"{image_id}.jpg"
            
            future = executor.submit(download_image, url, save_path)
            futures.append((future, example, image_id))
        
        for future, example, image_id in tqdm(as_completed(futures), total=len(futures)):
            result = future.result()
            if result is not None:
                metadata.append({
                    "image_id": image_id,
                    "image_path": str(result),
                    "caption": example["caption"],
                    "url": example["url"],
                    "width": example.get("width"),
                    "height": example.get("height"),
                    "aesthetic_score": example.get("aesthetic_score"),
                })
    
    # Save metadata
    logger.info(f"Successfully downloaded {len(metadata)} images")
    
    # Save metadata to disk
    import json
    with open(output_dir / "metadata.json", "w") as f:
        json.dump(metadata, f, indent=2)
    
    logger.info(f"Metadata saved to {output_dir / 'metadata.json'}")

if __name__ == "__main__":
    main()
