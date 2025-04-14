"""
COCO dataset handling module.
"""
import os
import logging
import io
from typing import Dict, List, Optional, Tuple, Union

import torch
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import numpy as np
from datasets import load_dataset
from transformers import CLIPTokenizer
import requests
from huggingface_hub import login

logger = logging.getLogger(__name__)

class COCODataset(Dataset):
    """
    Dataset class for COCO dataset.

    This dataset loads image-text pairs from the COCO dataset
    and prepares them for training Stable Diffusion models.
    """

    def __init__(
        self,
        dataset_name: str = "lambdalabs/pokemon-blip-captions",
        split: str = "train",
        max_samples: Optional[int] = None,
        resolution: int = 512,
        center_crop: bool = True,
        tokenizer_name: str = "openai/clip-vit-large-patch14",
        max_token_length: int = 77,
        cache_dir: Optional[str] = None,
        hf_token: Optional[str] = None,
    ):
        """
        Initialize the COCO dataset.

        Args:
            dataset_name: Name of the dataset on Hugging Face Hub
            split: Dataset split (train, validation, test)
            max_samples: Maximum number of samples to load
            resolution: Image resolution
            center_crop: Whether to center crop images
            tokenizer_name: Name of the tokenizer to use
            max_token_length: Maximum token length
            cache_dir: Directory to cache the dataset
        """
        self.dataset_name = dataset_name
        self.split = split
        self.max_samples = max_samples
        self.resolution = resolution
        self.center_crop = center_crop
        self.max_token_length = max_token_length

        # Load tokenizer
        self.tokenizer = CLIPTokenizer.from_pretrained(tokenizer_name)

        # Login to Hugging Face Hub if token is provided
        if hf_token is not None:
            logger.info("Logging in to Hugging Face Hub")
            login(token=hf_token)

        # Load dataset
        logger.info(f"Loading dataset {dataset_name} (split: {split})")
        self.dataset = load_dataset(
            dataset_name,
            split=split,
            cache_dir=cache_dir,
        )

        if max_samples is not None and max_samples < len(self.dataset):
            logger.info(f"Using {max_samples} samples out of {len(self.dataset)}")
            self.dataset = self.dataset.select(range(max_samples))

        logger.info(f"Dataset loaded with {len(self.dataset)} samples")

    def __len__(self) -> int:
        """Return the number of samples in the dataset."""
        return len(self.dataset)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        """
        Get a sample from the dataset.

        Args:
            idx: Index of the sample

        Returns:
            Dictionary containing the image and text tokens
        """
        sample = self.dataset[idx]

        # Process image
        try:
            # COCO dataset provides image as PIL Image or as bytes
            if "image" in sample and sample["image"] is not None:
                if isinstance(sample["image"], Image.Image):
                    image = sample["image"].convert("RGB")
                else:
                    image = Image.open(io.BytesIO(sample["image"])).convert("RGB")
            elif "image_path" in sample:
                image = Image.open(sample["image_path"]).convert("RGB")
            elif "url" in sample:
                # Download image from URL
                response = requests.get(sample["url"], timeout=10)
                image = Image.open(io.BytesIO(response.content)).convert("RGB")
            else:
                raise ValueError("No image data found in sample")

            # Resize and crop
            if self.center_crop:
                image = self._center_crop_and_resize(image, self.resolution)
            else:
                image = image.resize((self.resolution, self.resolution))

            # Convert to tensor and normalize
            image = np.array(image).astype(np.float32) / 255.0
            image = image.transpose(2, 0, 1)  # HWC -> CHW
            image = torch.from_numpy(image)

            # Normalize to [-1, 1]
            image = 2.0 * image - 1.0

        except Exception as e:
            logger.warning(f"Error processing image at index {idx}: {e}")
            # Return a placeholder image
            image = torch.zeros((3, self.resolution, self.resolution))

        # Process text
        try:
            # COCO dataset has captions in different formats
            if "caption" in sample:
                text = sample["caption"]
            elif "text" in sample:
                text = sample["text"]
            elif "captions" in sample and isinstance(sample["captions"], list) and len(sample["captions"]) > 0:
                # Take the first caption if multiple are available
                text = sample["captions"][0]
            else:
                raise ValueError("No caption found in sample")

            tokens = self.tokenizer(
                text,
                padding="max_length",
                max_length=self.max_token_length,
                truncation=True,
                return_tensors="pt",
            )
            input_ids = tokens.input_ids[0]
            attention_mask = tokens.attention_mask[0]

        except Exception as e:
            logger.warning(f"Error processing text at index {idx}: {e}")
            # Return placeholder tokens
            input_ids = torch.zeros(self.max_token_length, dtype=torch.long)
            attention_mask = torch.zeros(self.max_token_length, dtype=torch.long)
            text = ""

        return {
            "pixel_values": image,
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "text": text,
        }

    def _center_crop_and_resize(self, image: Image.Image, resolution: int) -> Image.Image:
        """
        Center crop and resize an image.

        Args:
            image: Input image
            resolution: Target resolution

        Returns:
            Processed image
        """
        width, height = image.size

        # Calculate dimensions for center crop
        if width > height:
            left = (width - height) // 2
            right = left + height
            top, bottom = 0, height
        else:
            top = (height - width) // 2
            bottom = top + width
            left, right = 0, width

        # Perform center crop
        image = image.crop((left, top, right, bottom))

        # Resize to target resolution
        image = image.resize((resolution, resolution))

        return image


def create_dataloader(
    dataset: COCODataset,
    batch_size: int = 16,
    num_workers: int = 4,
    shuffle: bool = True,
) -> DataLoader:
    """
    Create a DataLoader for the COCO dataset.

    Args:
        dataset: COCO dataset
        batch_size: Batch size
        num_workers: Number of workers for data loading
        shuffle: Whether to shuffle the dataset

    Returns:
        DataLoader for the dataset
    """
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=True,
    )
