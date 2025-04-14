"""
LAION-400M dataset handling module.
"""
import os
import io
import logging
import time
import requests
from typing import Dict, List, Optional, Tuple, Union
from functools import lru_cache

import torch
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import numpy as np
from datasets import load_dataset
from transformers import CLIPTokenizer
from huggingface_hub import login
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

logger = logging.getLogger(__name__)

# Create a session with retry logic
def create_session_with_retry():
    """
    Create a requests session with retry logic.

    Returns:
        requests.Session: Session with retry logic
    """
    session = requests.Session()
    retries = Retry(
        total=3,  # Total number of retries
        backoff_factor=0.5,  # Backoff factor for retries
        status_forcelist=[429, 500, 502, 503, 504],  # HTTP status codes to retry on
        allowed_methods=["GET"],  # Only retry on GET requests
    )
    adapter = HTTPAdapter(max_retries=retries)
    session.mount("http://", adapter)
    session.mount("https://", adapter)
    session.headers.update({"User-Agent": "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36"})
    return session

# Create a cached session
@lru_cache(maxsize=1)
def get_session():
    return create_session_with_retry()

class LAION400MDataset(Dataset):
    """
    Dataset class for LAION-400M dataset.

    This dataset loads image-text pairs from the LAION-400M dataset
    and prepares them for training Stable Diffusion models.
    """

    def __init__(
        self,
        dataset_name: str = "laion/laion400m",
        subset_name: str = "default",  # Updated to use 'default' config
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
        Initialize the LAION-400M dataset.

        Args:
            dataset_name: Name of the dataset on Hugging Face Hub
            subset_name: Name of the subset
            split: Dataset split (train, validation, test)
            max_samples: Maximum number of samples to load
            resolution: Image resolution
            center_crop: Whether to center crop images
            tokenizer_name: Name of the tokenizer to use
            max_token_length: Maximum token length
            cache_dir: Directory to cache the dataset
        """
        self.dataset_name = dataset_name
        self.subset_name = subset_name
        self.split = split
        self.max_samples = max_samples
        self.resolution = resolution
        self.center_crop = center_crop
        self.max_token_length = max_token_length

        # Login to Hugging Face Hub if token is provided
        if hf_token is not None:
            logger.info("Logging in to Hugging Face Hub")
            login(token=hf_token)

        # Load tokenizer
        self.tokenizer = CLIPTokenizer.from_pretrained(tokenizer_name)

        # Load dataset
        logger.info(f"Loading dataset {dataset_name}/{subset_name} (split: {split})")
        try:
            self.dataset = load_dataset(
                dataset_name,
                subset_name,
                split=split,
                cache_dir=cache_dir,
                streaming=False,  # Disable streaming to avoid worker issues
                trust_remote_code=True,  # Allow remote code execution
            )
            logger.info(f"Dataset loaded successfully with {len(self.dataset)} samples")

            # Filter out samples with known problematic domains
            def is_valid_sample(sample):
                # List of problematic domains to filter out
                problematic_domains = [
                    'cdn3.volusion.com', 'cdn-s3-3.wanelo.com', 'cloudfront.net',
                    'blogcdn.com', 'stylebistro.com', 'dear-lover.com',
                    'wanelo.com', 'cinewsnow.com', 'msccruises.com',
                    'icvalledeilaghi.it', 'exnord-stephan.de'
                ]

                # Check if URL exists and doesn't contain problematic domains
                if 'url' in sample:
                    url = sample['url'].lower()
                    return not any(domain in url for domain in problematic_domains)
                return True

            # Apply the filter
            original_size = len(self.dataset)
            self.dataset = self.dataset.filter(is_valid_sample)
            filtered_size = len(self.dataset)
            logger.info(f"Filtered out {original_size - filtered_size} problematic samples")

        except Exception as e:
            logger.error(f"Error loading dataset: {e}")
            raise

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
        try:
            sample = self.dataset[idx]

            # Process image
            try:
                # Check if the sample has URL or image_path
                if "image" in sample and sample["image"] is not None:
                    # If image is already a PIL Image
                    if isinstance(sample["image"], Image.Image):
                        image = sample["image"].convert("RGB")
                    else:
                        # If image is bytes or something else
                        try:
                            image = Image.open(io.BytesIO(sample["image"])).convert("RGB")
                        except Exception as e:
                            logger.debug(f"Failed to open image bytes: {e}")
                            raise ValueError(f"Invalid image data: {e}")
                elif "image_path" in sample:
                    try:
                        image = Image.open(sample["image_path"]).convert("RGB")
                    except Exception as e:
                        logger.debug(f"Failed to open image path {sample['image_path']}: {e}")
                        raise ValueError(f"Invalid image path: {e}")
                elif "url" in sample:
                    # Download image from URL with retry logic
                    try:
                        # Get session with retry logic
                        session = get_session()
                        # Use a shorter timeout to avoid long waits
                        response = session.get(sample["url"], timeout=5)
                        response.raise_for_status()  # Raise exception for 4XX/5XX responses
                        image = Image.open(io.BytesIO(response.content)).convert("RGB")
                    except requests.exceptions.RequestException as e:
                        logger.debug(f"Failed to download image from {sample['url']}: {e}")
                        raise ValueError(f"Failed to download image: {e}")
                    except Exception as e:
                        logger.debug(f"Failed to process downloaded image: {e}")
                        raise ValueError(f"Invalid downloaded image: {e}")
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
                # Check different possible caption fields
                if "caption" in sample:
                    text = sample["caption"]
                elif "text" in sample:
                    text = sample["text"]
                elif "captions" in sample and isinstance(sample["captions"], list) and len(sample["captions"]) > 0:
                    text = sample["captions"][0]
                else:
                    text = "No caption available"

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
                text = "No caption available"

            return {
                "pixel_values": image,
                "input_ids": input_ids,
                "attention_mask": attention_mask,
                "text": text,
            }

        except Exception as e:
            logger.error(f"Error processing sample at index {idx}: {e}")
            # Return a completely placeholder sample
            return {
                "pixel_values": torch.zeros((3, self.resolution, self.resolution)),
                "input_ids": torch.zeros(self.max_token_length, dtype=torch.long),
                "attention_mask": torch.zeros(self.max_token_length, dtype=torch.long),
                "text": "Error loading sample",
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
    dataset: LAION400MDataset,
    batch_size: int = 16,
    num_workers: int = 0,  # Reduced to 0 to avoid multiprocessing issues
    shuffle: bool = True,
) -> DataLoader:
    """
    Create a DataLoader for the LAION-400M dataset.

    Args:
        dataset: LAION-400M dataset
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
        persistent_workers=False if num_workers == 0 else True,
        prefetch_factor=2 if num_workers > 0 else None,
    )
