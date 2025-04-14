#!/usr/bin/env python
"""
Inference script for generating images with a fine-tuned Stable Diffusion model.
"""
import os
import argparse
import logging
import sys
from pathlib import Path

import torch
from diffusers import StableDiffusionPipeline
from PIL import Image

# Add the current directory to the Python path
sys.path.insert(0, os.path.abspath(os.path.dirname(__file__)))

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)

def parse_args():
    parser = argparse.ArgumentParser(description="Generate images with a fine-tuned Stable Diffusion model")

    parser.add_argument(
        "--model_path",
        type=str,
        required=True,
        help="Path to the fine-tuned model",
    )
    parser.add_argument(
        "--prompt",
        type=str,
        required=True,
        help="Text prompt for image generation",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="outputs",
        help="Directory to save generated images",
    )
    parser.add_argument(
        "--num_images",
        type=int,
        default=4,
        help="Number of images to generate",
    )
    parser.add_argument(
        "--num_inference_steps",
        type=int,
        default=50,
        help="Number of inference steps",
    )
    parser.add_argument(
        "--guidance_scale",
        type=float,
        default=7.5,
        help="Guidance scale",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="Random seed for reproducibility",
    )
    parser.add_argument(
        "--height",
        type=int,
        default=512,
        help="Height of generated images",
    )
    parser.add_argument(
        "--width",
        type=int,
        default=512,
        help="Width of generated images",
    )

    return parser.parse_args()

def main():
    args = parse_args()

    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)

    # Set random seed
    if args.seed is not None:
        torch.manual_seed(args.seed)
        torch.cuda.manual_seed_all(args.seed)

    # Load model
    logger.info(f"Loading model from {args.model_path}")
    pipeline = StableDiffusionPipeline.from_pretrained(
        args.model_path,
        torch_dtype=torch.float16,
        safety_checker=None,
    )
    pipeline = pipeline.to("cuda")

    # Enable memory optimization
    pipeline.enable_attention_slicing()

    # Generate images
    logger.info(f"Generating {args.num_images} images with prompt: {args.prompt}")

    for i in range(args.num_images):
        # Generate image
        with torch.autocast("cuda"):
            image = pipeline(
                prompt=args.prompt,
                height=args.height,
                width=args.width,
                num_inference_steps=args.num_inference_steps,
                guidance_scale=args.guidance_scale,
            ).images[0]

        # Save image
        output_path = os.path.join(args.output_dir, f"image_{i:04d}.png")
        image.save(output_path)
        logger.info(f"Image saved to {output_path}")

if __name__ == "__main__":
    main()
