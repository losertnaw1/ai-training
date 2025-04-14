#!/usr/bin/env python
"""
Script to run the entire Stable Diffusion training pipeline.
"""
import os
import argparse
import logging
import subprocess
from pathlib import Path

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)

def parse_args():
    parser = argparse.ArgumentParser(description="Run the entire Stable Diffusion training pipeline")

    parser.add_argument(
        "--config",
        type=str,
        default="configs/default_config.yaml",
        help="Path to the configuration file",
    )
    parser.add_argument(
        "--data_dir",
        type=str,
        default="data/coco",
        help="Directory to cache the dataset",
    )
    parser.add_argument(
        "--max_train_samples",
        type=int,
        default=10000,
        help="Maximum number of training samples",
    )
    parser.add_argument(
        "--max_val_samples",
        type=int,
        default=1000,
        help="Maximum number of validation samples",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="models/stable-diffusion-finetuned",
        help="Directory to save model checkpoints",
    )
    parser.add_argument(
        "--skip_download",
        action="store_true",
        help="Skip dataset download",
    )
    parser.add_argument(
        "--skip_training",
        action="store_true",
        help="Skip model training",
    )
    parser.add_argument(
        "--skip_inference",
        action="store_true",
        help="Skip image generation",
    )
    parser.add_argument(
        "--prompt",
        type=str,
        default="a photo of a cat",
        help="Prompt for image generation",
    )

    return parser.parse_args()

def run_command(command):
    """
    Run a shell command and log the output.

    Args:
        command: Shell command to run
    """
    logger.info(f"Running command: {command}")
    process = subprocess.Popen(
        command,
        shell=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        universal_newlines=True,
    )

    for line in process.stdout:
        logger.info(line.strip())

    process.wait()

    if process.returncode != 0:
        logger.error(f"Command failed with return code {process.returncode}")
        raise subprocess.CalledProcessError(process.returncode, command)

def main():
    args = parse_args()

    # Create directories
    os.makedirs(args.data_dir, exist_ok=True)
    os.makedirs(args.output_dir, exist_ok=True)

    # Step 1: We're using the COCO dataset which will be downloaded automatically
    # during training, so we don't need a separate download step
    logger.info("Step 1: Using COCO dataset (will be downloaded during training)")

    # Step 2: Train model
    if not args.skip_training:
        logger.info("Step 2: Training model")
        train_command = (
            f"python train.py "
            f"--config {args.config} "
            f"--output_dir {args.output_dir} "
            f"--max_train_samples {args.max_train_samples} "
            f"--max_val_samples {args.max_val_samples}"
        )
        run_command(train_command)
    else:
        logger.info("Skipping model training")

    # Step 3: Generate images
    if not args.skip_inference:
        logger.info("Step 3: Generating images")
        inference_command = (
            f"python inference.py "
            f"--model_path {args.output_dir}/checkpoint-final "
            f"--prompt \"{args.prompt}\" "
            f"--num_images 4"
        )
        run_command(inference_command)
    else:
        logger.info("Skipping image generation")

    logger.info("Pipeline completed successfully")

if __name__ == "__main__":
    main()
