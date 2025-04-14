#!/usr/bin/env python
"""
Main training script for fine-tuning Stable Diffusion models.
"""
import os
import argparse
import logging
import sys
from pathlib import Path

import torch
import numpy as np
from accelerate import Accelerator
from accelerate.utils import set_seed

# Add the current directory to the Python path
sys.path.insert(0, os.path.abspath(os.path.dirname(__file__)))
from src.data.laion_dataset import LAION400MDataset, create_dataloader
from src.models.stable_diffusion import StableDiffusionModel
from src.training.trainer import StableDiffusionTrainer

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)

def parse_args():
    parser = argparse.ArgumentParser(description="Fine-tune Stable Diffusion on LAION-400M")

    # Data arguments
    parser.add_argument(
        "--data_dir",
        type=str,
        default="data/coco",
        help="Directory to cache the dataset",
    )
    parser.add_argument(
        "--resolution",
        type=int,
        default=512,
        help="Resolution of the images",
    )
    parser.add_argument(
        "--center_crop",
        action="store_true",
        help="Whether to center crop images",
    )
    parser.add_argument(
        "--random_flip",
        action="store_true",
        help="Whether to randomly flip images horizontally",
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
        "--hf_token",
        type=str,
        default=None,
        help="Hugging Face API token for accessing gated datasets",
    )

    # Model arguments
    parser.add_argument(
        "--pretrained_model_name_or_path",
        type=str,
        default="CompVis/stable-diffusion-v1-4",
        help="Path to pretrained model or model identifier from huggingface.co/models",
    )
    parser.add_argument(
        "--revision",
        type=str,
        default=None,
        help="Revision of pretrained model identifier from huggingface.co/models",
    )
    parser.add_argument(
        "--train_text_encoder",
        action="store_true",
        help="Whether to train the text encoder",
    )

    # Training arguments
    parser.add_argument(
        "--output_dir",
        type=str,
        default="models/stable-diffusion-finetuned",
        help="Directory to save model checkpoints",
    )
    parser.add_argument(
        "--logging_dir",
        type=str,
        default="logs",
        help="Directory to save logs",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for reproducibility",
    )
    parser.add_argument(
        "--mixed_precision",
        type=str,
        default="fp16",
        choices=["no", "fp16", "bf16"],
        help="Mixed precision training",
    )
    parser.add_argument(
        "--train_batch_size",
        type=int,
        default=4,
        help="Batch size for training",
    )
    parser.add_argument(
        "--val_batch_size",
        type=int,
        default=4,
        help="Batch size for validation",
    )
    parser.add_argument(
        "--num_train_epochs",
        type=int,
        default=100,
        help="Number of training epochs",
    )
    parser.add_argument(
        "--max_train_steps",
        type=int,
        default=None,
        help="Maximum number of training steps",
    )
    parser.add_argument(
        "--gradient_accumulation_steps",
        type=int,
        default=1,
        help="Number of gradient accumulation steps",
    )
    parser.add_argument(
        "--learning_rate",
        type=float,
        default=1e-5,
        help="Learning rate",
    )
    parser.add_argument(
        "--lr_scheduler",
        type=str,
        default="cosine",
        choices=["linear", "cosine", "cosine_with_restarts", "polynomial", "constant", "constant_with_warmup"],
        help="Learning rate scheduler",
    )
    parser.add_argument(
        "--lr_warmup_steps",
        type=int,
        default=500,
        help="Number of learning rate warmup steps",
    )
    parser.add_argument(
        "--adam_beta1",
        type=float,
        default=0.9,
        help="Adam beta1",
    )
    parser.add_argument(
        "--adam_beta2",
        type=float,
        default=0.999,
        help="Adam beta2",
    )
    parser.add_argument(
        "--adam_weight_decay",
        type=float,
        default=1e-2,
        help="Adam weight decay",
    )
    parser.add_argument(
        "--adam_epsilon",
        type=float,
        default=1e-8,
        help="Adam epsilon",
    )
    parser.add_argument(
        "--max_grad_norm",
        type=float,
        default=1.0,
        help="Maximum gradient norm",
    )
    parser.add_argument(
        "--logging_steps",
        type=int,
        default=10,
        help="Number of steps between logging",
    )
    parser.add_argument(
        "--save_steps",
        type=int,
        default=500,
        help="Number of steps between saving checkpoints",
    )
    parser.add_argument(
        "--resume_from_checkpoint",
        type=str,
        default=None,
        help="Path to checkpoint to resume from",
    )

    # Weights & Biases arguments
    parser.add_argument(
        "--use_wandb",
        action="store_true",
        help="Whether to use Weights & Biases for logging",
    )
    parser.add_argument(
        "--wandb_project",
        type=str,
        default="stable-diffusion-finetuning",
        help="Weights & Biases project name",
    )
    parser.add_argument(
        "--wandb_entity",
        type=str,
        default=None,
        help="Weights & Biases entity name",
    )

    return parser.parse_args()

def main():
    args = parse_args()

    # Set random seed
    set_seed(args.seed)

    # Create output and logging directories
    os.makedirs(args.output_dir, exist_ok=True)
    os.makedirs(args.logging_dir, exist_ok=True)

    # Load dataset
    logger.info("Loading dataset")
    train_dataset = LAION400MDataset(
        dataset_name="laion/laion400m",
        subset_name="default",  # Updated to use 'default' config
        split="train",
        resolution=args.resolution,
        center_crop=args.center_crop,
        max_samples=args.max_train_samples if hasattr(args, 'max_train_samples') else None,
        hf_token=args.hf_token,
    )

    # For validation, use a subset of the training data
    val_dataset = LAION400MDataset(
        dataset_name="laion/laion400m",
        subset_name="default",  # Updated to use 'default' config
        split="train",  # Using train split for validation too since LAION doesn't have a validation split
        resolution=args.resolution,
        center_crop=args.center_crop,
        max_samples=args.max_val_samples if hasattr(args, 'max_val_samples') else None,
        hf_token=args.hf_token,
    )

    # Create dataloaders
    train_dataloader = create_dataloader(
        train_dataset,
        batch_size=args.train_batch_size,
        shuffle=True,
    )

    val_dataloader = create_dataloader(
        val_dataset,
        batch_size=args.val_batch_size,
        shuffle=False,
    )

    # Calculate total training steps if not provided
    if args.max_train_steps is None:
        args.max_train_steps = args.num_train_epochs * len(train_dataloader) // args.gradient_accumulation_steps

    logger.info(f"Total training steps: {args.max_train_steps}")

    # Load model
    logger.info("Loading model")
    # Always use float32 for model initialization to avoid data type mismatches
    model = StableDiffusionModel(
        pretrained_model_name_or_path=args.pretrained_model_name_or_path,
        revision=args.revision,
        torch_dtype=torch.float32,  # Always use float32 for model initialization
    )

    logger.info(f"Model initialized with dtype: {model.torch_dtype}")

    # Prepare optimizer
    optimizer = model.prepare_for_training(
        learning_rate=args.learning_rate,
        adam_beta1=args.adam_beta1,
        adam_beta2=args.adam_beta2,
        adam_weight_decay=args.adam_weight_decay,
        adam_epsilon=args.adam_epsilon,
        train_text_encoder=args.train_text_encoder,
    )

    # Create accelerator
    logger.info(f"Setting up accelerator with mixed_precision: {args.mixed_precision}")
    accelerator = Accelerator(
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        mixed_precision=args.mixed_precision if args.mixed_precision != "no" else None,
    )

    # Log device information
    logger.info(f"Accelerator device: {accelerator.device}")
    logger.info(f"Accelerator mixed precision: {accelerator.mixed_precision}")

    # Prepare model, optimizer, and dataloaders
    model.unet, optimizer, train_dataloader, val_dataloader = accelerator.prepare(
        model.unet, optimizer, train_dataloader, val_dataloader
    )

    # Create learning rate scheduler
    from transformers import get_scheduler
    lr_scheduler = get_scheduler(
        name=args.lr_scheduler,
        optimizer=optimizer,
        num_warmup_steps=args.lr_warmup_steps,
        num_training_steps=args.max_train_steps,
    )
    lr_scheduler = accelerator.prepare(lr_scheduler)

    # Create trainer
    trainer = StableDiffusionTrainer(
        model=model,
        train_dataloader=train_dataloader,
        val_dataloader=val_dataloader,
        lr_scheduler=lr_scheduler,
        num_train_epochs=args.num_train_epochs,
        max_train_steps=args.max_train_steps,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        mixed_precision=args.mixed_precision if args.mixed_precision != "no" else None,
        output_dir=args.output_dir,
        logging_dir=args.logging_dir,
        logging_steps=args.logging_steps,
        save_steps=args.save_steps,
        resume_from_checkpoint=args.resume_from_checkpoint,
        use_wandb=args.use_wandb,
        wandb_project=args.wandb_project,
        wandb_entity=args.wandb_entity,
    )

    # Train model
    trainer.train()

if __name__ == "__main__":
    main()
