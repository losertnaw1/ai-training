#!/usr/bin/env python
"""
Script to run the training script with the correct Python path.
"""
import os
import sys
import subprocess
import argparse

def main():
    parser = argparse.ArgumentParser(description="Run the training script")
    parser.add_argument(
        "--token",
        type=str,
        required=True,
        help="Hugging Face API token",
    )
    parser.add_argument(
        "--max_train_samples",
        type=int,
        default=5000,  # Reduced to 5000 for better stability
        help="Maximum number of training samples from LAION-400M",
    )
    parser.add_argument(
        "--max_val_samples",
        type=int,
        default=500,  # Reduced to 500 for better stability
        help="Maximum number of validation samples",
    )
    parser.add_argument(
        "--train_batch_size",
        type=int,
        default=2,
        help="Training batch size",
    )
    parser.add_argument(
        "--val_batch_size",
        type=int,
        default=2,
        help="Validation batch size",
    )
    parser.add_argument(
        "--num_train_epochs",
        type=int,
        default=1,
        help="Number of training epochs",
    )
    parser.add_argument(
        "--mixed_precision",
        type=str,
        default="no",  # Changed to 'no' to avoid data type mismatches
        choices=["no", "fp16", "bf16"],
        help="Mixed precision training (use 'no' for better compatibility)",
    )
    parser.add_argument(
        "--gradient_accumulation_steps",
        type=int,
        default=4,
        help="Number of gradient accumulation steps",
    )
    parser.add_argument(
        "--lr_warmup_steps",
        type=int,
        default=500,
        help="Number of learning rate warmup steps",
    )
    args = parser.parse_args()

    # Add the current directory to the Python path
    current_dir = os.path.abspath(os.path.dirname(__file__))
    sys.path.insert(0, current_dir)

    # Run the training script
    train_script = os.path.join(current_dir, "train.py")
    script_args = [
        "python", train_script,
        "--hf_token", args.token,
        "--max_train_samples", str(args.max_train_samples),
        "--max_val_samples", str(args.max_val_samples),
        "--train_batch_size", str(args.train_batch_size),
        "--val_batch_size", str(args.val_batch_size),
        "--num_train_epochs", str(args.num_train_epochs),
        "--mixed_precision", args.mixed_precision,
        "--gradient_accumulation_steps", str(args.gradient_accumulation_steps),
        "--lr_warmup_steps", str(args.lr_warmup_steps),
    ]

    print(f"Running: {' '.join(script_args)}")
    print(f"With PYTHONPATH: {current_dir}")

    # Set the PYTHONPATH environment variable
    env = os.environ.copy()
    env["PYTHONPATH"] = current_dir

    # Run the script
    subprocess.run(script_args, env=env)

if __name__ == "__main__":
    main()
