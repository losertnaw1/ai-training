#!/usr/bin/env python
"""
Script to run the test dataset script with the correct Python path.
"""
import os
import sys
import subprocess
import argparse

def main():
    parser = argparse.ArgumentParser(description="Run the test dataset script")
    parser.add_argument(
        "--token",
        type=str,
        required=True,
        help="Hugging Face API token",
    )
    parser.add_argument(
        "--num_samples",
        type=int,
        default=2,
        help="Number of samples to display",
    )
    args = parser.parse_args()

    # Add the current directory to the Python path
    current_dir = os.path.abspath(os.path.dirname(__file__))
    sys.path.insert(0, current_dir)

    # Run the test dataset script
    test_script = os.path.join(current_dir, "scripts", "test_dataset.py")
    script_args = [
        "python",
        test_script,
        "--num_samples", str(args.num_samples),
        "--hf_token", args.token
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
