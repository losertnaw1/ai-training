#!/usr/bin/env python
"""
Script to run the authentication test script with the correct Python path.
"""
import os
import sys
import subprocess
import argparse

def main():
    parser = argparse.ArgumentParser(description="Run the authentication test script")
    parser.add_argument(
        "--token",
        type=str,
        required=True,
        help="Hugging Face API token",
    )
    args = parser.parse_args()

    # Add the current directory to the Python path
    current_dir = os.path.abspath(os.path.dirname(__file__))
    sys.path.insert(0, current_dir)

    # Run the test auth script
    test_script = os.path.join(current_dir, "scripts", "test_auth.py")
    script_args = ["python", test_script, "--token", args.token]

    print(f"Running: {' '.join(script_args)}")
    print(f"With PYTHONPATH: {current_dir}")

    # Set the PYTHONPATH environment variable
    env = os.environ.copy()
    env["PYTHONPATH"] = current_dir

    # Run the script
    subprocess.run(script_args, env=env)

if __name__ == "__main__":
    main()
