#!/usr/bin/env python
"""
Script to test Hugging Face authentication.
"""
import os
import sys
import argparse
import logging

# Add the parent directory to the Python path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from huggingface_hub import login, HfApi

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)

def parse_args():
    parser = argparse.ArgumentParser(description="Test Hugging Face authentication")

    parser.add_argument(
        "--token",
        type=str,
        required=True,
        help="Hugging Face API token",
    )

    return parser.parse_args()

def main():
    args = parse_args()

    # Login to Hugging Face Hub
    logger.info("Logging in to Hugging Face Hub")
    login(token=args.token)

    # Test authentication
    api = HfApi()
    user_info = api.whoami()

    # Log user info
    if 'email' in user_info:
        logger.info(f"Successfully authenticated as: {user_info['name']} ({user_info['email']})")
    else:
        logger.info(f"Successfully authenticated as: {user_info['name']}")

    # Print all available user info for debugging
    logger.info("User info:")
    for key, value in user_info.items():
        logger.info(f"  {key}: {value}")

    logger.info("Authentication test passed!")

if __name__ == "__main__":
    main()
