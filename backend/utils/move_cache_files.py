#!/usr/bin/env python3
"""
Move cache files from root directory to cache directory.

This script moves all .pkl cache files from the root directory to the cache directory.
"""

import logging
import os
import shutil

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def move_cache_files():
    """Move cache files from root directory to cache directory."""
    # Define directories
    root_dir = os.path.dirname(os.path.abspath(__file__))
    cache_dir = os.path.join(root_dir, "cache")

    # Create cache directory if it doesn't exist
    if not os.path.exists(cache_dir):
        logger.info(f"Creating cache directory: {cache_dir}")
        os.makedirs(cache_dir, exist_ok=True)

    # List of cache files to move
    cache_files = [
        "bm25_cache.pkl",
        "document_cache.pkl",
        "embeddings_cache.pkl",
        "metadata_cache.pkl",
    ]

    # Move each file if it exists
    for file_name in cache_files:
        source_path = os.path.join(root_dir, file_name)
        dest_path = os.path.join(cache_dir, file_name)

        if os.path.exists(source_path):
            try:
                # Copy the file to the cache directory
                shutil.copy2(source_path, dest_path)
                logger.info(f"Copied {file_name} to cache directory")

                # Remove the original file
                os.remove(source_path)
                logger.info(f"Removed original {file_name} from root directory")
            except Exception as e:
                logger.error(f"Error moving {file_name}: {e}")
        else:
            logger.warning(f"File {file_name} not found in root directory")

    # List files in cache directory
    cache_files = os.listdir(cache_dir)
    logger.info(f"Files in cache directory: {cache_files}")


if __name__ == "__main__":
    move_cache_files()
