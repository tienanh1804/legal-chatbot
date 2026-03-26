#!/usr/bin/env python3
"""
Ensure cache directory exists and has the right permissions.

This script creates the cache directory if it doesn't exist and sets the right permissions.
It's meant to be run before starting the application.
"""

import os
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def ensure_cache_directory():
    """Create cache directory if it doesn't exist and set permissions."""
    # Define cache directory
    cache_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "cache")
    
    # Create cache directory if it doesn't exist
    if not os.path.exists(cache_dir):
        logger.info(f"Creating cache directory: {cache_dir}")
        os.makedirs(cache_dir, exist_ok=True)
    else:
        logger.info(f"Cache directory already exists: {cache_dir}")
    
    # Set permissions (read/write for owner and group)
    try:
        os.chmod(cache_dir, 0o775)
        logger.info(f"Set permissions for cache directory: {cache_dir}")
    except Exception as e:
        logger.warning(f"Failed to set permissions for cache directory: {e}")
    
    # Create a test file to verify write permissions
    test_file = os.path.join(cache_dir, "test.txt")
    try:
        with open(test_file, "w") as f:
            f.write("Test write permissions")
        logger.info(f"Successfully wrote test file: {test_file}")
        
        # Remove test file
        os.remove(test_file)
        logger.info(f"Removed test file: {test_file}")
    except Exception as e:
        logger.error(f"Failed to write to cache directory: {e}")
        logger.error("The application may not be able to save cache files!")

if __name__ == "__main__":
    ensure_cache_directory()
