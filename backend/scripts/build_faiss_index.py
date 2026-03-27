#!/usr/bin/env python3
"""
Build FAISS index for vector search.

This script builds a FAISS index from the document embeddings cache:
1. Loads document embeddings from cache
2. Creates a FAISS index
3. Saves the index to disk

Run this script after updating the document collection to rebuild the FAISS index.
"""

import logging
import os
import time
from typing import Dict, List

from core.config import USE_FAISS
from search.hybrid_search import (
    FAISS_INDEX_PATH,
    FAISS_MAPPING_PATH,
    create_faiss_index,
    load_cached_embeddings,
)

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Constants
# Use absolute paths for cache files
CACHE_DIR = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "cache"
)
# Create cache directory if it doesn't exist
os.makedirs(CACHE_DIR, exist_ok=True)


def build_faiss_index(force_rebuild: bool = False) -> None:
    """
    Build FAISS index for vector search.

    Args:
        force_rebuild: Whether to force rebuilding the index
    """
    if not USE_FAISS:
        logger.warning("FAISS is disabled in config. Skipping index creation.")
        return

    start_time = time.time()
    logger.info("Starting to build FAISS index...")

    # Step 1: Load document embeddings from cache
    logger.info("Loading document embeddings...")
    document_embeddings = load_cached_embeddings()
    
    if not document_embeddings:
        logger.error("No document embeddings found. Cannot create FAISS index.")
        return
    
    logger.info(f"Loaded {len(document_embeddings)} document embeddings")

    # Step 2: Create FAISS index
    logger.info("Creating FAISS index...")
    success = create_faiss_index(document_embeddings, FAISS_INDEX_PATH, FAISS_MAPPING_PATH)
    
    if success:
        logger.info(f"FAISS index created successfully at {FAISS_INDEX_PATH}")
        logger.info(f"FAISS mapping saved at {FAISS_MAPPING_PATH}")
    else:
        logger.error("Failed to create FAISS index")

    logger.info(f"FAISS index building completed in {time.time() - start_time:.2f} seconds")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Build FAISS index for vector search"
    )
    parser.add_argument("--force", action="store_true", help="Force rebuild the index")

    args = parser.parse_args()

    build_faiss_index(args.force)
