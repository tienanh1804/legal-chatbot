#!/usr/bin/env python3
"""
Build search resources for hybrid search.

This script builds and caches all necessary resources for hybrid search:
1. Document cache - raw document content
2. BM25 index - for lexical search
3. Document embeddings - for vector search
4. Document metadata - for result formatting

Run this script after updating the document collection to rebuild the cache.
"""

import logging
import os
import time
from typing import Any, Dict, List

from core.config import EMBEDDING_MODEL, MARKDOWN_DIR
from search.hybrid_search import (
    load_or_create_bm25_index,
    load_or_create_document_cache,
    load_or_create_embeddings,
    read_markdown_files,
)
from sentence_transformers import SentenceTransformer

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

BM25_CACHE_PATH = os.path.join(CACHE_DIR, "bm25_cache.pkl")
EMBEDDINGS_CACHE_PATH = os.path.join(CACHE_DIR, "embeddings_cache.pkl")
METADATA_CACHE_PATH = os.path.join(CACHE_DIR, "metadata_cache.pkl")
DOCUMENT_CACHE_PATH = os.path.join(CACHE_DIR, "document_cache.pkl")


def build_search_resources(force_rebuild: bool = False) -> None:
    """
    Build all search resources for hybrid search.

    Args:
        force_rebuild: Whether to force rebuilding all caches
    """
    start_time = time.time()
    logger.info("Starting to build search resources...")

    # Step 1: Load or create document cache
    logger.info("Building document cache...")
    documents = load_or_create_document_cache(MARKDOWN_DIR, DOCUMENT_CACHE_PATH)
    logger.info(f"Document cache built with {len(documents)} documents")

    # Step 2: Load or create BM25 index
    logger.info("Building BM25 index...")
    bm25_model, corpus, doc_ids = load_or_create_bm25_index(documents, BM25_CACHE_PATH)
    logger.info(f"BM25 index built with {len(doc_ids)} documents")

    # Step 3: Load or create document embeddings
    logger.info("Building document embeddings...")
    try:
        # Load the model
        model = SentenceTransformer(EMBEDDING_MODEL)

        # Create embeddings
        document_embeddings = load_or_create_embeddings(
            documents, model, EMBEDDINGS_CACHE_PATH, METADATA_CACHE_PATH
        )
        logger.info(
            f"Document embeddings built for {len(document_embeddings)} documents"
        )
    except Exception as e:
        logger.error(f"Error building document embeddings: {e}")
        logger.warning("Hybrid search will fall back to BM25 only")

    logger.info(f"All search resources built in {time.time() - start_time:.2f} seconds")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Build search resources for hybrid search"
    )
    parser.add_argument("--force", action="store_true", help="Force rebuild all caches")

    args = parser.parse_args()

    build_search_resources(args.force)
