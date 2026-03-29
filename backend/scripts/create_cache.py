"""
Script to create cache files for hybrid search.

This script creates the following cache files:
1. document_cache.pkl - Cache for documents
2. bm25_cache.pkl - Cache for BM25 index
3. embeddings_cache.pkl - Cache for document embeddings
4. metadata_cache.pkl - Cache for metadata

Usage:
    python create_cache.py
"""

import logging
import os
import sys
import time

# Thêm thư mục gốc của backend vào sys.path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from core.config import EMBEDDING_MODEL

# Import hybrid search functionality
from search.hybrid_search import (
    BM25_CACHE_PATH,
    CACHE_DIR,
    DOCUMENT_CACHE_PATH,
    EMBEDDINGS_CACHE_PATH,
    FAISS_INDEX_PATH,
    FAISS_MAPPING_PATH,
    MARKDOWN_DIR,
    METADATA_CACHE_PATH,
    create_faiss_index,
    load_or_create_bm25_index,
    load_or_create_document_cache,
    load_or_create_embeddings,
    read_markdown_files,
)
from sentence_transformers import SentenceTransformer
from tqdm import tqdm

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def create_cache_files():
    """Create cache files for hybrid search."""
    start_time = time.time()

    # Create cache directory if it doesn't exist
    os.makedirs(CACHE_DIR, exist_ok=True)

    # Step 1: Load or create document cache
    logger.info("Step 1: Loading or creating document cache...")
    documents = load_or_create_document_cache(MARKDOWN_DIR, DOCUMENT_CACHE_PATH)
    logger.info(f"Loaded {len(documents)} documents.")

    # Step 2: Load or create BM25 index
    logger.info("Step 2: Loading or creating BM25 index...")
    bm25_model, corpus, doc_ids = load_or_create_bm25_index(documents, BM25_CACHE_PATH)
    logger.info(f"Created BM25 index with {len(doc_ids)} documents.")

    # Step 3: Load embedding model
    logger.info(f"Step 3: Loading embedding model {EMBEDDING_MODEL}...")
    try:
        model = SentenceTransformer(EMBEDDING_MODEL)
        logger.info(f"Loaded embedding model: {EMBEDDING_MODEL}")
    except Exception as e:
        logger.error(f"Error loading embedding model: {e}")
        return

    # Step 4: Load or create document embeddings
    logger.info("Step 4: Loading or creating document embeddings...")
    document_embeddings = load_or_create_embeddings(
        documents, model, EMBEDDINGS_CACHE_PATH, METADATA_CACHE_PATH
    )
    logger.info(f"Created embeddings for {len(document_embeddings)} documents.")

    # Step 5: Create FAISS index from embeddings
    logger.info("Step 5: Creating FAISS index from embeddings...")
    faiss_success = create_faiss_index(
        document_embeddings, FAISS_INDEX_PATH, FAISS_MAPPING_PATH
    )
    if faiss_success:
        logger.info("Successfully created FAISS index.")
    else:
        logger.error("Failed to create FAISS index.")

    # Calculate total time
    total_time = time.time() - start_time
    logger.info(f"Cache creation completed in {total_time:.2f} seconds.")

    # Print cache file paths
    logger.info(f"Document cache: {DOCUMENT_CACHE_PATH}")
    logger.info(f"BM25 cache: {BM25_CACHE_PATH}")
    logger.info(f"Embeddings cache: {EMBEDDINGS_CACHE_PATH}")
    logger.info(f"Metadata cache: {METADATA_CACHE_PATH}")
    logger.info(f"FAISS index: {FAISS_INDEX_PATH}")
    logger.info(f"FAISS mapping: {FAISS_MAPPING_PATH}")


if __name__ == "__main__":
    create_cache_files()
