"""
Script to test FAISS with normalized vectors for cosine similarity.

This script:
1. Creates a FAISS index with normalized vectors using IndexFlatIP
2. Performs searches and compares results with direct cosine similarity
3. Verifies that the results are identical

Usage:
    python test_faiss_cosine.py
"""

import logging
import os
import sys
import time
from typing import Dict, List, Tuple

import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

# Add backend directory to sys.path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from core.config import EMBEDDING_MODEL
from search.hybrid_search import (
    CACHE_DIR,
    FAISS_INDEX_PATH,
    FAISS_MAPPING_PATH,
    create_faiss_index,
    load_cached_embeddings,
    load_faiss_index,
    search_faiss_index,
)
from sentence_transformers import SentenceTransformer

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def test_faiss_cosine_similarity():
    """Test FAISS with normalized vectors for cosine similarity."""
    # Test queries
    test_queries = [
        "Quy định về bảo vệ dữ liệu cá nhân",
        "Chính sách hỗ trợ người lao động",
        "Quyền lợi của người khuyết tật",
        "Thủ tục đăng ký kinh doanh",
        "Luật bảo vệ môi trường",
    ]

    # Load embedding model
    logger.info(f"Loading embedding model: {EMBEDDING_MODEL}")
    try:
        model = SentenceTransformer(EMBEDDING_MODEL)
        logger.info(f"Loaded embedding model: {EMBEDDING_MODEL}")
    except Exception as e:
        logger.error(f"Error loading embedding model: {e}")
        return

    # Load document embeddings
    logger.info("Loading document embeddings...")
    document_embeddings = load_cached_embeddings()
    if not document_embeddings:
        logger.error("Failed to load document embeddings")
        return

    # Create a new FAISS index with normalized vectors
    logger.info("Creating FAISS index with normalized vectors...")
    test_index_path = os.path.join(CACHE_DIR, "test_faiss_cosine.bin")
    test_mapping_path = os.path.join(CACHE_DIR, "test_faiss_cosine_mapping.pkl")

    success = create_faiss_index(
        document_embeddings, test_index_path, test_mapping_path
    )
    if not success:
        logger.error("Failed to create FAISS index")
        return

    # Load the FAISS index
    logger.info("Loading FAISS index...")
    faiss_index, doc_ids = load_faiss_index(test_index_path, test_mapping_path)
    if faiss_index is None or doc_ids is None:
        logger.error("Failed to load FAISS index")
        return

    # Process each test query
    for query in test_queries:
        logger.info(f"\nTesting query: '{query}'")

        # Encode the query
        query_embedding = model.encode(query, show_progress_bar=False)

        # Method 1: Direct cosine similarity
        logger.info("Computing direct cosine similarity...")
        start_time = time.time()

        direct_results = []
        for doc_id, embedding in document_embeddings.items():
            sim = cosine_similarity([query_embedding], [embedding])[0][0]
            direct_results.append((doc_id, sim))

        direct_results = sorted(direct_results, key=lambda x: x[1], reverse=True)[:5]
        direct_time = time.time() - start_time

        # Method 2: FAISS with normalized vectors
        logger.info("Searching FAISS index with normalized vectors...")
        start_time = time.time()
        faiss_results = search_faiss_index(query_embedding, faiss_index, doc_ids, k=5)
        faiss_time = time.time() - start_time

        # Compare results
        logger.info(f"Direct cosine similarity time: {direct_time:.6f} seconds")
        logger.info(f"FAISS search time: {faiss_time:.6f} seconds")
        logger.info(f"Speed improvement: {direct_time / faiss_time:.2f}x")

        # Check if the top results are the same
        direct_ids = [doc_id for doc_id, _ in direct_results]
        faiss_ids = [doc_id for doc_id, _ in faiss_results]

        logger.info("\nTop 5 results from direct cosine similarity:")
        for i, (doc_id, score) in enumerate(direct_results):
            logger.info(f"{i+1}. {doc_id} (score: {score:.6f})")

        logger.info("\nTop 5 results from FAISS search:")
        for i, (doc_id, score) in enumerate(faiss_results):
            logger.info(f"{i+1}. {doc_id} (score: {score:.6f})")

        # Check if the order is the same
        same_order = direct_ids == faiss_ids
        logger.info(f"\nResults in same order: {same_order}")

        # Check similarity scores
        if same_order:
            logger.info("Comparing similarity scores:")
            for i in range(len(direct_results)):
                direct_id, direct_score = direct_results[i]
                faiss_id, faiss_score = faiss_results[i]

                # For normalized vectors with inner product, the scores should be very close
                # The difference is due to floating point precision and normalization
                diff = abs(direct_score - faiss_score)
                logger.info(
                    f"  {direct_id}: Direct={direct_score:.6f}, FAISS={faiss_score:.6f}, Diff={diff:.6f}"
                )

    # Clean up test files
    logger.info("\nCleaning up test files...")
    try:
        os.remove(test_index_path)
        os.remove(test_mapping_path)
        logger.info("Test files removed")
    except Exception as e:
        logger.warning(f"Error removing test files: {e}")

    logger.info("\nTest completed.")


if __name__ == "__main__":
    test_faiss_cosine_similarity()
