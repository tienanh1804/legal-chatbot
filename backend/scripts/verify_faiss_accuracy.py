"""
Script to verify FAISS accuracy compared to direct cosine similarity.

This script:
1. Creates a FAISS index with normalized vectors using IndexFlatIP
2. Performs searches and compares results with direct cosine similarity
3. Verifies that the results and scores are identical

Usage:
    python verify_faiss_accuracy.py
"""

import logging
import os
import sys
import time

import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

# Add backend directory to sys.path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from core.config import EMBEDDING_MODEL
from search.hybrid_search import (
    CACHE_DIR,
    create_faiss_index,
    load_cached_embeddings,
    load_faiss_index,
    search_faiss_index,
)
from sentence_transformers import SentenceTransformer

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def verify_faiss_accuracy():
    """Verify FAISS accuracy compared to direct cosine similarity."""
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
    test_index_path = os.path.join(CACHE_DIR, "test_faiss_accuracy.bin")
    test_mapping_path = os.path.join(CACHE_DIR, "test_faiss_accuracy_mapping.pkl")

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
    total_accuracy = 0.0
    total_queries = len(test_queries)

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

        same_order = direct_ids == faiss_ids
        logger.info(f"\nResults in same order: {same_order}")

        # Check similarity scores
        score_diffs = []
        if same_order:
            logger.info("Comparing similarity scores:")
            for i in range(len(direct_results)):
                direct_id, direct_score = direct_results[i]
                faiss_id, faiss_score = faiss_results[i]

                # Calculate absolute difference
                diff = abs(direct_score - faiss_score)
                score_diffs.append(diff)
                logger.info(
                    f"  {direct_id}: Direct={direct_score:.6f}, FAISS={faiss_score:.6f}, Diff={diff:.6f}"
                )

            # Calculate average score difference
            avg_diff = sum(score_diffs) / len(score_diffs) if score_diffs else 0
            logger.info(f"Average score difference: {avg_diff:.6f}")

            # Calculate accuracy as 1 - avg_diff (closer to 1 is better)
            accuracy = 1.0 - min(avg_diff, 1.0)
            logger.info(f"Accuracy: {accuracy:.6f} (1.0 is perfect)")

            total_accuracy += accuracy

    # Calculate overall accuracy
    if total_queries > 0:
        overall_accuracy = total_accuracy / total_queries
        logger.info(f"\nOverall accuracy across all queries: {overall_accuracy:.6f}")

    # Clean up test files
    logger.info("\nCleaning up test files...")
    try:
        os.remove(test_index_path)
        os.remove(test_mapping_path)
        logger.info("Test files removed")
    except Exception as e:
        logger.warning(f"Error removing test files: {e}")

    logger.info("\nVerification completed.")


if __name__ == "__main__":
    verify_faiss_accuracy()
