"""
Script to test FAISS search functionality.

This script tests the FAISS search functionality by:
1. Loading the FAISS index
2. Performing a search using a test query
3. Comparing the results with the original cosine similarity search

Usage:
    python test_faiss_search.py
"""

import logging
import os
import sys
import time

# Add backend directory to sys.path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from core.config import EMBEDDING_MODEL
from search.hybrid_search import (
    FAISS_INDEX_PATH,
    FAISS_MAPPING_PATH,
    load_cached_documents,
    load_cached_embeddings,
    load_faiss_index,
    query_hybrid_search,
    search_faiss_index,
)
from sentence_transformers import SentenceTransformer

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def test_faiss_search():
    """Test FAISS search functionality."""
    # Test query
    test_query = "Quy định về bảo vệ dữ liệu cá nhân"

    logger.info(f"Testing FAISS search with query: '{test_query}'")

    # Load embedding model
    logger.info(f"Loading embedding model: {EMBEDDING_MODEL}")
    try:
        model = SentenceTransformer(EMBEDDING_MODEL)
        logger.info(f"Loaded embedding model: {EMBEDDING_MODEL}")
    except Exception as e:
        logger.error(f"Error loading embedding model: {e}")
        return

    # Encode the query
    query_embedding = model.encode(test_query, show_progress_bar=False)

    # Load FAISS index
    logger.info("Loading FAISS index...")
    faiss_index, doc_ids = load_faiss_index(FAISS_INDEX_PATH, FAISS_MAPPING_PATH)
    if faiss_index is None or doc_ids is None:
        logger.error("Failed to load FAISS index")
        return

    # Perform FAISS search
    logger.info("Performing FAISS search...")
    start_time = time.time()
    faiss_results = search_faiss_index(query_embedding, faiss_index, doc_ids, k=5)
    faiss_time = time.time() - start_time

    # Load document embeddings for comparison
    logger.info("Loading document embeddings...")
    document_embeddings = load_cached_embeddings()
    if not document_embeddings:
        logger.error("Failed to load document embeddings")
        return

    # Load documents to display results
    documents = load_cached_documents()
    if not documents:
        logger.error("Failed to load documents")
        return

    # Display FAISS results
    logger.info(f"FAISS search completed in {faiss_time:.4f} seconds")
    logger.info("FAISS search results:")
    for i, (doc_id, score) in enumerate(faiss_results):
        # Get first 100 characters of document content
        content_preview = documents.get(doc_id, "")[:100].replace("\n", " ") + "..."
        logger.info(f"{i+1}. {doc_id} (score: {score:.4f}): {content_preview}")

    # Compare FAISS search with direct embedding search
    logger.info("\nComparing FAISS search with direct embedding search:")

    # Direct embedding search (cosine similarity)
    logger.info("Performing direct embedding search...")
    start_time = time.time()

    # Sort document embeddings by cosine similarity
    import numpy as np
    from sklearn.metrics.pairwise import cosine_similarity

    direct_results = []
    for doc_id, embedding in document_embeddings.items():
        sim = cosine_similarity([query_embedding], [embedding])[0][0]
        direct_results.append((doc_id, sim))

    # Sort by similarity (descending)
    direct_results = sorted(direct_results, key=lambda x: x[1], reverse=True)[:5]
    direct_time = time.time() - start_time

    # Display direct search results
    logger.info(f"Direct search completed in {direct_time:.4f} seconds")
    logger.info("Direct search results:")
    for i, (doc_id, score) in enumerate(direct_results):
        # Get first 100 characters of document content
        content_preview = documents.get(doc_id, "")[:100].replace("\n", " ") + "..."
        logger.info(f"{i+1}. {doc_id} (score: {score:.4f}): {content_preview}")

    # Compare times
    logger.info(f"\nFAISS search time: {faiss_time:.4f} seconds")
    logger.info(f"Direct search time: {direct_time:.4f} seconds")
    logger.info(f"Speed improvement: {(direct_time / faiss_time):.2f}x")

    # Compare results
    logger.info("\nComparing top results:")
    faiss_ids = [doc_id for doc_id, _ in faiss_results]
    direct_ids = [doc_id for doc_id, _ in direct_results]

    common_ids = set(faiss_ids).intersection(set(direct_ids))
    logger.info(
        f"Common documents in top 5 results: {len(common_ids)}/{min(len(faiss_ids), len(direct_ids))}"
    )

    logger.info("\nTest completed.")


if __name__ == "__main__":
    test_faiss_search()
