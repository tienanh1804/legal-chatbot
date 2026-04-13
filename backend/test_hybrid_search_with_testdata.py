#!/usr/bin/env python3
"""
Test hybrid search with test data from test_data.json.

This script evaluates the performance of hybrid search on a test dataset.
It calculates precision, recall, and accuracy for each query in the test set.
"""

import json
import logging
import os
import sys
import time
from typing import Dict, List, Set, Tuple

import numpy as np
from sklearn.metrics import precision_recall_fscore_support
from tqdm import tqdm

# Add backend directory to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# Import hybrid search functionality
try:
    from core.config import EMBEDDING_MODEL, HYBRID_ALPHA, QUERY_EXPANSION
    from search.hybrid_search import (
        CACHE_DIR,
        format_hybrid_results,
        get_embedding,
        hybrid_search,
        load_cached_documents,
        load_cached_embeddings,
        load_cached_metadata,
        load_or_create_bm25_index,
    )
except ImportError as e:
    print(f"Error importing modules: {e}")
    sys.exit(1)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler(os.path.join(CACHE_DIR, "test_hybrid_search.log")),
    ],
)
logger = logging.getLogger(__name__)

# Path to test data
TEST_DATA_PATH = os.path.join(
    os.path.dirname(os.path.abspath(__file__)), "json_data", "test_data.json"
)


def load_test_data() -> List[Dict]:
    """Load test data from JSON file."""
    try:
        with open(TEST_DATA_PATH, "r", encoding="utf-8") as f:
            test_data = json.load(f)
        logger.info(f"Loaded {len(test_data)} test queries from {TEST_DATA_PATH}")
        return test_data
    except Exception as e:
        logger.error(f"Error loading test data: {e}")
        return []


def evaluate_search_results(
    retrieved_docs: List[str], relevant_docs: List[str]
) -> Tuple[float, float, float]:
    """
    Evaluate search results by comparing retrieved documents with relevant documents.

    Args:
        retrieved_docs: List of document IDs retrieved by the search
        relevant_docs: List of document IDs that are relevant to the query

    Returns:
        Tuple of (precision, recall, accuracy)
    """
    # Chuẩn hóa ID tài liệu (loại bỏ phần mở rộng .md nếu có)
    retrieved_set = {doc_id.replace(".md", "") for doc_id in retrieved_docs}
    relevant_set = {doc_id.replace(".md", "") for doc_id in relevant_docs}

    # In ra để debug
    logger.info(f"Retrieved docs (normalized): {retrieved_set}")
    logger.info(f"Relevant docs (normalized): {relevant_set}")

    # Calculate metrics
    true_positives = len(retrieved_set.intersection(relevant_set))

    if len(retrieved_set) == 0:
        precision = 0.0
    else:
        precision = true_positives / len(retrieved_set)

    if len(relevant_set) == 0:
        recall = 0.0
    else:
        recall = true_positives / len(relevant_set)

    if precision + recall == 0:
        accuracy = 0.0
    else:
        accuracy = 2 * (precision * recall) / (precision + recall)

    return precision, recall, accuracy


def run_test(
    alpha: float = HYBRID_ALPHA, query_expansion: bool = QUERY_EXPANSION
) -> Dict:
    """
    Run hybrid search test on test data.

    Args:
        alpha: Weight for vector search (0-1)
        query_expansion: Whether to use query expansion

    Returns:
        Dictionary with test results
    """
    # Load test data
    test_data = load_test_data()
    if not test_data:
        logger.error("No test data found")
        return {}

    # Load resources for hybrid search
    documents = load_cached_documents()
    if not documents:
        logger.error("No cached documents found")
        return {}

    # Kiểm tra ID tài liệu trong cache
    logger.info(f"Document IDs in cache: {list(documents.keys())[:10]}...")
    for query_data in test_data:
        for doc_id in query_data["documents"]:
            if doc_id not in documents:
                logger.warning(f"Document ID '{doc_id}' not found in cache")
                # Thử tìm kiếm không có phần mở rộng .md
                if doc_id.replace(".md", "") in documents:
                    logger.info(
                        f"Document ID '{doc_id.replace('.md', '')}' found in cache (without .md extension)"
                    )
            else:
                logger.info(f"Document ID '{doc_id}' found in cache")

    bm25_model, corpus, doc_ids = load_or_create_bm25_index(documents)
    logger.info(f"BM25 resources loaded: {len(doc_ids)} documents in corpus")

    document_embeddings = load_cached_embeddings()
    if not document_embeddings:
        logger.error("No cached embeddings found")
        return {}

    doc_metadata = load_cached_metadata()
    if not doc_metadata:
        logger.warning("No cached metadata found, results will not include metadata")

    # Load embedding model
    try:
        from sentence_transformers import SentenceTransformer

        # Sử dụng fine-tuned embedding model
        model = SentenceTransformer(EMBEDDING_MODEL)
        logger.info(f"Loaded fine-tuned embedding model successfully")
    except Exception as e:
        logger.error(f"Error loading embedding model: {e}")
        return {}

    # Initialize results
    results = {
        "queries": [],
        "metrics": {
            "precision": {
                "at_1": 0.0,
                "at_3": 0.0,
                "at_5": 0.0,
                "at_10": 0.0,
            },
            "recall": {
                "at_1": 0.0,
                "at_3": 0.0,
                "at_5": 0.0,
                "at_10": 0.0,
            },
            "accuracy": {
                "at_1": 0.0,
                "at_3": 0.0,
                "at_5": 0.0,
                "at_10": 0.0,
            },
            "mrr": 0.0,  # Mean Reciprocal Rank
            "map": 0.0,  # Mean Average Precision
            "ndcg": {
                "at_1": 0.0,
                "at_3": 0.0,
                "at_5": 0.0,
                "at_10": 0.0,
            },
        },
        "config": {
            "alpha": alpha,
            "query_expansion": query_expansion,
        },
        "execution_time": 0.0,
    }

    # Process each query
    total_queries = len(test_data)
    total_time = 0.0

    for i, query_data in enumerate(tqdm(test_data, desc="Processing queries")):
        query = query_data["question"]
        relevant_docs = query_data["documents"]

        logger.info(f"Query {i+1}/{total_queries}: {query}")
        logger.info(f"Relevant documents: {relevant_docs}")

        # Measure execution time
        start_time = time.time()

        # Perform hybrid search
        query_embedding = model.encode(query, show_progress_bar=False).tolist()

        hybrid_results = hybrid_search(
            query=query,
            document_embeddings=document_embeddings,
            bm25_model=bm25_model,
            doc_ids=doc_ids,
            model=model,
            corpus=corpus,
            alpha=alpha,
            k=10,  # Get top 10 results for evaluation
            query_expansion=query_expansion,
            query_embedding=query_embedding,
        )

        # Kiểm tra kết quả hybrid_search
        logger.info(f"Hybrid search returned {len(hybrid_results)} results")
        if hybrid_results:
            logger.info(f"Top result: {hybrid_results[0]}")
        else:
            logger.warning("Hybrid search returned no results")

        execution_time = time.time() - start_time
        total_time += execution_time

        # Extract document IDs from results
        retrieved_docs = [doc_id for doc_id, _ in hybrid_results]

        # Calculate metrics at different thresholds
        precision_at_1, recall_at_1, accuracy_at_1 = evaluate_search_results(
            retrieved_docs[:1], relevant_docs
        )
        precision_at_3, recall_at_3, accuracy_at_3 = evaluate_search_results(
            retrieved_docs[:3], relevant_docs
        )
        precision_at_5, recall_at_5, accuracy_at_5 = evaluate_search_results(
            retrieved_docs[:5], relevant_docs
        )
        precision_at_10, recall_at_10, accuracy_at_10 = evaluate_search_results(
            retrieved_docs[:10], relevant_docs
        )

        # Calculate MRR (Mean Reciprocal Rank)
        mrr = 0.0
        for j, doc_id in enumerate(retrieved_docs):
            # Chuẩn hóa ID tài liệu để so sánh
            doc_id_normalized = doc_id.replace(".md", "")
            relevant_docs_normalized = [d.replace(".md", "") for d in relevant_docs]

            if doc_id_normalized in relevant_docs_normalized:
                mrr = 1.0 / (j + 1)
                break

        # Calculate MAP (Mean Average Precision)
        avg_precision = 0.0
        relevant_found = 0

        # Chuẩn hóa ID tài liệu để so sánh
        retrieved_docs_normalized = [d.replace(".md", "") for d in retrieved_docs]
        relevant_docs_normalized = [d.replace(".md", "") for d in relevant_docs]

        for j, doc_id in enumerate(retrieved_docs_normalized):
            if doc_id in relevant_docs_normalized:
                relevant_found += 1
                avg_precision += relevant_found / (j + 1)

        if len(relevant_docs_normalized) > 0:
            avg_precision /= len(relevant_docs_normalized)

        # Calculate NDCG (Normalized Discounted Cumulative Gain)
        def dcg_at_k(r, k):
            r = np.asarray(r, dtype=float)[:k]
            if r.size:
                return np.sum(r / np.log2(np.arange(2, r.size + 2)))
            return 0.0

        def ndcg_at_k(r, k):
            dcg_max = dcg_at_k(sorted(r, reverse=True), k)
            if not dcg_max:
                return 0.0
            return dcg_at_k(r, k) / dcg_max

        # Chuẩn hóa ID tài liệu để so sánh
        retrieved_docs_normalized = [d.replace(".md", "") for d in retrieved_docs]
        relevant_docs_normalized = [d.replace(".md", "") for d in relevant_docs]

        # Tạo danh sách relevance (1 nếu tài liệu liên quan, 0 nếu không)
        relevance = [
            1 if doc_id in relevant_docs_normalized else 0
            for doc_id in retrieved_docs_normalized
        ]
        ndcg_at_1 = ndcg_at_k(relevance, 1)
        ndcg_at_3 = ndcg_at_k(relevance, 3)
        ndcg_at_5 = ndcg_at_k(relevance, 5)
        ndcg_at_10 = ndcg_at_k(relevance, 10)

        # Store query results
        query_result = {
            "query": query,
            "relevant_docs": relevant_docs,
            "retrieved_docs": retrieved_docs,
            "precision": {
                "at_1": precision_at_1,
                "at_3": precision_at_3,
                "at_5": precision_at_5,
                "at_10": precision_at_10,
            },
            "recall": {
                "at_1": recall_at_1,
                "at_3": recall_at_3,
                "at_5": recall_at_5,
                "at_10": recall_at_10,
            },
            "accuracy": {
                "at_1": accuracy_at_1,
                "at_3": accuracy_at_3,
                "at_5": accuracy_at_5,
                "at_10": accuracy_at_10,
            },
            "mrr": mrr,
            "map": avg_precision,
            "ndcg": {
                "at_1": ndcg_at_1,
                "at_3": ndcg_at_3,
                "at_5": ndcg_at_5,
                "at_10": ndcg_at_10,
            },
            "execution_time": execution_time,
        }

        results["queries"].append(query_result)

        # Update overall metrics
        results["metrics"]["precision"]["at_1"] += precision_at_1
        results["metrics"]["precision"]["at_3"] += precision_at_3
        results["metrics"]["precision"]["at_5"] += precision_at_5
        results["metrics"]["precision"]["at_10"] += precision_at_10

        results["metrics"]["recall"]["at_1"] += recall_at_1
        results["metrics"]["recall"]["at_3"] += recall_at_3
        results["metrics"]["recall"]["at_5"] += recall_at_5
        results["metrics"]["recall"]["at_10"] += recall_at_10

        results["metrics"]["accuracy"]["at_1"] += accuracy_at_1
        results["metrics"]["accuracy"]["at_3"] += accuracy_at_3
        results["metrics"]["accuracy"]["at_5"] += accuracy_at_5
        results["metrics"]["accuracy"]["at_10"] += accuracy_at_10

        results["metrics"]["mrr"] += mrr
        results["metrics"]["map"] += avg_precision

        results["metrics"]["ndcg"]["at_1"] += ndcg_at_1
        results["metrics"]["ndcg"]["at_3"] += ndcg_at_3
        results["metrics"]["ndcg"]["at_5"] += ndcg_at_5
        results["metrics"]["ndcg"]["at_10"] += ndcg_at_10

    # Calculate average metrics
    if total_queries > 0:
        results["metrics"]["precision"]["at_1"] /= total_queries
        results["metrics"]["precision"]["at_3"] /= total_queries
        results["metrics"]["precision"]["at_5"] /= total_queries
        results["metrics"]["precision"]["at_10"] /= total_queries

        results["metrics"]["recall"]["at_1"] /= total_queries
        results["metrics"]["recall"]["at_3"] /= total_queries
        results["metrics"]["recall"]["at_5"] /= total_queries
        results["metrics"]["recall"]["at_10"] /= total_queries

        results["metrics"]["accuracy"]["at_1"] /= total_queries
        results["metrics"]["accuracy"]["at_3"] /= total_queries
        results["metrics"]["accuracy"]["at_5"] /= total_queries
        results["metrics"]["accuracy"]["at_10"] /= total_queries

        results["metrics"]["mrr"] /= total_queries
        results["metrics"]["map"] /= total_queries

        results["metrics"]["ndcg"]["at_1"] /= total_queries
        results["metrics"]["ndcg"]["at_3"] /= total_queries
        results["metrics"]["ndcg"]["at_5"] /= total_queries
        results["metrics"]["ndcg"]["at_10"] /= total_queries

    results["execution_time"] = total_time

    return results


def save_results(results: Dict, filename: str = "hybrid_search_results.json") -> None:
    """Save test results to a JSON file."""
    output_path = os.path.join(CACHE_DIR, filename)
    try:
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(results, f, ensure_ascii=False, indent=2)
        logger.info(f"Results saved to {output_path}")
    except Exception as e:
        logger.error(f"Error saving results: {e}")


def print_summary(results: Dict) -> None:
    """Print a summary of the test results."""
    if not results:
        print("No results to display")
        return

    print("\n" + "=" * 80)
    print(
        f"HYBRID SEARCH TEST SUMMARY (alpha={results['config']['alpha']}, query_expansion={results['config']['query_expansion']})"
    )
    print("=" * 80)

    metrics = results["metrics"]

    print("\nPrecision:")
    print(f"  @1:  {metrics['precision']['at_1']:.4f}")
    print(f"  @3:  {metrics['precision']['at_3']:.4f}")
    print(f"  @5:  {metrics['precision']['at_5']:.4f}")
    print(f"  @10: {metrics['precision']['at_10']:.4f}")

    print("\nRecall:")
    print(f"  @1:  {metrics['recall']['at_1']:.4f}")
    print(f"  @3:  {metrics['recall']['at_3']:.4f}")
    print(f"  @5:  {metrics['recall']['at_5']:.4f}")
    print(f"  @10: {metrics['recall']['at_10']:.4f}")

    print("\nAccuracy:")
    print(f"  @1:  {metrics['accuracy']['at_1']:.4f}")
    print(f"  @3:  {metrics['accuracy']['at_3']:.4f}")
    print(f"  @5:  {metrics['accuracy']['at_5']:.4f}")
    print(f"  @10: {metrics['accuracy']['at_10']:.4f}")

    print("\nNDCG:")
    print(f"  @1:  {metrics['ndcg']['at_1']:.4f}")
    print(f"  @3:  {metrics['ndcg']['at_3']:.4f}")
    print(f"  @5:  {metrics['ndcg']['at_5']:.4f}")
    print(f"  @10: {metrics['ndcg']['at_10']:.4f}")

    print(f"\nMRR: {metrics['mrr']:.4f}")
    print(f"MAP: {metrics['map']:.4f}")

    print(f"\nTotal execution time: {results['execution_time']:.2f} seconds")
    print(
        f"Average time per query: {results['execution_time'] / len(results['queries']):.2f} seconds"
    )
    print("=" * 80 + "\n")


def test_with_qdrant_integration(
    alpha: float = HYBRID_ALPHA, query_expansion: bool = QUERY_EXPANSION
) -> Dict:
    """
    Run hybrid search test with Qdrant integration.

    This is a placeholder function for future implementation.

    Args:
        alpha: Weight for vector search (0-1)
        query_expansion: Whether to use query expansion

    Returns:
        Dictionary with test results
    """
    # This function would implement Qdrant integration for vector search
    # For now, it's just a placeholder
    logger.info("Qdrant integration not implemented yet")
    return {}


def main():
    """Run the hybrid search test."""
    print("Running hybrid search test with test data...")

    # Run test with default parameters
    results = run_test()

    # Print summary
    print_summary(results)

    # Save results
    save_results(results)

    print("Test completed.")


if __name__ == "__main__":
    main()
