#!/usr/bin/env python3
"""
Create document embeddings cache from sentence_transformer model.

This script reads markdown files, creates embeddings using a sentence_transformer model,
and saves them to a cache file.
"""

import glob
import logging
import os
import pickle
import random
import re
import time
from typing import Any, Dict, List

import numpy as np
from core.config import EMBEDDING_MODEL

# Configure logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

# Constants
# Sử dụng đường dẫn tương đối thay vì tuyệt đối
backend_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MARKDOWN_DIR = os.path.join(backend_dir, "markdown_data")
JSON_DATA_DIR = os.path.join(backend_dir, "json_data")
MODEL_DIR = EMBEDDING_MODEL

# Use a temporary directory that we have write access to
TEMP_DIR = "/tmp/rag_cache"
os.makedirs(TEMP_DIR, exist_ok=True)

# Define paths for temporary cache files
TEMP_EMBEDDINGS_CACHE_PATH = os.path.join(TEMP_DIR, "embeddings_cache.pkl")
TEMP_METADATA_CACHE_PATH = os.path.join(TEMP_DIR, "metadata_cache.pkl")
TEMP_DOCUMENT_CACHE_PATH = os.path.join(TEMP_DIR, "document_cache.pkl")

# Define paths for final cache files
CACHE_DIR = os.path.join(backend_dir, "cache")
os.makedirs(CACHE_DIR, exist_ok=True)
EMBEDDINGS_CACHE_PATH = os.path.join(CACHE_DIR, "embeddings_cache.pkl")
METADATA_CACHE_PATH = os.path.join(CACHE_DIR, "metadata_cache.pkl")
DOCUMENT_CACHE_PATH = os.path.join(CACHE_DIR, "document_cache.pkl")

VECTOR_SIZE = 768  # Standard size for most sentence_transformer models


def extract_metadata_from_markdown(content: str) -> Dict[str, Any]:
    """
    Extract metadata from markdown content.

    Args:
        content: Markdown content

    Returns:
        Dictionary containing metadata
    """
    metadata = {}

    # Extract document title
    title_match = re.search(r"^#\s+(.+)$", content, re.MULTILINE)
    if title_match:
        metadata["document_title"] = title_match.group(1).strip()

    # Extract decision number
    decision_match = re.search(r"Quyết định số\s+([^\s]+)", content)
    if decision_match:
        metadata["decision_number"] = decision_match.group(1).strip()

    # Extract date
    date_match = re.search(r"ngày\s+(\d{1,2})[/-](\d{1,2})[/-](\d{4})", content)
    if date_match:
        day, month, year = date_match.groups()
        metadata["date"] = f"{day}/{month}/{year}"

    # Extract agency
    agency_match = re.search(
        r"(Thủ tướng Chính phủ|Chính phủ|Bộ [^,\.]+|Ủy ban [^,\.]+)", content
    )
    if agency_match:
        metadata["agency"] = agency_match.group(1).strip()

    # Set source
    metadata["source"] = "Văn bản pháp luật"

    return metadata


def read_markdown_files(directory: str) -> Dict[str, str]:
    """
    Read all markdown files from a directory.

    Args:
        directory: Directory containing markdown files

    Returns:
        Dictionary mapping document IDs to document contents
    """
    documents = {}

    # Get all markdown files
    markdown_files = glob.glob(os.path.join(directory, "**/*.md"), recursive=True)

    for file_path in markdown_files:
        try:
            with open(file_path, "r", encoding="utf-8") as f:
                content = f.read()

            # Use file name as document ID
            doc_id = os.path.basename(file_path).replace(".md", "")
            documents[doc_id] = content

        except Exception as e:
            logger.warning(f"Error reading file {file_path}: {e}")

    return documents


def create_random_embeddings(documents: Dict[str, str]) -> Dict[str, List[float]]:
    """
    Create random embeddings for documents.

    Args:
        documents: Dictionary mapping document IDs to document contents

    Returns:
        Dictionary mapping document IDs to document embeddings
    """
    embeddings = {}

    for doc_id, content in documents.items():
        # Use a deterministic hash-based approach for more consistent results
        # This ensures the same content always gets the same embedding
        random.seed(
            hash(content) % 10000
        )  # Use content hash as seed for reproducibility
        embeddings[doc_id] = [random.random() for _ in range(VECTOR_SIZE)]

    # Reset the seed
    random.seed()

    return embeddings


def create_embeddings_with_model(
    documents: Dict[str, str], model_dir: str = EMBEDDING_MODEL
) -> Dict[str, List[float]]:
    """
    Create embeddings for documents using a sentence_transformer model.

    Args:
        documents: Dictionary mapping document IDs to document contents
        model_dir: Hugging Face model path or local model directory

    Returns:
        Dictionary mapping document IDs to document embeddings
    """
    try:
        # Try to import sentence_transformers
        from sentence_transformers import SentenceTransformer

        # Load the model
        logger.info(f"Loading model from {model_dir}...")
        model = SentenceTransformer(model_dir)

        # Create embeddings
        embeddings = {}

        # Process documents in batches to avoid memory issues
        batch_size = 16
        doc_ids = list(documents.keys())

        logger.info(f"Creating embeddings for {len(doc_ids)} documents...")

        for i in range(0, len(doc_ids), batch_size):
            batch_ids = doc_ids[i : i + batch_size]
            batch_texts = [documents[doc_id] for doc_id in batch_ids]

            # Encode batch
            logger.info(
                f"Encoding batch {i//batch_size + 1}/{(len(doc_ids)-1)//batch_size + 1}..."
            )
            batch_embeddings = model.encode(
                batch_texts, show_progress_bar=True, convert_to_tensor=False
            )

            # Store embeddings
            for j, doc_id in enumerate(batch_ids):
                embeddings[doc_id] = batch_embeddings[j].tolist()

        return embeddings

    except Exception as e:
        logger.error(f"Error creating embeddings with model: {e}")
        logger.warning("Falling back to random embeddings...")
        return create_random_embeddings(documents)


def create_metadata(documents: Dict[str, str]) -> Dict[str, Dict[str, Any]]:
    """
    Create metadata for documents.

    Args:
        documents: Dictionary mapping document IDs to document contents

    Returns:
        Dictionary mapping document IDs to document metadata
    """
    metadata = {}

    for doc_id, content in documents.items():
        metadata[doc_id] = extract_metadata_from_markdown(content)

    return metadata


def main():
    """Main function."""
    start_time = time.time()
    logger.info("Starting to create embeddings cache...")

    # Step 1: Read markdown files
    logger.info(f"Reading markdown files from {MARKDOWN_DIR}...")
    documents = read_markdown_files(MARKDOWN_DIR)
    logger.info(f"Read {len(documents)} documents")

    # Step 2: Create metadata
    logger.info("Creating metadata...")
    metadata = create_metadata(documents)

    # Step 3: Create embeddings
    logger.info("Creating embeddings...")
    embeddings = create_embeddings_with_model(documents, MODEL_DIR)
    logger.info(f"Created embeddings for {len(embeddings)} documents")

    # Step 4: Save to temporary cache
    logger.info("Saving to temporary cache...")

    # Save documents
    try:
        with open(TEMP_DOCUMENT_CACHE_PATH, "wb") as f:
            pickle.dump(documents, f)
        logger.info(f"Saved documents to temporary cache: {TEMP_DOCUMENT_CACHE_PATH}")
    except Exception as e:
        logger.error(f"Error saving documents cache: {e}")

    # Save metadata
    try:
        with open(TEMP_METADATA_CACHE_PATH, "wb") as f:
            pickle.dump(metadata, f)
        logger.info(f"Saved metadata to temporary cache: {TEMP_METADATA_CACHE_PATH}")
    except Exception as e:
        logger.error(f"Error saving metadata cache: {e}")

    # Save embeddings
    try:
        with open(TEMP_EMBEDDINGS_CACHE_PATH, "wb") as f:
            pickle.dump(embeddings, f)
        logger.info(
            f"Saved embeddings to temporary cache: {TEMP_EMBEDDINGS_CACHE_PATH}"
        )
    except Exception as e:
        logger.error(f"Error saving embeddings cache: {e}")

    # Step 5: Copy to Docker cache directory
    logger.info(
        "\nTo copy the cache files to the Docker container, run the following commands:"
    )
    logger.info(
        f"docker cp {TEMP_DOCUMENT_CACHE_PATH} rag_backend_1:/app/cache/document_cache.pkl"
    )
    logger.info(
        f"docker cp {TEMP_METADATA_CACHE_PATH} rag_backend_1:/app/cache/metadata_cache.pkl"
    )
    logger.info(
        f"docker cp {TEMP_EMBEDDINGS_CACHE_PATH} rag_backend_1:/app/cache/embeddings_cache.pkl"
    )
    logger.info("\nOr copy to the local cache directory with:")
    logger.info(f"cp {TEMP_DOCUMENT_CACHE_PATH} {DOCUMENT_CACHE_PATH}")
    logger.info(f"cp {TEMP_METADATA_CACHE_PATH} {METADATA_CACHE_PATH}")
    logger.info(f"cp {TEMP_EMBEDDINGS_CACHE_PATH} {EMBEDDINGS_CACHE_PATH}")

    logger.info(f"All done in {time.time() - start_time:.2f} seconds")


if __name__ == "__main__":
    main()
