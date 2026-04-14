"""
Document processing and search resource building utilities.

This module provides functions for:
1. Processing markdown documents and extracting content and metadata
2. Building and caching search resources for faster retrieval:
   - BM25 index for all documents
   - Document embeddings for vector search
   - Document metadata for quick access
   - Document content cache

Run this script after updating the document collection to rebuild the cache.
"""

import argparse
import logging
import os
import pickle
import time
from typing import Any, Dict, List, Optional, Tuple

import frontmatter
import numpy as np
from core.config import COLLECTION_NAME, EMBEDDING_MODEL, MARKDOWN_DIR, VECTOR_SIZE
from hybrid_search import (
    create_bm25_index,
    preprocess_text_for_bm25,
    read_markdown_files,
)
from sentence_transformers import SentenceTransformer
from tqdm import tqdm

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Constants
BM25_CACHE_PATH = "bm25_cache.pkl"
EMBEDDINGS_CACHE_PATH = "embeddings_cache.pkl"
METADATA_CACHE_PATH = "metadata_cache.pkl"
DOCUMENT_CACHE_PATH = "document_cache.pkl"


def process_markdown_file(file_path: str) -> Dict:
    """Process a single markdown file and extract content and metadata.

    Args:
        file_path: Path to the markdown file

    Returns:
        Dictionary containing content and metadata
    """
    with open(file_path, "r", encoding="utf-8") as f:
        post = frontmatter.load(f)

    # Extract metadata from frontmatter
    metadata = post.metadata
    content = post.content

    # Extract source from metadata if available
    source = ""
    for line in content.split("\n"):
        if line.startswith("*Source:"):
            source = line.replace("*Source:", "").strip()
            content = content.replace(line, "").strip()
            break

    # Extract title from first heading
    title = ""
    lines = content.split("\n")
    for line in lines:
        if line.startswith("# "):
            title = line.replace("# ", "").strip()
            content = content.replace(line, "").strip()
            break

    return {
        "content": content,
        "metadata": {
            "document_title": title,
            "context": title,
            "document_type": metadata.get("type", ""),
            "agency": metadata.get("agency", ""),
            "decision_number": metadata.get("decision_number", ""),
            "date": metadata.get("date", ""),
            "source": source or metadata.get("source", ""),
        },
    }


def process_all_documents(markdown_dir: str = MARKDOWN_DIR) -> List[Dict]:
    """Process all markdown files and return list of contents.

    Args:
        markdown_dir: Directory containing markdown files

    Returns:
        List of dictionaries containing content and metadata
    """
    all_contents = []

    for filename in os.listdir(markdown_dir):
        if not filename.endswith(".md"):
            continue

        file_path = os.path.join(markdown_dir, filename)
        try:
            content = process_markdown_file(file_path)
            all_contents.append(content)
        except Exception as e:
            logger.error(f"Error processing {filename}: {str(e)}")

    logger.info(f"Processed {len(all_contents)} documents successfully.")
    return all_contents


def build_bm25_index(
    markdown_dir: str, cache_path: str, force_rebuild: bool = False
) -> None:
    """Build and cache BM25 index.

    Args:
        markdown_dir: Directory containing markdown files
        cache_path: Path to save the cache
        force_rebuild: Whether to force rebuilding the index
    """
    start_time = time.time()

    # Check if cache exists and is valid
    if os.path.exists(cache_path) and not force_rebuild:
        cache_mtime = os.path.getmtime(cache_path)
        markdown_files = [
            os.path.join(markdown_dir, f)
            for f in os.listdir(markdown_dir)
            if f.endswith(".md")
        ]
        newest_markdown_mtime = (
            max([os.path.getmtime(f) for f in markdown_files]) if markdown_files else 0
        )

        if cache_mtime > newest_markdown_mtime:
            logger.info(f"BM25 cache is up to date: {cache_path}")
            return

    logger.info("Building BM25 index...")

    # Read markdown files
    documents = read_markdown_files(markdown_dir)
    logger.info(f"Read {len(documents)} documents from {markdown_dir}")

    # Create BM25 index
    bm25_model, corpus, doc_ids = create_bm25_index(documents)

    # Save to cache
    with open(cache_path, "wb") as f:
        pickle.dump((bm25_model, corpus, doc_ids), f)

    logger.info(
        f"BM25 index built and saved to {cache_path} in {time.time() - start_time:.2f} seconds"
    )


def build_document_embeddings(
    markdown_dir: str, model_path: str, cache_path: str, force_rebuild: bool = False
) -> None:
    """Build and cache document embeddings.

    Args:
        markdown_dir: Directory containing markdown files
        model_path: Path to the embedding model
        cache_path: Path to save the cache
        force_rebuild: Whether to force rebuilding the embeddings
    """
    start_time = time.time()

    # Check if cache exists and is valid
    if os.path.exists(cache_path) and not force_rebuild:
        cache_mtime = os.path.getmtime(cache_path)
        markdown_files = [
            os.path.join(markdown_dir, f)
            for f in os.listdir(markdown_dir)
            if f.endswith(".md")
        ]
        newest_markdown_mtime = (
            max([os.path.getmtime(f) for f in markdown_files]) if markdown_files else 0
        )

        if cache_mtime > newest_markdown_mtime:
            logger.info(f"Embeddings cache is up to date: {cache_path}")
            return

    logger.info("Building document embeddings...")

    # Load embedding model
    try:
        model = SentenceTransformer(model_path)
        logger.info(f"Loaded embedding model: {model_path}")
    except Exception as e:
        logger.error(f"Error loading embedding model: {e}")
        return

    # Read markdown files
    documents = read_markdown_files(markdown_dir)
    logger.info(f"Read {len(documents)} documents from {markdown_dir}")

    # Create embeddings
    document_embeddings = {}
    for doc_id, content in tqdm(documents.items(), desc="Creating embeddings"):
        try:
            embedding = model.encode(content, show_progress_bar=False)
            document_embeddings[doc_id] = embedding
        except Exception as e:
            logger.error(f"Error creating embedding for {doc_id}: {e}")

    # Save to cache
    with open(cache_path, "wb") as f:
        pickle.dump(document_embeddings, f)

    logger.info(
        f"Document embeddings built and saved to {cache_path} in {time.time() - start_time:.2f} seconds"
    )


def extract_document_metadata(
    markdown_dir: str, cache_path: str, force_rebuild: bool = False
) -> None:
    """Extract and cache document metadata.

    Args:
        markdown_dir: Directory containing markdown files
        cache_path: Path to save the cache
        force_rebuild: Whether to force rebuilding the metadata
    """
    start_time = time.time()

    # Check if cache exists and is valid
    if os.path.exists(cache_path) and not force_rebuild:
        cache_mtime = os.path.getmtime(cache_path)
        markdown_files = [
            os.path.join(markdown_dir, f)
            for f in os.listdir(markdown_dir)
            if f.endswith(".md")
        ]
        newest_markdown_mtime = (
            max([os.path.getmtime(f) for f in markdown_files]) if markdown_files else 0
        )

        if cache_mtime > newest_markdown_mtime:
            logger.info(f"Metadata cache is up to date: {cache_path}")
            return

    logger.info("Extracting document metadata...")

    # Read markdown files
    documents = read_markdown_files(markdown_dir)
    logger.info(f"Read {len(documents)} documents from {markdown_dir}")

    # Extract metadata
    metadata = {}
    for doc_id, content in tqdm(documents.items(), desc="Extracting metadata"):
        try:
            # Parse frontmatter
            with open(os.path.join(markdown_dir, doc_id), "r", encoding="utf-8") as f:
                post = frontmatter.load(f)

            # Extract metadata
            meta = post.metadata

            # Extract title from first heading
            title = ""
            lines = post.content.split("\n")
            for line in lines:
                if line.startswith("# "):
                    title = line.replace("# ", "").strip()
                    break

            # Extract source
            source = ""
            for line in lines:
                if line.startswith("*Source:"):
                    source = line.replace("*Source:", "").strip()
                    break

            # Store metadata
            metadata[doc_id] = {
                "document_title": title,
                "context": title,
                "document_type": meta.get("type", ""),
                "agency": meta.get("agency", ""),
                "decision_number": meta.get("decision_number", ""),
                "date": meta.get("date", ""),
                "source": source or meta.get("source", ""),
            }
        except Exception as e:
            logger.error(f"Error extracting metadata for {doc_id}: {e}")

    # Save to cache
    with open(cache_path, "wb") as f:
        pickle.dump(metadata, f)

    logger.info(
        f"Document metadata extracted and saved to {cache_path} in {time.time() - start_time:.2f} seconds"
    )


def build_document_cache(
    markdown_dir: str, cache_path: str, force_rebuild: bool = False
) -> None:
    """Build and cache document content.

    Args:
        markdown_dir: Directory containing markdown files
        cache_path: Path to save the cache
        force_rebuild: Whether to force rebuilding the cache
    """
    start_time = time.time()

    # Check if cache exists and is valid
    if os.path.exists(cache_path) and not force_rebuild:
        cache_mtime = os.path.getmtime(cache_path)
        markdown_files = [
            os.path.join(markdown_dir, f)
            for f in os.listdir(markdown_dir)
            if f.endswith(".md")
        ]
        newest_markdown_mtime = (
            max([os.path.getmtime(f) for f in markdown_files]) if markdown_files else 0
        )

        if cache_mtime > newest_markdown_mtime:
            logger.info(f"Document cache is up to date: {cache_path}")
            return

    logger.info("Building document cache...")

    # Read markdown files
    documents = read_markdown_files(markdown_dir)
    logger.info(f"Read {len(documents)} documents from {markdown_dir}")

    # Save to cache
    with open(cache_path, "wb") as f:
        pickle.dump(documents, f)

    logger.info(
        f"Document cache built and saved to {cache_path} in {time.time() - start_time:.2f} seconds"
    )


def build_all_caches(force_rebuild: bool = False) -> None:
    """Build all caches.

    Args:
        force_rebuild: Whether to force rebuilding all caches
    """
    start_time = time.time()
    logger.info("Starting to build all search resources...")

    build_bm25_index(MARKDOWN_DIR, BM25_CACHE_PATH, force_rebuild)
    build_document_embeddings(
        MARKDOWN_DIR, EMBEDDING_MODEL, EMBEDDINGS_CACHE_PATH, force_rebuild
    )
    extract_document_metadata(MARKDOWN_DIR, METADATA_CACHE_PATH, force_rebuild)
    build_document_cache(MARKDOWN_DIR, DOCUMENT_CACHE_PATH, force_rebuild)

    logger.info(f"All resources built in {time.time() - start_time:.2f} seconds")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Document processing and search resource building utilities"
    )
    parser.add_argument("--process", action="store_true", help="Process all documents")
    parser.add_argument("--force", action="store_true", help="Force rebuild all caches")
    parser.add_argument("--bm25", action="store_true", help="Build only BM25 index")
    parser.add_argument(
        "--embeddings", action="store_true", help="Build only document embeddings"
    )
    parser.add_argument(
        "--metadata", action="store_true", help="Build only document metadata"
    )
    parser.add_argument(
        "--documents", action="store_true", help="Build only document cache"
    )
    parser.add_argument("--all", action="store_true", help="Build all caches")

    args = parser.parse_args()

    if args.process:
        contents = process_all_documents()
        print(f"Processed {len(contents)} documents successfully.")
    elif args.bm25:
        build_bm25_index(MARKDOWN_DIR, BM25_CACHE_PATH, args.force)
    elif args.embeddings:
        build_document_embeddings(
            MARKDOWN_DIR, EMBEDDING_MODEL, EMBEDDINGS_CACHE_PATH, args.force
        )
    elif args.metadata:
        extract_document_metadata(MARKDOWN_DIR, METADATA_CACHE_PATH, args.force)
    elif args.documents:
        build_document_cache(MARKDOWN_DIR, DOCUMENT_CACHE_PATH, args.force)
    elif args.all or not any(
        [args.process, args.bm25, args.embeddings, args.metadata, args.documents]
    ):
        build_all_caches(args.force)
