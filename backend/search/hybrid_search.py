"""
Hybrid Search implementation combining vector search with BM25 for improved retrieval.

This module implements a hybrid search approach that combines:
1. Vector search using a fine-tuned embedding model
2. BM25 lexical search
3. Query expansion for improved recall

The implementation is optimized for legal document retrieval with parameters:
- alpha=0.4 (weight for vector search)
- query_expansion=True (enables query expansion)
"""

import logging
import os
import pickle
import re
import time
from typing import Any, Dict, List, Optional, Set, Tuple

import faiss
import numpy as np
from rank_bm25 import BM25Okapi
from sklearn.metrics.pairwise import cosine_similarity
from tqdm import tqdm

try:
    from backend.core.config import (
        EMBEDDING_MODEL,
        HYBRID_ALPHA,
        MARKDOWN_DIR,
        QUERY_EXPANSION,
        TOP_K,
        USE_FAISS,
    )
except ImportError:
    # Trong Docker container, không có thư mục 'backend' ở root
    from core.config import (
        EMBEDDING_MODEL,
        HYBRID_ALPHA,
        MARKDOWN_DIR,
        QUERY_EXPANSION,
        TOP_K,
        USE_FAISS,
    )

# Configure logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

# Constants
# Use absolute paths for cache files
CACHE_DIR = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "cache"
)
# Create cache directory if it doesn't exist
os.makedirs(CACHE_DIR, exist_ok=True)
# Log cache directory for debugging
logger.info(f"Using cache directory: {CACHE_DIR}")

BM25_CACHE_PATH = os.path.join(CACHE_DIR, "bm25_cache.pkl")
EMBEDDINGS_CACHE_PATH = os.path.join(CACHE_DIR, "embeddings_cache.pkl")
METADATA_CACHE_PATH = os.path.join(CACHE_DIR, "metadata_cache.pkl")
DOCUMENT_CACHE_PATH = os.path.join(CACHE_DIR, "document_cache.pkl")
FAISS_INDEX_PATH = os.path.join(CACHE_DIR, "faiss_index.bin")
FAISS_MAPPING_PATH = os.path.join(CACHE_DIR, "faiss_mapping.pkl")

# LEGAL_TERMS set
LEGAL_TERMS = set(
    [
        "luật",
        "điều",
        "khoản",
        "nghị định",
        "thông tư",
        "quyết định",
        "pháp lệnh",
        "hiến pháp",
        "bộ luật",
        "văn bản",
        "hành chính",
        "tư pháp",
        "tòa án",
        "viện kiểm sát",
        "thẩm phán",
        "kiểm sát viên",
        "luật sư",
        "công chứng",
        "thừa phát lại",
        "giám định",
        "hòa giải",
        "trọng tài",
        "khiếu nại",
        "tố cáo",
        "khởi kiện",
        "kháng cáo",
        "kháng nghị",
        "giám đốc thẩm",
        "tái thẩm",
        "thi hành án",
        "bồi thường",
        "bồi hoàn",
        "phạt",
        "phạt tiền",
        "phạt tù",
        "tù",
        "giam",
        "tạm giam",
        "tạm giữ",
        "quản chế",
        "cải tạo",
        "án treo",
        "tha tù",
        "đặc xá",
        "ân xá",
        "miễn trách nhiệm",
        "miễn tố",
        "miễn hình phạt",
        "giảm hình phạt",
        "hoãn thi hành",
        "tạm đình chỉ",
        "đình chỉ",
        "chấm dứt",
        "hủy bỏ",
        "thu hồi",
        "tịch thu",
        "tịch biên",
        "kê biên",
        "phong tỏa",
        "cấm",
        "cấm đoán",
        "cấm vận",
        "cấm xuất cảnh",
        "cấm nhập cảnh",
        "trục xuất",
        "dẫn độ",
        "tương trợ tư pháp",
        "ủy thác tư pháp",
        "rửa tiền",
        "tham nhũng",
        "hối lộ",
        "lạm quyền",
        "lợi dụng chức vụ",
        "lợi dụng quyền hạn",
        "lợi dụng ảnh hưởng",
        "lợi dụng tín nhiệm",
        "lừa đảo",
        "lạm dụng",
        "chiếm đoạt",
        "chiếm giữ",
        "cưỡng đoạt",
        "cưỡng chế",
        "ép buộc",
        "đe dọa",
        "khủng bố",
        "bạo lực",
        "xâm phạm",
        "xâm hại",
        "xâm nhập",
        "xâm chiếm",
        "xâm phạm an ninh quốc gia",
        "xâm phạm chủ quyền",
        "xâm phạm lãnh thổ",
        "xâm phạm danh dự",
        "xâm phạm nhân phẩm",
        "xâm phạm thân thể",
        "xâm phạm sức khỏe",
        "xâm phạm tính mạng",
        "xâm phạm tài sản",
        "xâm phạm quyền sở hữu",
        "xâm phạm quyền tự do",
        "xâm phạm quyền dân chủ",
        "xâm phạm quyền con người",
        "xâm phạm quyền công dân",
        "xâm phạm quyền bình đẳng",
    ]
)

# Vietnamese stopwords
VI_STOP_WORDS = {
    "bị",
    "bởi",
    "cả",
    "các",
    "cái",
    "cần",
    "càng",
    "chỉ",
    "chiếc",
    "cho",
    "chứ",
    "chưa",
    "chuyện",
    "có",
    "có thể",
    "cứ",
    "của",
    "cùng",
    "cũng",
    "đã",
    "đang",
    "đây",
    "để",
    "đến nỗi",
    "đều",
    "điều",
    "do",
    "đó",
    "được",
    "dưới",
    "gì",
    "khi",
    "không",
    "là",
    "lại",
    "lên",
    "lúc",
    "mà",
    "mỗi",
    "một cách",
    "này",
    "nên",
    "nếu",
    "ngay",
    "nhiều",
    "như",
    "nhưng",
    "những",
    "nơi",
    "nữa",
    "phải",
    "qua",
    "ra",
    "rằng",
    "rất",
    "rồi",
    "sau",
    "sẽ",
    "so",
    "sự",
    "tại",
    "theo",
    "thì",
    "trên",
    "trước",
    "từ",
    "từng",
    "và",
    "vẫn",
    "vào",
    "vậy",
    "vì",
    "việc",
    "với",
    "vừa",
}


def preprocess_text_for_bm25(text: str) -> List[str]:
    """
    Tiền xử lý văn bản cho BM25 với hỗ trợ tiếng Việt.

    Args:
        text: Văn bản cần tiền xử lý

    Returns:
        Danh sách các token đã tiền xử lý
    """
    if not text or text.strip() == "":
        return []

    # Chuyển thành chữ thường
    text = text.lower()

    # Xử lý tiếng Việt
    try:
        # Sử dụng pyvi để tokenize tiếng Việt
        from pyvi import ViTokenizer

        tokens = ViTokenizer.tokenize(text).split()

        # Thêm bigrams cho các cụm từ pháp lý quan trọng
        bigrams = [f"{tokens[i]}_{tokens[i+1]}" for i in range(len(tokens) - 1)]
        tokens.extend(bigrams)

        # Loại bỏ stopwords nhưng giữ lại các thuật ngữ pháp lý quan trọng
        tokens = [
            token
            for token in tokens
            if token not in VI_STOP_WORDS or token in LEGAL_TERMS
        ]
    except Exception as e:
        logger.warning(f"Lỗi trong quá trình tiền xử lý tiếng Việt: {e}")
        tokens = text.split()

    # Chỉ loại bỏ các token quá ngắn và không phải số
    tokens = [token for token in tokens if len(token) > 1 or token.isdigit()]

    return tokens


def read_markdown_files(directory: str) -> Dict[str, str]:
    """
    Read all markdown files from a directory.

    Args:
        directory: Directory containing markdown files

    Returns:
        Dictionary mapping file paths to file contents
    """
    markdown_files = {}

    for root, _, files in os.walk(directory):
        for file in files:
            if file.endswith(".md"):
                file_path = os.path.join(root, file)
                try:
                    with open(file_path, "r", encoding="utf-8") as f:
                        content = f.read()

                    # Use just the filename as document ID for consistency
                    # This ensures IDs are consistent across different functions
                    doc_id = file
                    markdown_files[doc_id] = content

                    # Also add the relative path as an alternative ID for backward compatibility
                    rel_path = os.path.relpath(file_path, directory)
                    if rel_path != file and rel_path not in markdown_files:
                        markdown_files[rel_path] = content
                        logger.debug(
                            f"Added alternative ID {rel_path} for document {file}"
                        )
                except Exception as e:
                    logger.error(f"Error reading {file_path}: {e}")

    logger.info(f"Read {len(markdown_files)} markdown files from {directory}")
    return markdown_files


def chunk_documents(
    documents: Dict[str, str], chunk_size: int = 1000, overlap: int = 200
) -> Dict[str, str]:
    """
    Split documents into smaller chunks for better retrieval.

    Args:
        documents: Dictionary mapping document IDs to document contents
        chunk_size: Maximum chunk size in characters
        overlap: Overlap between chunks in characters

    Returns:
        Dictionary mapping chunk IDs to chunk contents
    """
    chunks = {}

    for doc_id, content in documents.items():
        if len(content) <= chunk_size:
            # Document is small enough, no need to chunk
            chunks[doc_id] = content
        else:
            # Split document into chunks
            for i in range(0, len(content), chunk_size - overlap):
                chunk_content = content[i : i + chunk_size]
                if len(chunk_content) < 100:  # Skip very small chunks
                    continue
                chunk_id = f"{doc_id}#chunk{i // (chunk_size - overlap)}"
                chunks[chunk_id] = chunk_content

    return chunks


def create_bm25_index(
    documents: Dict[str, str],
) -> Tuple[BM25Okapi, Dict[str, List[str]], List[str]]:
    """
    Create a BM25 index from documents.

    Args:
        documents: Dictionary mapping document IDs to document contents

    Returns:
        Tuple of (BM25 model, corpus dictionary, document IDs list)
    """
    corpus = {}
    doc_ids = []

    # Preprocess documents
    for doc_id, content in documents.items():
        tokens = preprocess_text_for_bm25(content)
        if tokens:  # Only add documents with tokens
            corpus[doc_id] = tokens
            doc_ids.append(doc_id)

    # Create BM25 index
    tokenized_corpus = [corpus[doc_id] for doc_id in doc_ids]
    bm25 = BM25Okapi(tokenized_corpus)

    return bm25, corpus, doc_ids


def hybrid_search(
    query: str,
    document_embeddings: Dict[str, List[float]],
    bm25_model: BM25Okapi,
    doc_ids: List[str],
    model: Any = None,
    corpus: Dict[str, List[str]] = None,
    alpha: float = 0.3,
    k: int = 5,
    query_expansion: bool = True,
    query_embedding: List[float] = None,
    use_faiss: bool = USE_FAISS,
) -> List[Tuple[str, float]]:
    """
    Perform hybrid search combining vector search with BM25.

    Args:
        query: Query string
        document_embeddings: Dictionary mapping document IDs to document embeddings
        bm25_model: BM25 model
        doc_ids: List of document IDs
        model: Embedding model (optional if query_embedding is provided)
        corpus: Dictionary mapping document IDs to tokenized documents
        alpha: Weight for vector search (0-1)
        k: Number of results to return
        query_expansion: Whether to use query expansion
        query_embedding: Pre-computed query embedding (optional)
        use_faiss: Whether to use FAISS for vector search if available

    Returns:
        List of (document_id, score) tuples
    """
    # STEP 1: Analyze legal query structure
    is_legal_query = any(
        term in query.lower()
        for term in [
            "luật",
            "điều",
            "khoản",
            "quy định",
            "nghị định",
            "thông tư",
        ]
    )

    # Adjust alpha dynamically based on query type
    dynamic_alpha = (
        0.3 if is_legal_query else alpha
    )  # Prioritize BM25 for legal queries

    # STEP 2: Process original query
    if query_embedding is None and model is not None:
        question_embedding = model.encode(query, show_progress_bar=False)
    else:
        question_embedding = query_embedding

    tokenized_query = preprocess_text_for_bm25(query)

    # STEP 3: Query expansion
    expanded_terms = []
    if query_expansion and corpus:  # Chỉ thực hiện query expansion nếu có corpus
        # Kiểm tra xem có thể thực hiện query expansion không
        can_expand = True

        # Nếu không có model và query_expansion=True, ghi log cảnh báo
        if model is None:
            logger.warning(
                "Query expansion requested but no model provided. Using basic expansion."
            )
            # Vẫn có thể thực hiện query expansion đơn giản dựa trên corpus

        # Find initial top documents using vector search
        initial_results = {}

        # Try to use FAISS for initial vector search if available
        if use_faiss:
            try:
                faiss_index, faiss_doc_ids = load_faiss_index()
                if faiss_index is not None and faiss_doc_ids is not None:
                    # Use FAISS for initial search
                    faiss_results = search_faiss_index(
                        question_embedding, faiss_index, faiss_doc_ids, k=3
                    )
                    for doc_id, score in faiss_results:
                        initial_results[doc_id] = score
                    logger.info("Used FAISS for query expansion initial search")
                else:
                    # Fall back to cosine similarity
                    for doc_id, doc_emb in document_embeddings.items():
                        sim = cosine_similarity([question_embedding], [doc_emb])[0][0]
                        initial_results[doc_id] = sim
            except Exception as e:
                logger.warning(f"Error using FAISS for query expansion: {e}")
                # Fall back to cosine similarity
                for doc_id, doc_emb in document_embeddings.items():
                    sim = cosine_similarity([question_embedding], [doc_emb])[0][0]
                    initial_results[doc_id] = sim
        else:
            # Use cosine similarity
            for doc_id, doc_emb in document_embeddings.items():
                sim = cosine_similarity([question_embedding], [doc_emb])[0][0]
                initial_results[doc_id] = sim

        # Get top 3 documents to extract keywords
        top_docs = sorted(initial_results.items(), key=lambda x: x[1], reverse=True)[:3]

        # Extract keywords with higher weights for legal terms
        term_freq = {}
        legal_term_boost = 1.5  # Boost factor for legal terms
        legal_terms = ["luật", "điều", "khoản", "nghị", "định", "quyết", "quy", "pháp"]

        for doc_id, _ in top_docs:
            if doc_id in corpus:
                for term in corpus[doc_id]:
                    if (
                        term not in tokenized_query
                    ):  # Only add terms not in original query
                        # Boost legal terms
                        boost = (
                            legal_term_boost
                            if any(legal_term in term for legal_term in legal_terms)
                            else 1.0
                        )
                        term_freq[term] = term_freq.get(term, 0) + boost

        # Get highest frequency terms
        expanded_terms = sorted(term_freq.items(), key=lambda x: x[1], reverse=True)[:5]
        expanded_terms = [term for term, _ in expanded_terms]

        # Expand query with expanded_terms
        expanded_query = tokenized_query + expanded_terms
    else:
        expanded_query = tokenized_query

    # STEP 4: BM25 search
    bm25_scores = bm25_model.get_scores(expanded_query)

    # Normalize BM25 scores
    max_bm25_score = max(bm25_scores) if bm25_scores.any() else 1.0
    normalized_bm25_scores = {
        doc_ids[i]: bm25_scores[i] / max_bm25_score for i in range(len(doc_ids))
    }

    # STEP 5: Vector search
    vector_scores = {}

    # Try to use FAISS for vector search if available
    if use_faiss:
        try:
            faiss_index, faiss_doc_ids = load_faiss_index()
            if faiss_index is not None and faiss_doc_ids is not None:
                # Use FAISS for vector search
                faiss_results = search_faiss_index(
                    question_embedding, faiss_index, faiss_doc_ids, k=len(doc_ids)
                )
                for doc_id, score in faiss_results:
                    vector_scores[doc_id] = score
                logger.info("Used FAISS for vector search")
            else:
                # Fall back to cosine similarity
                for doc_id, doc_emb in document_embeddings.items():
                    sim = cosine_similarity([question_embedding], [doc_emb])[0][0]
                    vector_scores[doc_id] = sim
        except Exception as e:
            logger.warning(f"Error using FAISS for vector search: {e}")
            # Fall back to cosine similarity
            for doc_id, doc_emb in document_embeddings.items():
                sim = cosine_similarity([question_embedding], [doc_emb])[0][0]
                vector_scores[doc_id] = sim
    else:
        # Use cosine similarity
        for doc_id, doc_emb in document_embeddings.items():
            sim = cosine_similarity([question_embedding], [doc_emb])[0][0]
            vector_scores[doc_id] = sim

    # STEP 6: Count legal term matches in each document
    legal_match_scores = {}
    legal_terms_in_query = [
        term
        for term in expanded_query
        if any(
            legal_term in term
            for legal_term in ["luật", "điều", "khoản", "nghị", "định", "quyết"]
        )
    ]

    for i, doc_id in enumerate(doc_ids):
        if doc_id in corpus:
            doc_tokens = corpus[doc_id]
            legal_matches = sum(
                1 for term in legal_terms_in_query if term in doc_tokens
            )
            legal_match_scores[doc_id] = min(
                legal_matches / max(1, len(legal_terms_in_query)), 1.0
            )

    # STEP 7: Combine scores with dynamic weighting
    hybrid_scores = {}
    for doc_id in doc_ids:
        if doc_id in document_embeddings:
            # Get individual scores
            vector_score = vector_scores.get(doc_id, 0)
            bm25_score = normalized_bm25_scores.get(doc_id, 0)
            legal_score = legal_match_scores.get(doc_id, 0)

            # Combine scores with dynamic weighting
            hybrid_score = (
                dynamic_alpha * vector_score
                + (1 - dynamic_alpha) * bm25_score
                + (0.1 * legal_score if is_legal_query else 0)
            )

            hybrid_scores[doc_id] = hybrid_score

    # STEP 8: Reranking with priority for high-precision documents
    preliminary_results = sorted(
        hybrid_scores.items(), key=lambda x: x[1], reverse=True
    )[: min(k * 2, len(hybrid_scores))]

    # Calculate structure and content precision scores
    reranked_results = []
    for doc_id, score in preliminary_results:
        # Analyze document structure match
        structure_match = 0
        if doc_id in corpus and is_legal_query:
            # Check if document contains legal terms with similar structure
            doc_text = " ".join(corpus[doc_id])
            structure_match = (
                0.1
                if any(
                    term in doc_text for term in ["điều", "khoản", "luật", "nghị định"]
                )
                else 0
            )

        # Analyze keyword density
        content_match = 0
        if doc_id in corpus:
            # Count occurrences of important query terms
            important_terms = set(
                tokenized_query
            )  # Original query terms are more important
            doc_tokens = corpus[doc_id]

            # Calculate keyword density relative to document length
            matches = sum(doc_tokens.count(term) for term in important_terms)
            content_match = min(
                matches / max(1, len(doc_tokens)) * 2, 0.3
            )  # Limit impact

        # Combine original score with reranking scores
        final_score = score * 0.7 + content_match + structure_match
        reranked_results.append((doc_id, final_score))

    # Sort by final score
    sorted_results = sorted(reranked_results, key=lambda x: x[1], reverse=True)

    # Lọc các kết quả dựa trên độ liên quan
    if len(sorted_results) > 1:
        # Lấy điểm cao nhất
        top_score = sorted_results[0][1]

        # Tính ngưỡng điểm (50% của điểm cao nhất)
        threshold = top_score * 0.5

        # Lọc các kết quả có điểm cao hơn ngưỡng
        filtered_results = [
            (doc_id, score) for doc_id, score in sorted_results if score >= threshold
        ]

        # Đảm bảo có ít nhất 3 kết quả (nếu có thể)
        if len(filtered_results) < 3 and len(sorted_results) >= 3:
            filtered_results = sorted_results[:3]

        # Giới hạn số lượng kết quả tối đa
        if len(filtered_results) > k:
            filtered_results = filtered_results[:k]

        return filtered_results

    # Nếu chỉ có 1 hoặc 0 kết quả, trả về tất cả
    return sorted_results[:k]


def load_or_create_bm25_index(
    documents: Dict[str, str], cache_path: str = BM25_CACHE_PATH
) -> Tuple[BM25Okapi, Dict[str, List[str]], List[str]]:
    """
    Load BM25 index from cache or create a new one.

    Args:
        documents: Dictionary mapping document IDs to document contents
        cache_path: Path to cache file

    Returns:
        Tuple of (BM25 model, corpus dictionary, document IDs list)
    """
    # Check if cache exists and is valid
    if os.path.exists(cache_path):
        try:
            with open(cache_path, "rb") as f:
                cache = pickle.load(f)

            # Verify cache is valid for current documents
            if set(cache["doc_ids"]) == set(documents.keys()):
                logger.info(f"Loaded BM25 index from cache: {cache_path}")
                return cache["bm25"], cache["corpus"], cache["doc_ids"]
        except Exception as e:
            logger.warning(f"Error loading BM25 cache: {e}")

    # Create new index
    logger.info("Creating new BM25 index...")
    bm25, corpus, doc_ids = create_bm25_index(documents)

    # Save to cache
    try:
        with open(cache_path, "wb") as f:
            pickle.dump({"bm25": bm25, "corpus": corpus, "doc_ids": doc_ids}, f)
        logger.info(f"Saved BM25 index to cache: {cache_path}")
    except Exception as e:
        logger.warning(f"Error saving BM25 cache: {e}")

    return bm25, corpus, doc_ids


def load_or_create_embeddings(
    documents: Dict[str, str],
    model: Any,
    cache_path: str = EMBEDDINGS_CACHE_PATH,
    metadata_path: str = METADATA_CACHE_PATH,
) -> Dict[str, List[float]]:
    """
    Load document embeddings from cache or create new ones.

    Args:
        documents: Dictionary mapping document IDs to document contents
        model: Embedding model
        cache_path: Path to embeddings cache file
        metadata_path: Path to metadata cache file

    Returns:
        Dictionary mapping document IDs to document embeddings
    """
    # Check if cache exists and is valid
    if os.path.exists(cache_path) and os.path.exists(metadata_path):
        try:
            with open(metadata_path, "rb") as f:
                metadata = pickle.load(f)

            # Verify cache is valid for current documents and model
            if (
                set(metadata["doc_ids"]) == set(documents.keys())
                and metadata["model_name"] == model.__class__.__name__
            ):
                with open(cache_path, "rb") as f:
                    embeddings = pickle.load(f)
                logger.info(f"Loaded embeddings from cache: {cache_path}")
                return embeddings
        except Exception as e:
            logger.warning(f"Error loading embeddings cache: {e}")

    # Create new embeddings
    logger.info("Creating new document embeddings...")
    embeddings = {}

    # Process documents in batches to avoid memory issues
    batch_size = 32
    doc_ids = list(documents.keys())

    for i in range(0, len(doc_ids), batch_size):
        batch_ids = doc_ids[i : i + batch_size]
        batch_texts = [documents[doc_id] for doc_id in batch_ids]

        # Encode batch
        batch_embeddings = model.encode(
            batch_texts, show_progress_bar=True, convert_to_tensor=False
        )

        # Store embeddings
        for j, doc_id in enumerate(batch_ids):
            embeddings[doc_id] = batch_embeddings[j].tolist()

    # Save to cache
    try:
        with open(cache_path, "wb") as f:
            pickle.dump(embeddings, f)

        with open(metadata_path, "wb") as f:
            pickle.dump(
                {
                    "doc_ids": list(documents.keys()),
                    "model_name": model.__class__.__name__,
                },
                f,
            )

        logger.info(f"Saved embeddings to cache: {cache_path}")
    except Exception as e:
        logger.warning(f"Error saving embeddings cache: {e}")

    return embeddings


def create_faiss_index(
    embeddings: Dict[str, List[float]],
    index_path: str = FAISS_INDEX_PATH,
    mapping_path: str = FAISS_MAPPING_PATH,
) -> bool:
    """
    Create a FAISS index from document embeddings and save it to disk.
    Uses IndexFlatIP (inner product) with normalized vectors to maintain cosine similarity.

    Args:
        embeddings: Dictionary mapping document IDs to document embeddings
        index_path: Path to save the FAISS index
        mapping_path: Path to save the ID mapping

    Returns:
        True if successful, False otherwise
    """
    if not embeddings:
        logger.error("No embeddings provided to create FAISS index")
        return False

    try:
        # Convert embeddings to numpy array
        doc_ids = list(embeddings.keys())
        embedding_list = [
            np.array(embeddings[doc_id], dtype=np.float32) for doc_id in doc_ids
        ]
        embedding_matrix = np.vstack(embedding_list)

        # Get dimensionality of embeddings
        dimension = embedding_matrix.shape[1]

        # Normalize vectors to unit length (L2 norm)
        # This ensures that dot product = cosine similarity
        # Use numpy for normalization to ensure exact matching with sklearn's cosine_similarity
        norms = np.linalg.norm(embedding_matrix, axis=1, keepdims=True)
        normalized_matrix = embedding_matrix / norms

        # Create FAISS index using inner product (dot product)
        # For normalized vectors, inner product is equivalent to cosine similarity
        index = faiss.IndexFlatIP(dimension)

        # Add normalized vectors to the index
        index.add(normalized_matrix)

        # Save the index to disk
        faiss.write_index(index, index_path)

        # Save the mapping of FAISS index positions to document IDs
        with open(mapping_path, "wb") as f:
            pickle.dump(doc_ids, f)

        logger.info(
            f"Created FAISS index with {len(doc_ids)} normalized vectors and dimension {dimension}"
        )
        logger.info(f"Using IndexFlatIP for cosine similarity")
        logger.info(f"Saved FAISS index to: {index_path}")
        logger.info(f"Saved ID mapping to: {mapping_path}")

        return True

    except Exception as e:
        logger.error(f"Error creating FAISS index: {e}")
        return False


def load_faiss_index(
    index_path: str = FAISS_INDEX_PATH,
    mapping_path: str = FAISS_MAPPING_PATH,
) -> Tuple[Optional[Any], Optional[List[str]]]:
    """
    Load a FAISS index and ID mapping from disk.

    Args:
        index_path: Path to the FAISS index
        mapping_path: Path to the ID mapping

    Returns:
        Tuple of (FAISS index, document ID list) or (None, None) if loading fails
    """
    try:
        if not os.path.exists(index_path) or not os.path.exists(mapping_path):
            logger.warning(
                f"FAISS index or mapping file not found: {index_path}, {mapping_path}"
            )
            return None, None

        # Load the FAISS index
        index = faiss.read_index(index_path)

        # Load the document ID mapping
        with open(mapping_path, "rb") as f:
            doc_ids = pickle.load(f)

        logger.info(f"Loaded FAISS index with {index.ntotal} vectors")
        return index, doc_ids

    except Exception as e:
        logger.error(f"Error loading FAISS index: {e}")
        return None, None


def search_faiss_index(
    query_embedding: np.ndarray,
    index: Any,
    doc_ids: List[str],
    k: int = 5,
) -> List[Tuple[str, float]]:
    """
    Search a FAISS index for the nearest neighbors to a query embedding.
    Handles both L2 distance and inner product (cosine similarity) indices.

    Args:
        query_embedding: Query embedding vector
        index: FAISS index
        doc_ids: List of document IDs corresponding to index positions
        k: Number of results to return

    Returns:
        List of (document_id, similarity_score) tuples
    """
    try:
        # Ensure query embedding is a numpy array with the right shape and type
        if isinstance(query_embedding, list):
            query_embedding = np.array(query_embedding, dtype=np.float32)

        if len(query_embedding.shape) == 1:
            query_embedding = query_embedding.reshape(1, -1)

        # Convert to float32 if needed
        if query_embedding.dtype != np.float32:
            query_embedding = query_embedding.astype(np.float32)

        # Check if this is an inner product index (IndexFlatIP)
        is_ip_index = isinstance(index, faiss.IndexFlatIP) or (
            hasattr(index, "metric_type")
            and index.metric_type == faiss.METRIC_INNER_PRODUCT
        )

        # For inner product indices, normalize the query vector
        if is_ip_index:
            # Normalize the query vector to unit length for cosine similarity
            # Use numpy for normalization to ensure exact matching with sklearn's cosine_similarity
            norm = np.linalg.norm(query_embedding)
            if norm > 0:
                query_embedding = query_embedding / norm

        # Search the index
        distances, indices = index.search(query_embedding, k)

        # Convert to list of (doc_id, score) tuples
        results = []
        for i in range(len(indices[0])):
            idx = indices[0][i]
            distance = distances[0][i]

            # Skip invalid indices
            if idx < 0 or idx >= len(doc_ids):
                continue

            doc_id = doc_ids[idx]

            # Handle similarity score based on index type
            if is_ip_index:
                # For inner product with normalized vectors, the distance is already cosine similarity
                # For exact matching with sklearn's cosine_similarity, we don't need to adjust
                # Just use the raw inner product value as the similarity score
                similarity = distance
            else:
                # For L2 distance, convert to similarity score
                # using 1/(1+distance) which maps [0, inf) to (0, 1]
                similarity = 1.0 / (1.0 + distance)

            results.append((doc_id, similarity))

        return results

    except Exception as e:
        logger.error(f"Error searching FAISS index: {e}")
        return []


def load_or_create_document_cache(
    directory: str, cache_path: str = DOCUMENT_CACHE_PATH
) -> Dict[str, str]:
    """
    Load documents from cache or read from directory.

    Args:
        directory: Directory containing markdown files
        cache_path: Path to document cache file

    Returns:
        Dictionary mapping document IDs to document contents
    """
    # Check if cache exists
    if os.path.exists(cache_path):
        try:
            with open(cache_path, "rb") as f:
                documents = pickle.load(f)
            logger.info(f"Loaded documents from cache: {cache_path}")
            return documents
        except Exception as e:
            logger.warning(f"Error loading document cache: {e}")

    # Read documents from directory
    logger.info(f"Reading documents from directory: {directory}")
    documents = read_markdown_files(directory)

    # Save to cache
    try:
        with open(cache_path, "wb") as f:
            pickle.dump(documents, f)
        logger.info(f"Saved documents to cache: {cache_path}")
    except Exception as e:
        logger.warning(f"Error saving document cache: {e}")

    return documents


# Helper functions for test script
def load_cached_documents(cache_path: str = DOCUMENT_CACHE_PATH) -> Dict[str, str]:
    """
    Load documents from cache.

    Args:
        cache_path: Path to document cache file

    Returns:
        Dictionary mapping document IDs to document contents
    """
    if os.path.exists(cache_path):
        try:
            with open(cache_path, "rb") as f:
                documents = pickle.load(f)
            logger.info(f"Loaded documents from cache: {cache_path}")
            return documents
        except Exception as e:
            logger.warning(f"Error loading document cache: {e}")
            return {}
    else:
        logger.warning(f"Document cache not found: {cache_path}")
        return {}


def load_cached_embeddings(
    cache_path: str = EMBEDDINGS_CACHE_PATH,
) -> Dict[str, List[float]]:
    """
    Load embeddings from cache.

    Args:
        cache_path: Path to embeddings cache file

    Returns:
        Dictionary mapping document IDs to document embeddings
    """
    if os.path.exists(cache_path):
        try:
            with open(cache_path, "rb") as f:
                embeddings = pickle.load(f)
            logger.info(f"Loaded embeddings from cache: {cache_path}")
            return embeddings
        except ImportError as e:
            # Handle numpy version incompatibility
            logger.warning(
                f"Error loading embeddings cache due to version incompatibility: {e}"
            )
            logger.warning("Creating new embeddings cache file...")
            # Return empty dict to trigger recreation of embeddings
            return {}
        except Exception as e:
            logger.warning(f"Error loading embeddings cache: {e}")
            return {}
    else:
        logger.warning(f"Embeddings cache not found: {cache_path}")
        return {}


def load_cached_metadata(cache_path: str = METADATA_CACHE_PATH) -> Dict[str, Any]:
    """
    Load metadata from cache.

    Args:
        cache_path: Path to metadata cache file

    Returns:
        Dictionary containing metadata
    """
    if os.path.exists(cache_path):
        try:
            with open(cache_path, "rb") as f:
                metadata = pickle.load(f)
            logger.info(f"Loaded metadata from cache: {cache_path}")
            return metadata
        except Exception as e:
            logger.warning(f"Error loading metadata cache: {e}")
            return {}
    else:
        logger.warning(f"Metadata cache not found: {cache_path}")
        return {}


def format_hybrid_results(hybrid_results, documents, doc_metadata):
    """Format hybrid search results for use in the RAG pipeline.

    Args:
        hybrid_results: List of (document_id, score) tuples from hybrid_search
        documents: Dictionary mapping document IDs to document content
        doc_metadata: Dictionary mapping document IDs to metadata

    Returns:
        List of dictionaries with content and metadata
    """
    formatted_results = []

    # Log information for debugging
    logger.info(f"Formatting {len(hybrid_results)} hybrid search results")
    logger.info(f"Documents dictionary has {len(documents)} entries")
    logger.info(f"Metadata dictionary has {len(doc_metadata)} entries")

    if len(hybrid_results) == 0:
        logger.warning("No hybrid search results to format")
        return []

    # Log a sample of document IDs for debugging
    if len(documents) > 0:
        sample_doc_ids = list(documents.keys())[:5]
        logger.info(f"Sample document IDs: {sample_doc_ids}")

    # Log a sample of hybrid results for debugging
    sample_results = hybrid_results[:5]
    logger.info(f"Sample hybrid results: {sample_results}")

    # Create a mapping of all possible document ID variations
    id_mapping = {}
    for doc_id in documents.keys():
        # Original ID
        id_mapping[doc_id] = doc_id

        # Filename only (if it's a path)
        if os.path.sep in doc_id:
            filename = os.path.basename(doc_id)
            id_mapping[filename] = doc_id

        # With/without .md extension
        if doc_id.endswith(".md"):
            id_mapping[doc_id[:-3]] = doc_id
        else:
            id_mapping[doc_id + ".md"] = doc_id

    logger.info(f"Created ID mapping with {len(id_mapping)} entries")

    for doc_id, score in hybrid_results:
        original_doc_id = doc_id
        actual_doc_id = id_mapping.get(doc_id, doc_id)

        # Get document content
        content = documents.get(actual_doc_id, "")

        # If content is empty, try to find the document with a similar ID
        if not content:
            # Try all possible variations
            for possible_id in documents.keys():
                # Check if the document ID contains the original ID or vice versa
                if original_doc_id in possible_id or possible_id in original_doc_id:
                    content = documents.get(possible_id, "")
                    if content:
                        logger.info(
                            f"Found content using similar ID: {possible_id} for {original_doc_id}"
                        )
                        actual_doc_id = possible_id
                        break

            # If still no content, log warning
            if not content:
                logger.warning(f"No content found for document ID: {original_doc_id}")
                continue  # Skip this document

        # Get metadata
        metadata = doc_metadata.get(actual_doc_id, {})

        # If metadata is empty, try to find metadata with a similar ID
        if not metadata:
            # Try all possible variations
            for possible_id in doc_metadata.keys():
                # Check if the document ID contains the original ID or vice versa
                if original_doc_id in possible_id or possible_id in original_doc_id:
                    metadata = doc_metadata.get(possible_id, {})
                    if metadata:
                        logger.info(
                            f"Found metadata using similar ID: {possible_id} for {original_doc_id}"
                        )
                        break

            # If still no metadata, log warning and create default metadata from content
            if not metadata:
                logger.warning(
                    f"No metadata found for document ID: {original_doc_id}, creating default metadata"
                )
                # Extract title from content (first line that starts with #)
                lines = content.split("\n")
                document_title = ""
                for line in lines:
                    if line.strip().startswith("#"):
                        document_title = line.strip().replace("#", "").strip()
                        break

                # If no title found, use first 50 characters of content
                if not document_title and content:
                    document_title = (
                        content[:50] + "..." if len(content) > 50 else content
                    )

                # Create default metadata
                metadata = {
                    "document_title": document_title,
                    "context": "",
                    "agency": "",
                    "decision_number": "",
                    "date": "",
                    "source": f"Document ID: {original_doc_id}",
                }

        # Format result
        formatted_doc = {
            "content": content,
            "metadata": {
                "document_title": metadata.get("document_title", ""),
                "context": metadata.get("context", ""),
                "agency": metadata.get("agency", ""),
                "decision_number": metadata.get("decision_number", ""),
                "date": metadata.get("date", ""),
                "source": metadata.get("source", ""),
            },
            "score": score,
        }

        formatted_results.append(formatted_doc)

    logger.info(f"Formatted {len(formatted_results)} results")
    return formatted_results


def get_embedding(text: str, model=None) -> List[float]:
    """Get embedding for text using sentence-transformers.

    Args:
        text: The text to embed
        model: The SentenceTransformer model to use

    Returns:
        List of floats representing the embedding
    """
    if model is None:
        from sentence_transformers import SentenceTransformer

        model = SentenceTransformer(EMBEDDING_MODEL)

    return model.encode(text, show_progress_bar=False).tolist()


def query_hybrid_search(
    query: str, top_k: int = TOP_K, use_faiss: bool = USE_FAISS
) -> List[Dict]:
    """Query using hybrid search.

    Args:
        query: The search query
        top_k: Number of results to return
        use_faiss: Whether to use FAISS for vector search if available

    Returns:
        List of dictionaries with content and metadata
    """
    try:
        # Load cached documents
        documents = load_cached_documents()
        if not documents:
            logger.error("No cached documents found")
            return []

        # Load BM25 resources
        bm25_model, corpus, doc_ids = load_or_create_bm25_index(documents)
        logger.info(f"BM25 resources loaded: {len(doc_ids)} documents in corpus")

        # Load cached embeddings
        document_embeddings = load_cached_embeddings()
        if not document_embeddings:
            logger.error("No cached embeddings found")
            return []

        # Load cached metadata
        doc_metadata = load_cached_metadata()
        if not doc_metadata:
            logger.error("No cached metadata found")
            return []

        logger.info(f"Loaded {len(document_embeddings)} document embeddings")

        # Check if FAISS index is available
        if use_faiss:
            faiss_index, faiss_doc_ids = load_faiss_index()
            if faiss_index is not None:
                logger.info(f"FAISS index loaded with {faiss_index.ntotal} vectors")
            else:
                logger.warning(
                    "FAISS index not found, falling back to cosine similarity"
                )

        # Load model
        from sentence_transformers import SentenceTransformer

        model = SentenceTransformer(EMBEDDING_MODEL)

        # Perform hybrid search
        hybrid_results = hybrid_search(
            query=query,
            document_embeddings=document_embeddings,
            bm25_model=bm25_model,
            doc_ids=doc_ids,
            model=model,
            corpus=corpus,
            alpha=HYBRID_ALPHA,
            k=top_k,
            query_expansion=QUERY_EXPANSION,
            use_faiss=use_faiss,
        )

        # Format results
        formatted_results = format_hybrid_results(
            hybrid_results, documents, doc_metadata
        )

        return formatted_results

    except Exception as e:
        logger.error(f"Error in query_hybrid_search: {e}")
        return []


def format_answer(results: Dict) -> str:
    """Format the answer for display."""
    return results["answer"]


def format_sources(results: Dict) -> str:
    """Format the sources for display."""
    # Check if we need to display sources
    query = results.get("query", "").lower()

    # List of simple queries that don't need sources
    simple_queries = [
        "xin chào",
        "chào",
        "hi",
        "hello",
        "hey",
        "hỏi",
        "bạn là ai",
        "bạn có thể làm gì",
        "giúp tôi",
        "cảm ơn",
        "tạm biệt",
        "bye",
        "tôi cần giúp đỡ",
        "giúp đỡ",
        "help",
        "hihihi",
        "haha",
        "ok",
        "okay",
        "yes",
        "no",
        "có",
        "không",
        "tốt",
        "rất tốt",
        "tuyệt vời",
        "cảm ơn bạn",
        "cảm ơn nhiều",
        "thank",
        "thanks",
        "thank you",
    ]

    # Check if query contains any simple terms
    for simple_query in simple_queries:
        if simple_query in query:
            return ""

    # Check query length - very short queries don't need sources
    if len(query.split()) <= 3:
        return ""

    # Deduplicate sources
    unique_sources = []
    unique_urls = set()

    for source in results["sources"]:
        # Create unique key for each source
        source_key = (
            source.get("document_title", ""),
            source.get("decision_number", ""),
        )

        # Only add if not already added and has a document_title
        if source_key not in unique_urls and source_key[0]:
            unique_urls.add(source_key)
            unique_sources.append(source)

    # If no valid sources, return empty string
    if not unique_sources:
        return ""

    # Format sources
    sources_text = "\nNguồn tham khảo:\n"

    for idx, source in enumerate(unique_sources, 1):
        sources_text += f"\n{idx}. "

        if source["document_title"]:
            sources_text += f"{source['document_title']}"

        if source["decision_number"]:
            sources_text += f" - {source['decision_number']}"

        if source["date"]:
            sources_text += f" ({source['date']})"

        if source["agency"]:
            sources_text += f"\n   Cơ quan ban hành: {source['agency']}"

        if source["source"]:
            sources_text += f"\n   Nguồn: {source['source']}"

    return sources_text
