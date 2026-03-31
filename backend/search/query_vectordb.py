import csv
import logging
import os
import pickle
import time
from typing import Any, Dict, List, Optional

import google.generativeai as genai
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

# Import prediction function from predict.py
try:
    from backend.search.predict import prediction as predict_legal_query
except ImportError:
    # Trong Docker container, không có thư mục 'backend' ở root
    from search.predict import prediction as predict_legal_query

# Import SQLAlchemy để truy vấn lịch sử trò chuyện
try:
    from sqlalchemy.orm import Session

    try:
        from backend.core.database import get_db
        from backend.core.models import QueryHistory
    except ImportError:
        # Trong Docker container, không có thư mục 'backend' ở root
        from core.database import get_db
        from core.models import QueryHistory

    DB_AVAILABLE = True
except ImportError:
    DB_AVAILABLE = False
    logger = logging.getLogger(__name__)
    logger.warning(
        "Database modules not available, conversation history will not be used"
    )

# Cấu hình logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

# Kiểm tra xem có thể sử dụng mô hình phân loại không
CLASSIFIER_AVAILABLE = True
try:
    # Kiểm tra xem có file mô hình phân loại không
    backend_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    classifier_path = os.path.join(
        backend_dir, "classification_models", "question_classifier.pkl"
    )
    vectorizer_path = os.path.join(
        backend_dir, "classification_models", "tfidf_vectorizer.pkl"
    )

    if os.path.exists(classifier_path) and os.path.exists(vectorizer_path):
        logger.info("Classification model files found")
    else:
        logger.warning("Classification model files not found")
        CLASSIFIER_AVAILABLE = False
except Exception as e:
    logger.warning(f"Error checking classification model: {e}")
    CLASSIFIER_AVAILABLE = False


try:
    from backend.core.config import (
        COLLECTION_NAME,
        DB_PATH,
        EMBEDDING_MODEL,
        GEMINI_API_KEYS,
        GEMINI_MAX_TOKENS,
        GEMINI_MODEL,
        GEMINI_TEMPERATURE,
        HYBRID_ALPHA,
        MARKDOWN_DIR,
        QUERY_EXPANSION,
        TOP_K,
        USE_FAISS,
        USE_HYBRID_SEARCH,
        VECTOR_SIZE,
    )
except ImportError:
    # Trong Docker container, không có thư mục 'backend' ở root
    from core.config import (
        COLLECTION_NAME,
        DB_PATH,
        EMBEDDING_MODEL,
        GEMINI_API_KEYS,
        GEMINI_MAX_TOKENS,
        GEMINI_MODEL,
        GEMINI_TEMPERATURE,
        HYBRID_ALPHA,
        MARKDOWN_DIR,
        QUERY_EXPANSION,
        TOP_K,
        USE_FAISS,
        USE_HYBRID_SEARCH,
        VECTOR_SIZE,
    )

# Define cache directory
CACHE_DIR = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "cache"
)
# Create cache directory if it doesn't exist
os.makedirs(CACHE_DIR, exist_ok=True)
# Log cache directory for debugging
logger.info(f"Using cache directory: {CACHE_DIR}")

# Đường dẫn đến file metadata.csv
METADATA_CSV_PATH = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    "markdown_data",
    "metadata.csv",
)

# Biến lưu trữ metadata
_metadata_cache = None


def load_document_metadata():
    """Đọc file metadata.csv và trả về dictionary ánh xạ document_id với thông tin metadata."""
    global _metadata_cache

    # Nếu đã có cache, trả về luôn
    if _metadata_cache is not None:
        return _metadata_cache

    metadata = {}
    try:
        with open(METADATA_CSV_PATH, "r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                doc_id = row.get("DocumentID", "")
                if doc_id:
                    metadata[doc_id] = {
                        "source": row.get("Source", ""),
                        "content": row.get("Content", ""),
                        "number": row.get("Number", ""),
                    }

        logger.info(
            f"Loaded metadata for {len(metadata)} documents from {METADATA_CSV_PATH}"
        )
        _metadata_cache = metadata
        return metadata
    except Exception as e:
        logger.error(f"Error loading metadata from {METADATA_CSV_PATH}: {e}")
        return {}


from dotenv import load_dotenv

# Kiểm tra xem sentence-transformers có thể sử dụng được không
try:
    import torch
    from sentence_transformers import SentenceTransformer

    # Đường dẫn đến mô hình
    model_path = EMBEDDING_MODEL
    logger.info(f"Using embedding model path: {model_path}")

    # Kiểm tra xem có file safetensors không
    safetensors_path = os.path.join(model_path, "model.safetensors")
    pytorch_path = os.path.join(model_path, "pytorch_model.bin")

    # Nếu có file safetensors nhưng không có file pytorch_model.bin, chuyển đổi
    if os.path.exists(safetensors_path) and not os.path.exists(pytorch_path):
        logger.info(f"Converting safetensors to PyTorch format for model: {model_path}")
        try:
            # Tạo thư mục 0_ViT nếu chưa có
            vit_dir = os.path.join(model_path, "0_ViT")
            os.makedirs(vit_dir, exist_ok=True)

            # Chuyển đổi safetensors sang pytorch_model.bin
            from safetensors import safe_open
            from safetensors.torch import save_file

            with safe_open(safetensors_path, framework="pt", device="cpu") as f:
                tensors = {key: f.get_tensor(key) for key in f.keys()}

            torch.save(tensors, pytorch_path)
            logger.info(f"Successfully converted safetensors to PyTorch format")
        except Exception as e:
            logger.warning(f"Error converting safetensors to PyTorch format: {e}")

    # Tạo mô hình SentenceTransformer
    model = SentenceTransformer(model_path)

    # Thử encode một câu để kiểm tra
    test_embedding = model.encode("Test sentence")
    SENTENCE_TRANSFORMERS_AVAILABLE = True
    logger.info(f"Successfully loaded embedding model: {model_path}")
except Exception as e:
    logger.warning(f"Error loading sentence-transformers model: {e}")
    logger.warning("Using random embeddings as fallback")
    SENTENCE_TRANSFORMERS_AVAILABLE = False

load_dotenv()


def get_embedding(text: str) -> List[float]:
    """Get embedding for text using sentence-transformers or random fallback."""
    if SENTENCE_TRANSFORMERS_AVAILABLE:
        return model.encode(text).tolist()
    else:
        # Tạo embedding ngẫu nhiên với kích thước đúng
        np.random.seed(
            hash(text) % 2**32
        )  # Đảm bảo tính nhất quán cho cùng một văn bản
        return np.random.rand(VECTOR_SIZE).tolist()


# Import hybrid search functionality
try:
    try:
        from backend.search.hybrid_search import (
            format_hybrid_results,
            hybrid_search,
            load_cached_documents,
            load_cached_embeddings,
            load_cached_metadata,
            load_faiss_index,
            load_or_create_bm25_index,
            read_markdown_files,
            search_faiss_index,
        )
    except ImportError:
        # Trong Docker container, không có thư mục 'backend' ở root
        from search.hybrid_search import (
            format_hybrid_results,
            hybrid_search,
            load_cached_documents,
            load_cached_embeddings,
            load_cached_metadata,
            load_faiss_index,
            load_or_create_bm25_index,
            read_markdown_files,
            search_faiss_index,
        )

    HYBRID_SEARCH_AVAILABLE = True
    logger.info("Hybrid search functionality is available")

    # Initialize BM25 index and other resources
    _bm25_model = None
    _corpus = None
    _doc_ids = None
    _documents = None
    _document_embeddings = None
    _document_metadata = None

    # Cache for hybrid search results
    _hybrid_search_cache = {}
    _last_query = None
    _last_hybrid_results = None
    _last_classification_result = None  # Lưu kết quả phân loại gần đây nhất

    def get_bm25_resources():
        """Get or initialize BM25 resources."""
        global _bm25_model, _corpus, _doc_ids, _documents
        if _bm25_model is None:
            # Try to load from cache first
            _documents = load_cached_documents()
            # Đọc các file markdown từ thư mục
            markdown_files = read_markdown_files(MARKDOWN_DIR)
            _bm25_model, _corpus, _doc_ids = load_or_create_bm25_index(markdown_files)
        return _bm25_model, _corpus, _doc_ids, _documents

    def get_document_embeddings():
        """Get cached document embeddings if available."""
        global _document_embeddings
        if _document_embeddings is None:
            # Try to load from cache
            _document_embeddings = load_cached_embeddings()
        return _document_embeddings

    def get_document_metadata():
        """Get cached document metadata if available."""
        global _document_metadata
        if _document_metadata is None:
            # Try to load from cache
            _document_metadata = load_cached_metadata()
        return _document_metadata

except ImportError as e:
    logger.warning(f"Hybrid search not available: {e}")
    HYBRID_SEARCH_AVAILABLE = False


def query_documents(
    query: str,
    collection_name: str = COLLECTION_NAME,
    top_k: int = TOP_K,
    db_path: str = DB_PATH,
) -> List[Dict]:
    """Query documents using hybrid search when available."""
    global _last_query, _last_hybrid_results

    try:
        # Đo thời gian thực thi
        start_time = time.time()

        # Use hybrid search if available and enabled
        if USE_HYBRID_SEARCH and HYBRID_SEARCH_AVAILABLE:
            logger.info("Using hybrid search")

            # Kiểm tra cache trước
            if _last_query == query and _last_hybrid_results is not None:
                logger.info("Using cached hybrid search results")

                # Lấy kết quả từ cache
                hybrid_results = _last_hybrid_results["hybrid_results"]
                documents = _last_hybrid_results["documents"]

                # Load cached metadata (không được cache trong _last_hybrid_results)
                doc_metadata = load_cached_metadata()
                if not doc_metadata:
                    logger.error("No cached metadata found")
                    return []

                # Format results với top_k mới
                if (
                    top_k != 5
                ):  # Nếu top_k khác với giá trị mặc định trong is_legal_related_query
                    hybrid_results = hybrid_results[:top_k]

                # Format results
                formatted_results = format_hybrid_results(
                    hybrid_results, documents, doc_metadata
                )

                if formatted_results:
                    return formatted_results

            # Nếu không có cache hoặc cache không hợp lệ, thực hiện tìm kiếm mới
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

            # Tạo embedding cho query và chuẩn bị model cho query expansion
            query_embedding = get_embedding(query)

            # Chuẩn bị model cho query expansion nếu cần
            model_for_expansion = None
            if QUERY_EXPANSION and SENTENCE_TRANSFORMERS_AVAILABLE:
                model_for_expansion = model  # Sử dụng model đã được khởi tạo ở đầu file

            # Check if FAISS index is available
            faiss_index = None
            faiss_doc_ids = None
            if USE_FAISS:
                try:
                    faiss_index, faiss_doc_ids = load_faiss_index()
                    if faiss_index is not None:
                        logger.info(
                            f"FAISS index loaded with {faiss_index.ntotal} vectors"
                        )
                    else:
                        logger.warning(
                            "FAISS index not found, falling back to cosine similarity"
                        )
                except Exception as e:
                    logger.warning(f"Error loading FAISS index: {e}")

            # Perform hybrid search
            hybrid_results = hybrid_search(
                query=query,
                document_embeddings=document_embeddings,
                bm25_model=bm25_model,
                doc_ids=doc_ids,
                model=model_for_expansion,  # Truyền model nếu cần query expansion
                corpus=corpus,
                alpha=HYBRID_ALPHA,
                k=top_k,
                query_expansion=QUERY_EXPANSION,
                query_embedding=query_embedding,
                use_faiss=USE_FAISS,
            )

            # Lưu kết quả vào cache để tái sử dụng
            _last_query = query
            _last_hybrid_results = {
                "hybrid_results": hybrid_results,
                "documents": documents,
                "bm25_model": bm25_model,
                "corpus": corpus,
                "doc_ids": doc_ids,
                "document_embeddings": document_embeddings,
                "query_embedding": query_embedding,
                "model_for_expansion": model_for_expansion,
            }

            # Format results
            formatted_results = format_hybrid_results(
                hybrid_results, documents, doc_metadata
            )

            if formatted_results:
                return formatted_results

        # Fall back to standard vector search
        logger.info("Using standard vector search")

        # Load cached resources for vector search
        document_embeddings = load_cached_embeddings()
        if not document_embeddings:
            logger.warning(
                "No cached embeddings found for vector search, falling back to Gemini API"
            )
            # Continue with Gemini API fallback instead of returning empty results

        documents = load_cached_documents()
        doc_metadata = load_cached_metadata()

        # Perform vector search if we have embeddings
        results = {}
        if document_embeddings:
            query_vector = get_embedding(query)

            for doc_id, doc_emb in document_embeddings.items():
                sim = cosine_similarity([query_vector], [doc_emb])[0][0]
                results[doc_id] = sim

        # Sort results by similarity score
        sorted_results = sorted(results.items(), key=lambda x: x[1], reverse=True)[
            :top_k
        ]

        # Format results
        formatted_results = []
        for doc_id, score in sorted_results:
            if doc_id in documents and doc_id in doc_metadata:
                formatted_results.append(
                    {
                        "id": doc_id,
                        "score": float(score),
                        "content": documents[doc_id],
                        "metadata": doc_metadata[doc_id],
                    }
                )

        # Ghi log thời gian thực thi
        execution_time = time.time() - start_time
        logger.info(f"Query documents execution time: {execution_time:.2f} seconds")

        return formatted_results

    except Exception as e:
        logger.error(f"Error in query_documents: {e}")
        return []


def format_results(results: List) -> List[Dict[str, Any]]:
    """Format the results from Qdrant query."""
    formatted_results = []

    for result in results:
        formatted_doc = {
            "content": result.payload.get("content", ""),
            "metadata": {
                "document_title": result.payload.get("documenttitle", ""),
                "context": result.payload.get("context", ""),
                "agency": result.payload.get("agency", ""),
                "decision_number": result.payload.get("decisionnumber", ""),
                "date": result.payload.get("date", ""),
                "source": result.payload.get("source", ""),
            },
            "score": result.score,
        }
        formatted_results.append(formatted_doc)

    return formatted_results


def generate_answer_with_gemini(
    query: str,
    context: str,
    api_key: str = None,
    model_name: Optional[str] = None,
    temperature: float = GEMINI_TEMPERATURE,
    max_tokens: int = GEMINI_MAX_TOKENS,
    is_legal_query: bool = True,  # Thêm tham số để xác định loại câu hỏi
    conversation_history: List[Dict] = None,  # Thêm lịch sử trò chuyện
    doc_ids: List[str] = None,  # Thêm danh sách document_ids
) -> Dict[str, Any]:
    """Generate answer using Gemini API based on retrieved context."""
    # Tránh "đóng băng" tên model ở default argument tại import-time.
    # Nếu không truyền vào, dùng giá trị cấu hình hiện tại.
    if not model_name:
        model_name = GEMINI_MODEL

    # Sử dụng danh sách API key
    api_keys_to_try = []

    # Nếu có API key được truyền vào, đặt nó là key đầu tiên để thử
    if api_key:
        api_keys_to_try.append(api_key)

    # Thêm các API key từ cấu hình
    api_keys_to_try.extend(
        [key for key in GEMINI_API_KEYS if key not in api_keys_to_try]
    )

    # Nếu không có API key nào, báo lỗi
    if not api_keys_to_try:
        raise ValueError("No Gemini API keys available")

    # Biến lưu lỗi cuối cùng
    last_error = None

    # Thử lần lượt từng API key
    for i, current_api_key in enumerate(api_keys_to_try):
        try:
            # Cấu hình Gemini với API key hiện tại
            genai.configure(api_key=current_api_key)

            # Ghi log API key đang sử dụng (chỉ hiển thị 5 ký tự đầu và 5 ký tự cuối để bảo mật)
            key_preview = (
                f"{current_api_key[:5]}...{current_api_key[-5:]}"
                if len(current_api_key) > 10
                else "[masked]"
            )
            logger.info(
                f"Using Gemini API key {i+1}/{len(api_keys_to_try)}: {key_preview}"
            )

            # Chuẩn bị phần lịch sử trò chuyện nếu có
            conversation_context = ""
            if conversation_history and len(conversation_history) > 0:
                conversation_context = "\n\n**Lịch sử trò chuyện gần đây:**\n"
                for _, item in enumerate(conversation_history):  # Use _ to ignore idx
                    conversation_context += (
                        f"\nNgười dùng: {item['query']}\nTrợ lý: {item['answer']}\n"
                    )
                logger.info(
                    f"Added {len(conversation_history)} conversation history items to prompt"
                )

            # Chọn prompt phù hợp dựa trên loại câu hỏi
            if is_legal_query:
                # Chuẩn bị danh sách document_ids nếu có
                if doc_ids and len(doc_ids) > 0:
                    logger.info(f"Added {len(doc_ids)} document IDs to prompt")

                # Prompt cho câu hỏi liên quan đến luật/chính sách
                prompt = f"""
**Vai trò:** AI chuyên gia về chính sách người có công với cách mạng Việt Nam.

**Nhiệm vụ:** Trả lời `CÂU HỎI` dựa **duy nhất** và **chính xác** vào `THÔNG TIN` được cung cấp.

**Yêu cầu cốt lõi:**
1.  **Nguồn:** Tuyệt đối chỉ sử dụng `THÔNG TIN`. Không dùng kiến thức ngoài, không suy diễn.
2.  **Trích dẫn:** Trích dẫn nguyên văn ("...") từ `THÔNG TIN` khi cần thiết để làm rõ và đảm bảo tính chính xác.
3.  **Thiếu thông tin:** Nếu `THÔNG TIN` không đủ, chỉ trả lời: "Dựa trên thông tin được cung cấp, tôi không tìm thấy nội dung để trả lời cho câu hỏi này." và dòng đầu tiên phải là "Document IDs: None"
4.  **Phong cách:** Chuyên nghiệp, lịch sự, khách quan, tôn trọng, rõ ràng.
5.  **Định dạng đầu ra:** **CỰC KỲ QUAN TRỌNG:**
   - Dòng đầu tiên của câu trả lời PHẢI là danh sách các Document ID mà bạn đã sử dụng để trả lời câu hỏi, định dạng: "Document IDs: id1, id2, id3"
   - Nếu không tìm thấy thông tin để trả lời, dòng đầu tiên PHẢI là "Document IDs: None"
   - Từ dòng thứ hai trở đi mới là nội dung câu trả lời thuần túy
   - **KHÔNG** lặp lại `THÔNG TIN`, `CÂU HỎI`, không thêm tiêu đề, lời dẫn hay bất kỳ nội dung nào khác.
6.  **Sử dụng thông tin từ lịch sử trò chuyện:** Tham khảo thông tin từ `LỊCH SỬ TRÒ CHUYỆN` nếu có liên quan để đảm bảo tính nhất quán trong cuộc trò chuyện.

**Lưu ý:** `THÔNG TIN`, `CÂU HỎI`, và `LỊCH SỬ TRÒ CHUYỆN` dưới đây chỉ để bạn tham khảo, **KHÔNG** được xuất hiện trong đầu ra cuối cùng.

--- THÔNG TIN THAM KHẢO ---
THÔNG TIN:
{context}

CÂU HỎI:
{query}

LỊCH SỬ TRÒ CHUYỆN:
{conversation_context}
--- KẾT THÚC THÔNG TIN THAM KHẢO ---

**Quy trình suy nghĩ (Chain of Thought):**
1.  Tiếp nhận và phân tích CÂU HỎI.
2.  Đọc và xử lý THÔNG TIN được cung cấp.
3.  Kiểm tra LỊCH SỬ TRÒ CHUYỆN để đảm bảo tính nhất quán (nếu có).
4.  **Quan trọng:** Duy nhất tìm kiếm câu trả lời cho CÂU HỎI CHỈ trong nội dung THÔNG TIN và LỊCH SỬ TRÒ CHUYỆN.
5.  **Quyết định:** Nếu tìm thấy nội dung liên quan trực tiếp và đủ để trả lời CÂU HỎI từ THÔNG TIN/LỊCH SỬ TRÒ CHUYỆN, tiến hành soạn câu trả lời, trích dẫn nguyên văn nếu cần. Đảm bảo câu trả lời CHỈ chứa nội dung thuần túy.
6.  **Quyết định:** Nếu THÔNG TIN/LỊCH SỬ TRÒ CHUYỆN không chứa nội dung đủ để trả lời CÂU HỎI, chuẩn bị câu trả lời mặc định: "Document IDs: None\nDựa trên thông tin được cung cấp, tôi không tìm thấy nội dung để trả lời cho câu hỏi này."
7.  **Quyết định:** Nếu nội dung trong THÔNG TIN không liên quan đến văn bản pháp luật hoặc chính sách người có công với cách mạng, trả lời: "Document IDs: None\nNội dung không liên quan đến văn bản pháp luật."
8.  **Quan trọng:** Xác định các Document ID mà bạn đã sử dụng để trả lời câu hỏi. Mỗi đoạn thông tin trong THÔNG TIN đều có định dạng [Document ID: xxx].
9.  Xuất bản kết quả theo **Định dạng đầu ra CỰC KỲ QUAN TRỌNG** (dòng đầu tiên là Document IDs, từ dòng thứ hai là nội dung trả lời).

**Bắt đầu soạn câu trả lời:**
"""
            else:
                # Prompt cho câu hỏi thông thường (không liên quan đến luật/chính sách)
                prompt = f"""
**Vai trò:** Bạn là trợ lý AI hỗ trợ trả lời các câu hỏi về chính sách người có công với cách mạng Việt Nam.

**Nhiệm vụ:** Trả lời câu hỏi của người dùng một cách thân thiện và hữu ích.

**Hướng dẫn:**
1. Nếu người dùng chào hỏi, hãy chào lại một cách lịch sự và thân thiện.
2. Nếu người dùng hỏi về khả năng của bạn, hãy giới thiệu bạn là trợ lý AI chuyên về chính sách người có công với cách mạng Việt Nam.
3. Nếu người dùng yêu cầu gợi ý câu hỏi, hãy đưa ra một số ví dụ về câu hỏi liên quan đến chính sách người có công với cách mạng Việt Nam.
4. Nếu thông tin không liên quan gì đến chính sách người có công với cách mạng Việt Nam thì bảo họ chỉ hỏi về chính sách người có công với cách mạng Việt Nam.
5. Phong cách trả lời: Thân thiện, ngắn gọn, rõ ràng và hữu ích.
6. Sử dụng thông tin từ lịch sử trò chuyện để đảm bảo tính nhất quán trong cuộc trò chuyện.

CÂU HỎI:
{query}

LỊCH SỬ TRÒ CHUYỆN:
{conversation_context}

Trả lời:
"""

            # Ghi log loại câu hỏi
            logger.info(
                f"Query type: {'Legal-related' if is_legal_query else 'General'}"
            )

            # Đo thời gian thực thi
            start_time = time.time()

            # Tạo mô hình và gọi API với timeout
            model = genai.GenerativeModel(model_name=model_name)

            # Gọi API với timeout ngắn hơn
            response = model.generate_content(
                prompt,
                generation_config=genai.types.GenerationConfig(
                    temperature=temperature,
                    max_output_tokens=max_tokens,
                ),
            )

            # Ghi log thời gian thực thi
            execution_time = time.time() - start_time
            logger.info(f"Gemini API execution time: {execution_time:.2f} seconds")

            # Xử lý câu trả lời để trích xuất document_ids nếu có
            answer_text = response.text
            referenced_doc_ids = []

            if is_legal_query and answer_text:
                # Tách câu trả lời thành các dòng
                lines = answer_text.strip().split("\n")

                # Kiểm tra xem dòng đầu tiên có chứa Document IDs không
                if lines and "Document IDs:" in lines[0]:
                    # Trích xuất document_ids từ dòng đầu tiên
                    doc_ids_line = lines[0].replace("Document IDs:", "").strip()
                    referenced_doc_ids = [
                        doc_id.strip()
                        for doc_id in doc_ids_line.split(",")
                        if doc_id.strip()
                    ]

                    # Loại bỏ dòng đầu tiên khỏi câu trả lời
                    answer_text = "\n".join(lines[1:]).strip()

                    logger.info(
                        f"Extracted referenced document IDs: {referenced_doc_ids}"
                    )

            # Nếu thành công, trả về kết quả
            return {
                "answer": answer_text,
                "model": model_name,
                "query": query,
                "api_key_index": i,  # Lưu chỉ số của API key đã sử dụng
                "execution_time": execution_time,  # Thêm thời gian thực thi
                "referenced_doc_ids": referenced_doc_ids,  # Thêm document_ids được tham chiếu
            }

        except Exception as e:
            # Ghi log lỗi
            last_error = e
            logger.error(f"Error with API key {i+1}/{len(api_keys_to_try)}: {str(e)}")

            # Nếu đây là API key cuối cùng, báo lỗi
            if i == len(api_keys_to_try) - 1:
                logger.critical(
                    f"All Gemini API keys failed. Last error: {str(last_error)}"
                )
                raise ValueError(
                    f"All Gemini API keys failed. Last error: {str(last_error)}"
                )

            # Nếu không, tiếp tục thử API key tiếp theo
            logger.warning(f"Trying next API key...")
            continue


def rag_answer_gemini(
    query: str,
    collection_name: str = COLLECTION_NAME,
    top_k: int = TOP_K,
    db_path: str = DB_PATH,
    api_key: str = None,
    conversation_id: int = None,  # Thêm tham số conversation_id
) -> Dict:
    """Implement RAG system using document search and Gemini."""
    # Lấy lịch sử trò chuyện nếu có conversation_id
    conversation_history = []
    if conversation_id is not None:
        conversation_history = get_conversation_history(conversation_id)
        if conversation_history:
            logger.info(
                f"Using {len(conversation_history)} conversation history items for query"
            )

    # Kiểm tra xem câu hỏi có liên quan đến luật/chính sách hay không
    is_legal_query = is_legal_related_query(query)

    # Nếu câu hỏi không liên quan đến luật/chính sách, sử dụng prompt thông thường
    if not is_legal_query:
        logger.info(f"Non-legal query detected: {query}")
        # Sử dụng context rỗng cho các câu hỏi không liên quan đến luật/chính sách
        answer_result = generate_answer_with_gemini(
            query,
            "",
            api_key,
            is_legal_query=False,
            conversation_history=conversation_history,
        )
        return {
            "query": query,
            "answer": answer_result["answer"],
            "model": answer_result["model"],
            "context": "",
            "sources": [],
            "execution_time": answer_result.get("execution_time", 0),
        }

    # Nếu câu hỏi liên quan đến luật/chính sách, sử dụng RAG
    logger.info(f"Legal query detected: {query}")

    # Tìm kiếm các tài liệu liên quan
    relevant_docs = query_documents(query, collection_name, top_k, db_path)

    context = ""
    sources = []
    doc_ids = []  # Danh sách các document_id

    for doc in relevant_docs:
        # Lấy document_id từ metadata
        doc_id = doc["metadata"].get("source", "").replace("Document ID: ", "")
        logging.info(f"Document ID: {doc_id}")

        # Lấy ID ngắn gọn từ doc_id (chỉ lấy tên file không có phần mở rộng)
        short_id = doc_id

        # Nếu doc_id là URL hoặc đường dẫn dài, trích xuất phần cuối cùng
        if doc_id and ("/" in doc_id or "\\" in doc_id):
            # Lấy phần cuối cùng của đường dẫn (tên file)
            short_id = doc_id.split("/")[-1].split("\\")[-1]

        # Loại bỏ phần mở rộng file nếu có
        if "." in short_id:
            short_id = short_id.split(".")[0]

        # Thêm short_id vào danh sách nếu có
        if short_id:
            doc_ids.append(short_id)

        # Thêm nội dung vào context với short_id
        context += f"\n\n---\n[Document ID: {short_id}]\n{doc['content']}"

        source_info = {
            "id": doc_id,  # Thêm document_id gốc vào source_info
            "short_id": short_id,  # Thêm short_id vào source_info
            "document_title": doc["metadata"].get("document_title", ""),
            "context": doc["metadata"].get("context", ""),
            "agency": doc["metadata"].get("agency", ""),
            "decision_number": doc["metadata"].get("decision_number", ""),
            "date": doc["metadata"].get("date", ""),
            "source": doc["metadata"].get("source", ""),
            "relevance_score": doc["score"],
        }
        sources.append(source_info)

    # Tạo danh sách document_ids để hiển thị
    doc_ids_str = ", ".join(doc_ids)
    logger.info(f"Document IDs for query: {doc_ids_str}")

    answer_result = generate_answer_with_gemini(
        query,
        context,
        api_key,
        is_legal_query=True,
        conversation_history=conversation_history,
        doc_ids=doc_ids,  # Truyền danh sách document_ids
    )

    # Trích xuất document_ids từ câu trả lời nếu có
    answer_text = answer_result["answer"]
    referenced_doc_ids = answer_result.get("referenced_doc_ids", [])
    return {
        "query": query,
        "answer": answer_text,
        "model": answer_result["model"],
        "context": context,
        "sources": sources,
        "doc_ids": doc_ids,  # Thêm tất cả document_ids
        "referenced_doc_ids": referenced_doc_ids,  # Thêm document_ids được tham chiếu
        "execution_time": answer_result.get("execution_time", 0),
    }


def format_answer(results: Dict) -> str:
    """Format the answer for display."""
    answer = results["answer"]

    # Split the answer into lines
    lines = answer.split("\n")
    formatted_lines = []

    for line in lines:
        # Check if line starts with asterisk
        if line.strip().startswith("*"):
            # Replace asterisk with bullet point
            line = line.replace("*", "•", 1)
        formatted_lines.append(line)

    # Join lines back together
    return "\n".join(formatted_lines)


def is_legal_related_query(query: str) -> bool:
    """Kiểm tra xem câu hỏi có liên quan đến luật/chính sách hay không dựa trên mô hình phân loại."""
    # Đo thời gian thực thi
    start_time = time.time()

    try:
        # Kiểm tra độ dài câu hỏi - câu hỏi quá ngắn cũng không liên quan đến luật/chính sách
        if len(query.lower().split()) <= 2:
            logger.info(f"Query too short: {query}")
            execution_time = time.time() - start_time
            logger.info(
                f"is_legal_related_query execution time: {execution_time:.2f} seconds"
            )
            return False

        # Kiểm tra các câu hỏi chào hỏi đơn giản
        simple_greetings = [
            "xin chào",
            "chào",
            "hi",
            "hello",
            "hey",
            "cảm ơn",
            "tạm biệt",
            "bye",
        ]
        logger.info(
            f"Checking if '{query.lower()}' is in simple greetings list: {query.lower() in simple_greetings}"
        )
        if query.lower() in simple_greetings:
            logger.info(f"Simple greeting detected: {query}")
            execution_time = time.time() - start_time
            logger.info(
                f"is_legal_related_query execution time: {execution_time:.2f} seconds"
            )
            return False

        # Sử dụng mô hình phân loại nếu có sẵn
        if CLASSIFIER_AVAILABLE:
            try:
                # Sử dụng hàm prediction từ predict.py
                logger.info(
                    f"Using prediction function from predict.py for query: {query}"
                )
                prediction_result = predict_legal_query(query)

                if prediction_result is None:
                    logger.warning(
                        "Prediction function returned None, falling back to hybrid search"
                    )
                    # Nếu hàm prediction trả về None, sử dụng phương pháp dự phòng
                    raise Exception("Prediction function returned None")

                # Lấy kết quả từ hàm prediction
                is_legal = prediction_result.get("is_legal_question", False)
                prediction = prediction_result.get("prediction", 0)
                confidence = prediction_result.get("confidence", None)
                model_type = prediction_result.get("model_type", "Unknown")

                logger.info(f"Model prediction: {prediction}")
                if confidence is not None:
                    logger.info(f"Confidence: {confidence:.4f}")

                logger.info(
                    f"Final conclusion: This query is{' ' if is_legal else ' NOT '}a legal question"
                )

                # Lưu thông tin chi tiết về kết quả phân loại
                global _last_classification_result
                _last_classification_result = {
                    "query": query,
                    "is_legal_question": is_legal,
                    "prediction": int(prediction),
                    "confidence": float(confidence) if confidence is not None else None,
                    "model_type": model_type,
                }

                # Ghi log thời gian thực thi
                execution_time = time.time() - start_time
                logger.info(
                    f"is_legal_related_query execution time: {execution_time:.2f} seconds"
                )

                # Trả về kết quả phân loại
                return is_legal

            except Exception as e:
                logger.error(f"Error using classification model: {e}")
                # Nếu có lỗi khi sử dụng mô hình, sử dụng phương pháp dự phòng
                logger.warning(
                    "Falling back to hybrid search method for classification"
                )
        else:
            logger.warning(
                "Classification model not available, using hybrid search method"
            )

        # Phương pháp dự phòng: Sử dụng hybrid search để tìm kiếm các tài liệu liên quan
        if USE_HYBRID_SEARCH and HYBRID_SEARCH_AVAILABLE:
            # Load cached documents
            documents = load_cached_documents()
            if not documents:
                logger.error("No cached documents found")
                return False

            # Load BM25 resources
            bm25_model, corpus, doc_ids = load_or_create_bm25_index(documents)

            # Load cached embeddings
            document_embeddings = load_cached_embeddings()
            if not document_embeddings:
                logger.error("No cached embeddings found")
                return False

            # Tạo embedding cho query
            query_embedding = get_embedding(query)

            # Chuẩn bị model cho query expansion nếu cần
            model_for_expansion = None
            if QUERY_EXPANSION and SENTENCE_TRANSFORMERS_AVAILABLE:
                model_for_expansion = model  # Sử dụng model đã được khởi tạo ở đầu file

            # Check if FAISS index is available
            if USE_FAISS:
                try:
                    faiss_index, _ = load_faiss_index()  # Use _ to ignore faiss_doc_ids
                    if faiss_index is not None:
                        logger.info(
                            f"FAISS index loaded with {faiss_index.ntotal} vectors for classification"
                        )
                    else:
                        logger.warning(
                            "FAISS index not found for classification, falling back to cosine similarity"
                        )
                except Exception as e:
                    logger.warning(f"Error loading FAISS index for classification: {e}")

            # Perform hybrid search
            hybrid_results = hybrid_search(
                query=query,
                document_embeddings=document_embeddings,
                bm25_model=bm25_model,
                doc_ids=doc_ids,
                model=model_for_expansion,
                corpus=corpus,
                alpha=HYBRID_ALPHA,
                k=5,  # Chỉ cần 5 kết quả đầu tiên để kiểm tra
                query_expansion=QUERY_EXPANSION,
                query_embedding=query_embedding,
                use_faiss=USE_FAISS,
            )

            # Lưu kết quả vào cache để tái sử dụng
            global _last_query, _last_hybrid_results
            _last_query = query
            _last_hybrid_results = {
                "hybrid_results": hybrid_results,
                "documents": documents,
                "bm25_model": bm25_model,
                "corpus": corpus,
                "doc_ids": doc_ids,
                "document_embeddings": document_embeddings,
                "query_embedding": query_embedding,
                "model_for_expansion": model_for_expansion,
            }

            # Nếu không có kết quả nào, có thể không liên quan đến luật/chính sách
            if not hybrid_results:
                logger.info(f"No hybrid search results for query: {query}")
                execution_time = time.time() - start_time
                logger.info(
                    f"is_legal_related_query execution time: {execution_time:.2f} seconds"
                )
                return False

            # Lấy điểm cao nhất
            top_score = hybrid_results[0][1]
            logger.info(f"Top relevance score for query '{query}': {top_score}")

            # Hiển thị thông tin về tài liệu liên quan nhất
            top_doc_id = hybrid_results[0][0]
            logger.info(f"Top document ID: {top_doc_id}")
            if top_doc_id in documents:
                doc_content = (
                    documents[top_doc_id][:100] + "..."
                    if len(documents[top_doc_id]) > 100
                    else documents[top_doc_id]
                )
                logger.info(f"Top document content preview: {doc_content}")

            # Nếu điểm cao nhất vượt quá ngưỡng, coi là câu hỏi liên quan đến luật/chính sách
            # Sử dụng ngưỡng cao hơn (0.6) cho phương pháp dự phòng để giảm số lượng câu hỏi không liên quan đến luật được phân loại nhầm
            logger.info(
                f"Checking if top score {top_score} > threshold 0.6: {top_score > 0.6}"
            )
            if top_score > 0.6:  # Ngưỡng cao hơn cho phương pháp dự phòng
                execution_time = time.time() - start_time
                logger.info(
                    f"is_legal_related_query execution time: {execution_time:.2f} seconds"
                )
                return True

            # Nếu điểm thấp hơn ngưỡng, có thể không liên quan đến luật/chính sách
            execution_time = time.time() - start_time
            logger.info(
                f"is_legal_related_query execution time: {execution_time:.2f} seconds"
            )
            return False
        else:
            # Nếu không có hybrid search, sử dụng vector search
            document_embeddings = load_cached_embeddings()
            if not document_embeddings:
                logger.warning("No cached embeddings found for vector search")
                execution_time = time.time() - start_time
                logger.info(
                    f"is_legal_related_query execution time: {execution_time:.2f} seconds"
                )
                return False

            # Tạo embedding cho query
            query_vector = get_embedding(query)

            # Tính độ tương đồng với từng tài liệu
            similarities = []
            for _, doc_emb in document_embeddings.items():  # Use _ to ignore doc_id
                sim = cosine_similarity([query_vector], [doc_emb])[0][0]
                similarities.append(sim)

            # Lấy độ tương đồng cao nhất
            if similarities:
                max_similarity = max(similarities)
                logger.info(
                    f"Max vector similarity for query '{query}': {max_similarity}"
                )

                # Nếu độ tương đồng cao hơn ngưỡng, coi là câu hỏi liên quan đến luật/chính sách
                # Sử dụng ngưỡng cao hơn (0.6) cho phương pháp dự phòng
                if max_similarity > 0.6:  # Ngưỡng cao hơn cho phương pháp dự phòng
                    execution_time = time.time() - start_time
                    logger.info(
                        f"is_legal_related_query execution time: {execution_time:.2f} seconds"
                    )
                    return True

            # Nếu không có độ tương đồng cao, có thể không liên quan đến luật/chính sách
            execution_time = time.time() - start_time
            logger.info(
                f"is_legal_related_query execution time: {execution_time:.2f} seconds"
            )
            return False
    except Exception as e:
        logger.error(f"Error in is_legal_related_query: {e}")
        # Nếu có lỗi, mặc định coi là câu hỏi liên quan đến luật/chính sách để an toàn
        execution_time = time.time() - start_time
        logger.info(
            f"is_legal_related_query execution time (error): {execution_time:.2f} seconds"
        )
        return True


def get_conversation_history(conversation_id: int, limit: int = 5) -> List[Dict]:
    """Lấy lịch sử trò chuyện từ cơ sở dữ liệu.

    Args:
        conversation_id: ID của cuộc trò chuyện
        limit: Số lượng cặp câu hỏi-câu trả lời gần nhất cần lấy

    Returns:
        Danh sách các cặp câu hỏi-câu trả lời gần nhất
    """
    if not DB_AVAILABLE or conversation_id is None:
        return []

    try:
        # Tạo session mới
        db = next(get_db())

        # Truy vấn lịch sử trò chuyện
        history_items = (
            db.query(QueryHistory)
            .filter(QueryHistory.conversation_id == conversation_id)
            .order_by(QueryHistory.created_at.desc())
            .limit(limit)
            .all()
        )

        # Đảo ngược danh sách để có thứ tự từ cũ đến mới
        history_items.reverse()

        # Chuyển đổi thành danh sách các cặp câu hỏi-câu trả lời
        history = [
            {"query": item.query_text, "answer": item.answer_text}
            for item in history_items
        ]

        logger.info(
            f"Retrieved {len(history)} conversation history items for conversation {conversation_id}"
        )
        return history
    except Exception as e:
        logger.error(f"Error retrieving conversation history: {e}")
        return []


def format_sources(results: Dict) -> str:
    """Format the sources for display."""
    # Kiểm tra xem có cần hiển thị nguồn tham khảo không
    # Nếu không có sources hoặc sources rỗng, không hiển thị gì
    if not results.get("sources"):
        logger.info("No sources in results, not showing sources")
        return ""

    # Kiểm tra xem câu trả lời có phải là thông báo không tìm thấy nội dung không
    default_answer = "Dựa trên thông tin được cung cấp, tôi không tìm thấy nội dung để trả lời cho câu hỏi này."
    if results.get("answer", "").strip() == default_answer:
        logger.info("Answer is default 'no content found' message, not showing sources")
        return ""

    # Kiểm tra xem câu trả lời có bắt đầu bằng "Document IDs: None" không
    if results.get("answer", "").strip().startswith("Document IDs: None"):
        logger.info("Answer starts with 'Document IDs: None', not showing sources")
        return ""

    # Kiểm tra xem câu hỏi có liên quan đến luật/chính sách hay không
    query = results.get("query", "")

    # Kiểm tra xem câu trả lời có được tạo ra từ RAG hay không
    # Nếu context rỗng, có nghĩa là không phải câu hỏi liên quan đến văn bản pháp luật
    if not results.get("context"):
        logger.info(f"Not showing sources for non-legal query: {query[:50]}...")
        return ""

    # Kiểm tra độ dài câu hỏi - câu hỏi quá ngắn cũng không cần nguồn
    if len(query.split()) <= 2:
        logger.info(f"Not showing sources for short query: {query}")
        return ""

    # Lấy danh sách document_ids được tham chiếu
    referenced_doc_ids = results.get("referenced_doc_ids", [])

    # Tải metadata từ file CSV
    metadata_dict = load_document_metadata()

    # Nếu có referenced_doc_ids, chỉ hiển thị các nguồn tương ứng
    if referenced_doc_ids:
        logger.info(f"Using referenced document IDs: {referenced_doc_ids}")
        filtered_sources = []

        for source in results["sources"]:
            # Kiểm tra cả id và short_id
            if (
                source.get("short_id") and source.get("short_id") in referenced_doc_ids
            ) or (source.get("id") in referenced_doc_ids):
                filtered_sources.append(source)

        # Nếu không có nguồn tham khảo hợp lệ, không hiển thị gì
        if not filtered_sources:
            logger.info(f"No valid sources found for referenced document IDs")
            return ""

        # Format nguồn tham khảo
        sources_text = "Nguồn tham khảo:\n"

        # Tạo set để theo dõi các nguồn đã hiển thị
        shown_sources = set()
        source_count = 1

        for source in filtered_sources:
            # Lấy short_id để tìm trong metadata
            doc_id = source.get("short_id") or ""

            # Tìm thông tin từ metadata.csv
            if doc_id and doc_id in metadata_dict:
                metadata = metadata_dict[doc_id]
                # Tạo key duy nhất cho nguồn dựa trên nội dung và số hiệu
                source_key = (metadata["content"], metadata["number"])

                # Nếu nguồn này chưa được hiển thị
                if source_key not in shown_sources:
                    shown_sources.add(source_key)
                    sources_text += f"\n{source_count}. "
                    sources_text += f"{metadata['content']}"
                    sources_text += f"\nSố hiệu văn bản: {metadata['number']}"
                    sources_text += f"\nNguồn: {metadata['source']}\n"
                    source_count += 1
            else:
                # Fallback nếu không tìm thấy trong metadata
                # Tạo key duy nhất cho nguồn dựa trên document_title và decision_number
                source_key = (
                    source.get("document_title", ""),
                    source.get("decision_number", ""),
                )

                # Nếu nguồn này chưa được hiển thị
                if source_key not in shown_sources:
                    shown_sources.add(source_key)
                    sources_text += f"\n{source_count}. "
                    if source.get("document_title"):
                        sources_text += f"{source['document_title']}"
                    if source.get("decision_number"):
                        sources_text += f" - {source['decision_number']}"
                    if source.get("date"):
                        sources_text += f" ({source['date']})"
                    if source.get("source"):
                        sources_text += f"\n   Nguồn: {source['source']}"
                    source_count += 1

        logger.info(
            f"Showing {len(shown_sources)} unique sources for legal query based on referenced document IDs"
        )
        return sources_text.rstrip()

    # Nếu không có referenced_doc_ids, sử dụng cách cũ
    # Tạo set để theo dõi các nguồn đã hiển thị
    shown_sources = set()
    source_count = 1
    sources_text = "\nNguồn tham khảo:\n"

    for source in results["sources"]:
        # Lấy short_id để tìm trong metadata
        doc_id = source.get("short_id") or ""

        # Tìm thông tin từ metadata.csv
        if doc_id and doc_id in metadata_dict:
            metadata = metadata_dict[doc_id]
            # Tạo key duy nhất cho nguồn dựa trên nội dung và số hiệu
            source_key = (metadata["content"], metadata["number"])

            # Nếu nguồn này chưa được hiển thị
            if source_key not in shown_sources:
                shown_sources.add(source_key)
                sources_text += f"\n{source_count}. "
                sources_text += f"{metadata['content']}"
                sources_text += f"\nSố hiệu văn bản: {metadata['number']}"
                sources_text += f"\nNguồn: {metadata['source']}"
                source_count += 1
        else:
            # Fallback nếu không tìm thấy trong metadata
            # Tạo key duy nhất cho nguồn dựa trên document_title và decision_number
            source_key = (
                source.get("document_title", ""),
                source.get("decision_number", ""),
            )

            # Nếu nguồn này chưa được hiển thị
            if source_key not in shown_sources:
                shown_sources.add(source_key)
                sources_text += f"\n{source_count}. "
                if source.get("document_title"):
                    sources_text += f"{source['document_title']}"
                if source.get("decision_number"):
                    sources_text += f" - {source['decision_number']}"
                if source.get("date"):
                    sources_text += f" ({source['date']})"
                if source.get("source"):
                    sources_text += f"\n   Nguồn: {source['source']}"
                source_count += 1

    logger.info(
        f"Showing {len(shown_sources)} unique sources for legal query: {query[:50]}..."
    )
    return sources_text.rstrip()


if __name__ == "__main__":
    # Câu hỏi cho trước
    query = "Xin chào"

    # Chạy pipeline
    results = rag_answer_gemini(query)

    # In kết quả
    print("Câu hỏi:", query)
    print(format_answer(results))
    print(format_sources(results))
