import csv
import logging
import os
import pickle
import re
import time
from typing import Any, Dict, List, Optional, TYPE_CHECKING

if TYPE_CHECKING:
    from sqlalchemy.orm import Session

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


_LEGAL_INTERPRETIVE_HINT = re.compile(
    r"t[oó]m\s*t[ắa]t|summar|l[àa]m\s*r[õo]|gi[ảa]i\s*th[íi]ch|ph[âa]n\s*t[íi]ch|"
    r"kh[áa]i\s*ni[ệe]m|thu[ậa]t\s*ng[ữu]|ngh[ĩi]a\s*(l[àa]|g[ìi])|"
    r"v[ăa]n\s*b[ảa]n\s*ph[áa]p\s*lu[ậa]t|ph[áa]p\s*lu[ậa]t\s*vi[ệe]t|"
    r"quy\s*[đd][ịi]nh\s*ph[áa]p\s*lu[ậa]t|[đd]i[ềe]u\s*kho[ảa]n|"
    r"so\s*s[áa]nh|v[íi]\s*d[ụu]|v[íi]\s*sa|l[àa]m\s*r[õo]\s*gi[úu]p|"
    r"hi[ểe]u\s*r[õo]|[đd][ọo]c\s*hi[ểe]u",
    re.IGNORECASE,
)


def is_legal_interpretation_or_assist_query(query: str) -> bool:
    """Tóm tắt / làm rõ / giải thích pháp luật: cho phép bổ sung kiến thức như trợ lý Gemini."""
    q = (query or "").strip()
    if len(q) < 10:
        return False
    return bool(_LEGAL_INTERPRETIVE_HINT.search(q))




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
    user_chunk_refs: Optional[List[str]] = None,  # ví dụ U12-0 (tài liệu người dùng)
    user_documents_only: bool = False,  # chỉ đoạn trích file người dùng, không kho VBQPPL
    legal_interpretive_mode: bool = False,  # tóm tắt/làm rõ/giải thích — bổ sung kiến thức như Gemini khi cần
) -> Dict[str, Any]:
    """Generate answer using Gemini API based on retrieved context."""
    # Tránh "đóng băng" tên model ở default argument tại import-time.
    # Nếu không truyền vào, dùng giá trị cấu hình hiện tại.
    if not model_name:
        model_name = GEMINI_MODEL

    if user_chunk_refs is None:
        user_chunk_refs = []

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

                user_only_scope = ""
                if user_documents_only:
                    user_only_scope = """
**Phạm vi bắt buộc (chế độ chỉ tài liệu tải lên):** `THÔNG TIN` chỉ gồm các khối [Tài liệu người dùng ...] từ file người dùng đã gửi. Không có văn bản từ kho pháp luật quốc gia trong ngữ cảnh này. Bạn chỉ được phân tích, tóm tắt và trích dẫn nội dung trong các khối đó. Tuyệt đối không đưa kiến thức pháp luật khác hoặc văn bản từ kho dữ liệu mà bạn không thấy trong `THÔNG TIN`. Dòng 1 của định dạng đầu ra luôn là: Document IDs: None

"""

                flex_hint = ""
                if legal_interpretive_mode and not user_documents_only:
                    flex_hint = """
**Chế độ trợ lý linh hoạt (gần Gemini):** Ưu tiên căn cứ vào `THÔNG TIN` để trích dẫn. Bạn được phép bổ sung kiến thức pháp luật Việt Nam phổ biến để tóm tắt, làm rõ thuật ngữ, giải thích khái niệm hoặc đưa ví dụ khi đoạn trích không đủ hoặc câu hỏi mang tính hướng dẫn hiểu. Phải ghi rõ phần nào từ tài liệu trích, phần nào là kiến thức bổ sung. Tránh trả lời chỉ bằng một câu từ chối nếu vẫn có thể hỗ trợ hữu ích một phần.

"""
                if legal_interpretive_mode and not user_documents_only:
                    rule1 = "1.  **Nguồn:** Ưu tiên `THÔNG TIN`; khi cần tóm tắt, làm rõ hoặc giải thích mà đoạn trích chưa đủ, được bổ sung kiến thức pháp luật Việt Nam phổ biến và ghi rõ phần nào từ trích dẫn, phần nào là bổ sung."
                    rule3 = '3.  **Thiếu thông tin:** Chỉ dùng câu "Dựa trên thông tin được cung cấp, tôi không tìm thấy nội dung để trả lời cho câu hỏi này." khi thực sự không có nội dung liên quan trong `THÔNG TIN` và không thể hỗ trợ hợp lý bằng kiến thức chung; nếu vẫn giúp được một phần (tóm tắt khung, giải thích thuật ngữ), hãy trả lời phần đó và nêu hạn chế.'
                    cot4 = "4.  **Quan trọng:** Ưu tiên dùng `THÔNG TIN` và lịch sử; phần bổ sung phải phù hợp câu hỏi và ghi rõ là kiến thức chung."
                    cot5 = "5.  **Quyết định:** Soạn câu trả lời: ưu tiên nội dung trích từ THÔNG TIN; phần kiến thức bổ sung phải tách biệt rõ ràng, không bịa số điều cụ thể."
                    cot6 = "6.  **Quyết định:** Nếu thiếu đoạn trích nhưng có thể làm rõ bằng kiến thức pháp luật phổ biến, hãy trả lời có cấu trúc; chỉ từ chối khi không thể hỗ trợ gì an toàn."
                else:
                    rule1 = "1.  **Nguồn:** Chỉ sử dụng `THÔNG TIN` (có thể gồm văn bản pháp luật và đoạn trích từ tài liệu cá nhân người dùng đã tải lên). Không dùng kiến thức ngoài, không suy diễn."
                    rule3 = '3.  **Thiếu thông tin:** Nếu `THÔNG TIN` không đủ, chỉ trả lời: "Dựa trên thông tin được cung cấp, tôi không tìm thấy nội dung để trả lời cho câu hỏi này." — khi đó dòng 1 là "Document IDs: None" và dòng 2 là "User document refs: None".'
                    cot4 = "4.  **Quan trọng:** Duy nhất tìm kiếm câu trả lời cho CÂU HỎI CHỈ trong nội dung THÔNG TIN và LỊCH SỬ TRÒ CHUYỆN."
                    cot5 = "5.  **Quyết định:** Nếu tìm thấy nội dung liên quan trực tiếp và đủ để trả lời CÂU HỎI từ THÔNG TIN/LỊCH SỬ TRÒ CHUYỆN, tiến hành soạn câu trả lời, trích dẫn nguyên văn nếu cần. Đảm bảo câu trả lời CHỈ chứa nội dung thuần túy."
                    cot6 = "6.  **Quyết định:** Nếu THÔNG TIN/LỊCH SỬ TRÒ CHUYỆN không chứa nội dung đủ để trả lời CÂU HỎI, dùng hai dòng đầu \"Document IDs: None\" và \"User document refs: None\", sau đó là thông báo thiếu thông tin."

                # Prompt cho câu hỏi liên quan đến luật/chính sách
                prompt = f"""
**Vai trò:** AI chuyên gia về pháp luật Việt Nam.

**Nhiệm vụ:** Trả lời `CÂU HỎI` dựa **duy nhất** và **chính xác** vào `THÔNG TIN` được cung cấp.
{user_only_scope}{flex_hint}
**Yêu cầu cốt lõi:**
{rule1}
2.  **Trích dẫn:** Khi dùng văn bản pháp luật hoặc trích từ tài liệu cá nhân, ghi rõ nguồn: với luật — theo [Document ID: ...]; với file người dùng — ghi tên file và trang (có trong khối [Tài liệu người dùng ...]).
{rule3}
4.  **Phong cách:** Chuyên nghiệp, lịch sự, khách quan, tôn trọng, rõ ràng.
5.  **Định dạng đầu ra:** **CỰC KỲ QUAN TRỌNG:**
   - Dòng 1: "Document IDs: id1, id2" (các ID trong [Document ID: xxx]) hoặc "Document IDs: None" nếu không dùng văn bản pháp luật.
   - Dòng 2: "User document refs: U5-0, U5-1" (mã Ref trong các khối [Tài liệu người dùng ... | Ref: ...]) hoặc "User document refs: None" nếu không dùng tài liệu cá nhân.
   - Từ dòng 3 trở đi: nội dung trả lời thuần túy (không lặp lại THÔNG TIN/CÂU HỎI).
6.  **Sử dụng thông tin từ lịch sử trò chuyện:** Tham khảo `LỊCH SỬ TRÒ CHUYỆN` nếu có liên quan.

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
{cot4}
{cot5}
{cot6}
7.  **Quyết định:** Nếu THÔNG TIN không liên quan câu hỏi, vẫn tuân thủ định dạng hai dòng đầu rồi giải thích ngắn.
8.  **Quan trọng:** Ghi đúng các Document ID ([Document ID: xxx]) và mã User document refs (Ref: Ux-y) mà bạn đã dùng.
9.  Xuất bản theo **Định dạng đầu ra** (dòng 1 Document IDs, dòng 2 User document refs, từ dòng 3 là nội dung).

**Bắt đầu soạn câu trả lời:**
"""
            else:
                # Prompt chung — có thể kèm tài liệu người dùng đã tải (context) để tóm tắt / làm rõ
                user_doc_section = (context or "").strip()
                if user_doc_section:
                    doc_hint = """
**Tài liệu người dùng:** Phía dưới có thể có đoạn trích từ file PDF/Word đã tải lên hệ thống (kèm tên file và trang).
7. Nếu CÂU HỎI yêu cầu tóm tắt, giải thích, làm rõ, tìm hiểu thêm về **nội dung file đã gửi**, bạn **ưu tiên** trả lời dựa trên các đoạn trích đó; ghi rõ đang nói về file nào (và trang nếu có).
8. Nếu đoạn trích không đủ, nói rõ phần nào chưa có trong trích đoạn và gợi ý câu hỏi cụ thể hơn.
"""
                else:
                    doc_hint = """
7. Hiện không có đoạn trích tài liệu đính kèm trong phiên; nếu người dùng muốn tóm tắt/giải thích file, nhắc họ đảm bảo đã tải file lên và đặt câu hỏi liên quan nội dung.
"""
                prompt = f"""
**Vai trò:** Bạn là trợ lý AI hỗ trợ trả lời các câu hỏi về pháp luật Việt Nam.

**Nhiệm vụ:** Trả lời câu hỏi của người dùng một cách thân thiện và hữu ích.

**Hướng dẫn:**
1. Nếu người dùng chào hỏi, hãy chào lại một cách lịch sự và thân thiện.
2. Nếu người dùng hỏi về khả năng của bạn, hãy giới thiệu bạn là trợ lý AI chuyên về pháp luật Việt Nam.
3. Nếu người dùng yêu cầu gợi ý câu hỏi, hãy đưa ra một số ví dụ về câu hỏi liên quan đến pháp luật Việt Nam.
4. Nếu thông tin không liên quan gì đến pháp luật Việt Nam thì bảo họ chỉ hỏi về pháp luật Việt Nam (trừ khi đang xử lý tóm tắt/giải thích theo tài liệu họ đã tải — mục 7–8).
5. Phong cách trả lời: Thân thiện, ngắn gọn, rõ ràng và hữu ích.
6. Sử dụng thông tin từ lịch sử trò chuyện để đảm bảo tính nhất quán trong cuộc trò chuyện.
{doc_hint}

--- ĐOẠN TRÍCH TÀI LIỆU NGƯỜI DÙNG (có thể rỗng) ---
{user_doc_section if user_doc_section else "(Không có đoạn trích trong phiên này.)"}
--- HẾT ĐOẠN TRÍCH ---

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

            referenced_user_refs: List[str] = []
            if is_legal_query and answer_text:
                lines = answer_text.strip().split("\n")
                line_idx = 0
                if lines and "Document IDs:" in lines[0]:
                    doc_ids_line = lines[0].replace("Document IDs:", "").strip()
                    referenced_doc_ids = [
                        doc_id.strip()
                        for doc_id in doc_ids_line.split(",")
                        if doc_id.strip() and doc_id.strip().lower() != "none"
                    ]
                    line_idx = 1
                    logger.info(
                        f"Extracted referenced document IDs: {referenced_doc_ids}"
                    )
                if len(lines) > line_idx and "User document refs:" in lines[line_idx]:
                    uline = lines[line_idx].replace("User document refs:", "").strip()
                    if uline.lower() != "none":
                        referenced_user_refs = [
                            x.strip()
                            for x in uline.split(",")
                            if x.strip() and x.strip().lower() != "none"
                        ]
                    line_idx += 1
                    logger.info(
                        f"Extracted referenced user document refs: {referenced_user_refs}"
                    )
                answer_text = "\n".join(lines[line_idx:]).strip()

            # Nếu thành công, trả về kết quả
            return {
                "answer": answer_text,
                "model": model_name,
                "query": query,
                "api_key_index": i,  # Lưu chỉ số của API key đã sử dụng
                "execution_time": execution_time,  # Thêm thời gian thực thi
                "referenced_doc_ids": referenced_doc_ids,  # Thêm document_ids được tham chiếu
                "referenced_user_refs": referenced_user_refs,
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
    db: Any = None,
    user_id: Optional[int] = None,
    include_user_documents: bool = True,
    user_documents_only: bool = False,
    user_document_id: Optional[int] = None,
) -> Dict:
    """Implement RAG system using document search and Gemini."""
    # -------- Smart routing for user-uploaded docs (NotebookLM-like) --------
    # Heuristic intents where semantic retrieval over chunks is often weak:
    # - "tóm tắt tài liệu", "phân tích file", "giải thích nội dung file", ...
    # In these cases we prefer to load MANY chunks from the selected/latest file.
    _USER_DOC_SUMMARY_INTENT = re.compile(
        r"(t[oó]m\s*t[ắa]t|summar|t[óo]m\s*l[ạa]i|t[óo]m\s*t[ắa]t\s*(t[àa]i\s*li[ệe]u|v[ăa]n\s*b[ảa]n|file)|"
        r"ph[âa]n\s*t[íi]ch\s*(t[àa]i\s*li[ệe]u|v[ăa]n\s*b[ảa]n|file)|"
        r"gi[ảa]i\s*th[íi]ch\s*(n[ộo]i\s*dung|t[àa]i\s*li[ệe]u|file)|"
        r"l[àa]m\s*r[õo]\s*(n[ộo]i\s*dung|t[àa]i\s*li[ệe]u|file)|"
        r"file\s*v[ừa]\s*upload|file\s*m[ới]|t[àa]i\s*li[ệe]u\s*v[ừa]\s*t[ảa]i)",
        re.IGNORECASE,
    )

    def _is_user_doc_summary_intent(q: str) -> bool:
        qq = (q or "").strip()
        if not qq:
            return False
        if len(qq) > 240:
            return False
        return bool(_USER_DOC_SUMMARY_INTENT.search(qq))

    def _is_user_doc_query(q: str) -> bool:
        ql = (q or "").strip().lower()
        if not ql:
            return False
        keys = [
            "tài liệu",
            "tai lieu",
            "file",
            "văn bản này",
            "van ban nay",
            "tài liệu này",
            "tai lieu nay",
            "trong tài liệu",
            "trong tai lieu",
            "trong file",
            "theo tài liệu",
            "theo tai lieu",
            "nội dung file",
            "noi dung file",
            "nội dung tài liệu",
            "noi dung tai lieu",
            "ở trang",
            "o trang",
        ]
        return any(k in ql for k in keys)
    # Retrieval quality threshold for user-chunk semantic matches.
    # If below this and we don't have a doc-summary intent, we may fall back to LLM.
    USER_CHUNK_MIN_SCORE = 0.14
    # Lấy lịch sử trò chuyện nếu có conversation_id
    conversation_history = []
    if conversation_id is not None:
        conversation_history = get_conversation_history(conversation_id)
        if conversation_history:
            logger.info(
                f"Using {len(conversation_history)} conversation history items for query"
            )

    
    # Auto document QA routing (Way 1):
    # If the user has uploaded documents and the query refers to the document/file,
    # force document QA mode (skip national legal corpus).
    if (
        (not user_documents_only)
        and include_user_documents
        and user_id is not None
        and db is not None
        and (_is_user_doc_query(query) or _is_user_doc_summary_intent(query) or user_document_id is not None)
    ):
        user_documents_only = True

    # Chỉ trả lời trong phạm vi file người dùng — không gọi kho VBQPPL (tránh lẫn nội dung)
    if (
        user_documents_only
        and include_user_documents
        and user_id is not None
        and db is not None
    ):
        logger.info(
            "User-documents-only mode: skipping national legal corpus retrieval"
        )
        try:
            try:
                from core.config import USER_RAG_TOP_K
                from search.user_document_rag import (
                    get_latest_ready_document_id,
                    merge_semantic_and_recent_user_chunks,
                )
            except ImportError:
                from backend.core.config import USER_RAG_TOP_K
                from backend.search.user_document_rag import (
                    get_latest_ready_document_id,
                    merge_semantic_and_recent_user_chunks,
                )

            doc_filter: Optional[List[int]] = None
            if user_document_id is not None:
                doc_filter = [user_document_id]
            else:
                latest = get_latest_ready_document_id(db, user_id)
                if latest is not None:
                    doc_filter = [latest]

            if not doc_filter:
                return {
                    "query": query,
                    "answer": (
                        "Chưa có tài liệu nào sẵn sàng. "
                        "Vui lòng tải file lên và đợi xử lý xong, rồi thử lại."
                    ),
                    "model": GEMINI_MODEL,
                    "context": "",
                    "sources": [],
                    "user_sources": [],
                    "doc_ids": [],
                    "referenced_doc_ids": [],
                    "referenced_user_refs": [],
                    "execution_time": 0.0,
                }

            # For summary-like intents, prefer many chunks from the chosen/latest file
            # (NotebookLM-like) instead of relying on semantic top-k.
            if _is_user_doc_summary_intent(query):
                try:
                    from search.user_document_rag import get_recent_user_chunks
                except ImportError:
                    from backend.search.user_document_rag import get_recent_user_chunks
                hits = get_recent_user_chunks(
                    db, user_id, limit=80, document_ids=doc_filter
                )
            else:
                hits = merge_semantic_and_recent_user_chunks(
                    db,
                    user_id,
                    query,
                    top_k=USER_RAG_TOP_K,
                    document_ids=doc_filter,
                )
            if not hits:
                return {
                    "query": query,
                    "answer": (
                        "Không đọc được đoạn trích từ tài liệu đã chọn. "
                        "Thử tải lại file hoặc chọn file khác."
                    ),
                    "model": GEMINI_MODEL,
                    "context": "",
                    "sources": [],
                    "user_sources": [],
                    "answer_mode": "fallback_ai",
                    "doc_ids": [],
                    "referenced_doc_ids": [],
                    "referenced_user_refs": [],
                    "execution_time": 0.0,
                }

            user_context_str = ""
            user_sources_uo: List[Dict[str, Any]] = []
            for h in hits:
                user_context_str += (
                    f"\n\n---\n[Tài liệu người dùng — {h['filename']} | "
                    f"trang {h['page_start']}-{h['page_end']} | Ref: {h['ref']}]\n"
                    f"{h['text']}"
                )
                user_sources_uo.append(
                    {
                        "source_type": "user_upload",
                        "ref": h["ref"],
                        "filename": h["filename"],
                        "page_start": h["page_start"],
                        "page_end": h["page_end"],
                        "relevance_score": h["score"],
                        "excerpt": (
                            (h["text"][:400] + "…")
                            if len(h["text"]) > 400
                            else h["text"]
                        ),
                    }
                )
            refs = [h["ref"] for h in hits]
            # DOCUMENT QA: use general assistant prompt on uploaded document content.
            answer_result = generate_answer_with_gemini(
                query,
                user_context_str.strip(),
                api_key,
                is_legal_query=False,
                conversation_history=conversation_history,
            )
            answer_text = answer_result["answer"]
            referenced_doc_ids = answer_result.get("referenced_doc_ids", [])
            referenced_user_refs = answer_result.get("referenced_user_refs", [])
            return {
                "query": query,
                "answer": answer_text,
                "model": answer_result["model"],
                "context": user_context_str.strip(),
                "sources": [],
                "user_sources": [],
                "answer_mode": "doc_qa",
                "doc_ids": [],
                "referenced_doc_ids": referenced_doc_ids,
                "referenced_user_refs": referenced_user_refs,
                "execution_time": answer_result.get("execution_time", 0),
            }
        except Exception as exc:
            logger.warning("User-documents-only RAG failed: %s", exc)
            return {
                "query": query,
                "answer": (
                    f"Không xử lý được tài liệu chỉ định: {exc!s}. "
                    "Vui lòng thử lại hoặc tải lại file."
                ),
                "model": GEMINI_MODEL,
                "context": "",
                "sources": [],
                "user_sources": [],
                "answer_mode": "fallback_ai",
                "doc_ids": [],
                "referenced_doc_ids": [],
                "referenced_user_refs": [],
                "execution_time": 0.0,
            }

    # Kiểm tra xem câu hỏi có liên quan đến luật/chính sách hay không
    is_legal_query = is_legal_related_query(query)
    interpretive_assist = is_legal_interpretation_or_assist_query(query)
    if interpretive_assist:
        is_legal_query = True
        logger.info("Interpretive legal assist: legal RAG + flexible prompt")

    # Nếu câu hỏi không liên quan đến luật/chính sách, sử dụng prompt thông thường
    if not is_legal_query:
        logger.info(f"Non-legal query detected: {query}")
        user_context_str = ""
        user_sources_nl: List[Dict[str, Any]] = []
        if include_user_documents and user_id is not None and db is not None:
            try:
                try:
                    from core.config import USER_RAG_TOP_K
                    from search.user_document_rag import (
                        merge_semantic_and_recent_user_chunks,
                    )
                except ImportError:
                    from backend.core.config import USER_RAG_TOP_K
                    from backend.search.user_document_rag import (
                        merge_semantic_and_recent_user_chunks,
                    )
                hits = merge_semantic_and_recent_user_chunks(
                    db, user_id, query, top_k=USER_RAG_TOP_K
                )
                top_score = hits[0]["score"] if hits else 0.0
                # If we have user docs but retrieval is weak and the query is doc-summary-like,
                # load more recent chunks (NotebookLM-like).
                if _is_user_doc_summary_intent(query):
                    try:
                        from search.user_document_rag import (
                            get_latest_ready_document_id,
                            get_recent_user_chunks,
                        )
                    except ImportError:
                        from backend.search.user_document_rag import (
                            get_latest_ready_document_id,
                            get_recent_user_chunks,
                        )
                    latest = get_latest_ready_document_id(db, user_id)
                    if latest is not None:
                        hits = get_recent_user_chunks(
                            db, user_id, limit=80, document_ids=[latest]
                        )
                for h in hits:
                    user_context_str += (
                        f"\n\n---\n[Tài liệu người dùng — {h['filename']} | "
                        f"trang {h['page_start']}-{h['page_end']} | Ref: {h['ref']}]\n"
                        f"{h['text']}"
                    )
                    user_sources_nl.append(
                        {
                            "source_type": "user_upload",
                            "ref": h["ref"],
                            "filename": h["filename"],
                            "page_start": h["page_start"],
                            "page_end": h["page_end"],
                            "relevance_score": h["score"],
                            "excerpt": (
                                (h["text"][:400] + "…")
                                if len(h["text"]) > 400
                                else h["text"]
                            ),
                        }
                    )
                # If retrieval produced nothing useful, fall back to general LLM chat.
                # (Do NOT show "căn cứ pháp lý" when there is no document evidence.)
                if not user_context_str.strip() and top_score < USER_CHUNK_MIN_SCORE:
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
                        "user_sources": [],
                        "answer_mode": "fallback_ai",
                        "doc_ids": [],
                        "referenced_doc_ids": [],
                        "referenced_user_refs": [],
                        "execution_time": answer_result.get("execution_time", 0),
                    }
            except Exception as exc:
                logger.warning("User document retrieval (non-legal branch): %s", exc)

        answer_result = generate_answer_with_gemini(
            query,
            user_context_str.strip(),
            api_key,
            is_legal_query=False,
            conversation_history=conversation_history,
        )
        return {
            "query": query,
            "answer": answer_result["answer"],
            "model": answer_result["model"],
            "context": user_context_str.strip(),
            "sources": [],
            "user_sources": user_sources_nl,
            "answer_mode": "doc_summary" if _is_user_doc_summary_intent(query) and user_context_str.strip() else ("rag_user_docs" if user_context_str.strip() else "fallback_ai"),
            "doc_ids": [],
            "referenced_doc_ids": [],
            "referenced_user_refs": [],
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


    user_sources: List[Dict[str, Any]] = []
    user_chunk_refs: List[str] = []
    if include_user_documents and user_id is not None and db is not None:
        try:
            try:
                from core.config import USER_RAG_TOP_K
                from search.user_document_rag import merge_semantic_and_recent_user_chunks
            except ImportError:
                from backend.core.config import USER_RAG_TOP_K
                from backend.search.user_document_rag import (
                    merge_semantic_and_recent_user_chunks,
                )
            hits = merge_semantic_and_recent_user_chunks(
                db, user_id, query, top_k=USER_RAG_TOP_K
            )
            for h in hits:
                user_chunk_refs.append(h["ref"])
                context += (
                    f"\n\n---\n[Tài liệu người dùng — {h['filename']} | "
                    f"trang {h['page_start']}-{h['page_end']} | Ref: {h['ref']}]\n"
                    f"{h['text']}"
                )
                user_sources.append(
                    {
                        "source_type": "user_upload",
                        "ref": h["ref"],
                        "filename": h["filename"],
                        "page_start": h["page_start"],
                        "page_end": h["page_end"],
                        "relevance_score": h["score"],
                        "excerpt": (
                            (h["text"][:400] + "…")
                            if len(h["text"]) > 400
                            else h["text"]
                        ),
                    }
                )
        except Exception as _user_rag_err:
            logger.warning("User document RAG skipped: %s", _user_rag_err)


    # Fallback: if the classifier thought it's a legal query but we retrieved no
    # relevant context, we should answer as a normal chat assistant.
    # Otherwise the legal RAG prompt will force the default message:
    # "Dựa trên thông tin được cung cấp, tôi không tìm thấy nội dung..."
    if not context.strip():
        logger.info(
            "Legal query detected but context is empty; falling back to general Gemini chat."
        )
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
            "user_sources": [],
            "execution_time": answer_result.get("execution_time", 0),
        }

    answer_result = generate_answer_with_gemini(
        query,
        context,
        api_key,
        is_legal_query=True,
        conversation_history=conversation_history,
        doc_ids=doc_ids,
        user_chunk_refs=user_chunk_refs,
        legal_interpretive_mode=interpretive_assist,
    )

    # Robust fallback:
    # The legal RAG prompt forces a default message if it judges that THÔNG TIN
    # is insufficient. If that happens, rerun in "general chat" mode so the
    # user gets a normal Gemini answer instead of the default legal message.
    default_answer = (
        "Dựa trên thông tin được cung cấp, tôi không tìm thấy nội dung để trả lời cho câu hỏi này."
    )
    if (
        answer_result.get("answer", "").strip() == default_answer
        and context.strip()
    ):
        logger.info(
            "Legal RAG returned the default 'no content' answer; "
            "falling back to general Gemini chat."
        )
        answer_result = generate_answer_with_gemini(
            query,
            context if interpretive_assist else "",
            api_key,
            is_legal_query=False,
            conversation_history=conversation_history,
        )

    # Trích xuất document_ids từ câu trả lời nếu có
    answer_text = answer_result["answer"]
    referenced_doc_ids = answer_result.get("referenced_doc_ids", [])
    referenced_user_refs = answer_result.get("referenced_user_refs", [])
    return {
        "query": query,
        "answer": answer_text,
        "model": answer_result["model"],
        "context": context,
        "sources": sources,
        "user_sources": user_sources,
        "doc_ids": doc_ids,
        "referenced_doc_ids": referenced_doc_ids,
        "referenced_user_refs": referenced_user_refs,
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




def _append_user_document_sources(results: Dict, base: str) -> str:
    """Append citations for user-uploaded chunks after law sources."""
    us = results.get("user_sources") or []
    refs = results.get("referenced_user_refs") or []
    ref_set = set(refs) if refs else None
    if not us:
        return base.rstrip()
    lines: List[str] = []
    if base.strip():
        lines.append(base.rstrip())
    lines.append("")
    lines.append("Tài liệu đã tải lên:")
    n = 1
    for u in us:
        if ref_set is not None and u.get("ref") not in ref_set:
            continue
        lines.append(
            f"{n}. {u.get('filename', '')} — trang {u.get('page_start', '')}-{u.get('page_end', '')} "
            f"(Ref: {u.get('ref', '')})"
        )
        n += 1
    if n == 1:
        return base.rstrip()
    return "\n".join(lines).rstrip()


def format_sources(results: Dict) -> str:
    """Format the sources for display."""
    # Kiểm tra xem có cần hiển thị nguồn tham khảo không
    # Nếu không có sources hoặc sources rỗng, không hiển thị gì
    # Special UX: fallback to general AI (no document evidence)
    if results.get("answer_mode") == "fallback_ai":
        return "AI tổng hợp (không có trong tài liệu)"

    # DOCUMENT QA mode: answer is generated from the uploaded document content,
    # but UX should NOT show "căn cứ pháp lý"/chunk listings.
    if results.get("answer_mode") in ("doc_qa", "doc_summary"):
        return ""

    if not (results.get("sources") or results.get("user_sources")):
        logger.info("No sources in results, not showing sources")
        return ""

    # Kiểm tra xem câu trả lời có phải là thông báo không tìm thấy nội dung không
    default_answer = "Dựa trên thông tin được cung cấp, tôi không tìm thấy nội dung để trả lời cho câu hỏi này."
    if results.get("answer", "").strip() == default_answer:
        logger.info("Answer is default 'no content found' message, not showing sources")
        return ""

    # DOCUMENT QA / doc-summary UX: do not show citations card.
    if results.get("answer_mode") in ("doc_qa", "doc_summary"):
        return ""

    # Nếu câu trả lời theo định dạng user-doc-only thì law doc ids sẽ None,
    # nhưng vẫn có thể có citations từ file đã tải lên.
    if results.get("answer", "").strip().startswith("Document IDs: None"):
        logger.info(
            "Answer starts with 'Document IDs: None' (user-doc scope); showing user sources if any"
        )
        return _append_user_document_sources(results, "")

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

    def _render_source_line(source: Dict[str, Any], metadata_dict: Dict[str, Dict[str, str]]) -> Optional[str]:
        """Render one unique law source block.

        Prefer the metadata extracted during retrieval (document_title/decision_number/date/source)
        because it is guaranteed to correspond to the retrieved chunk. Fall back to metadata.csv
        only when the retrieval metadata is missing/empty.
        """
        title = (source.get("document_title") or "").strip()
        decision = (source.get("decision_number") or "").strip()
        dt = (source.get("date") or "").strip()
        src = (source.get("source") or "").strip()

        if title or decision or dt or src:
            parts: List[str] = []
            head = title or "Văn bản pháp luật"
            if decision:
                head += f" - {decision}"
            if dt:
                head += f" ({dt})"
            parts.append(head)
            if src:
                parts.append(f"Nguồn: {src}")
            return "\n".join(parts)

        # Fallback: lookup by short_id in metadata.csv
        doc_id = (source.get("short_id") or "").strip()
        md = metadata_dict.get(doc_id) if doc_id else None
        if not md:
            return None
        content = (md.get("content") or "").strip()
        number = (md.get("number") or "").strip()
        md_source = (md.get("source") or "").strip()
        if not (content or number or md_source):
            return None
        parts2: List[str] = []
        parts2.append(content or "Văn bản pháp luật")
        if number:
            parts2.append(f"Số hiệu văn bản: {number}")
        if md_source:
            parts2.append(f"Nguồn: {md_source}")
        return "\n".join(parts2)

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
            logger.info("No valid law sources for referenced document IDs; may still have user docs")
            return _append_user_document_sources(results, "")

        # Format nguồn tham khảo
        sources_text = "Nguồn tham khảo:\n"

        # Tạo set để theo dõi các nguồn đã hiển thị
        shown_sources = set()
        source_count = 1

        for source in filtered_sources:
            rendered = _render_source_line(source, metadata_dict)
            if not rendered:
                continue

            # Use the rendered block as the uniqueness key
            source_key = rendered
            if source_key in shown_sources:
                continue
            shown_sources.add(source_key)

            sources_text += f"\n{source_count}. {rendered}\n"
            source_count += 1

        logger.info(
            f"Showing {len(shown_sources)} unique sources for legal query based on referenced document IDs"
        )
        return _append_user_document_sources(results, sources_text.rstrip())

    # Nếu không có referenced_doc_ids, sử dụng cách cũ
    # Tạo set để theo dõi các nguồn đã hiển thị
    shown_sources = set()
    source_count = 1
    sources_text = "\nNguồn tham khảo:\n"

    for source in results["sources"]:
        rendered = _render_source_line(source, metadata_dict)
        if not rendered:
            continue
        source_key = rendered
        if source_key in shown_sources:
            continue
        shown_sources.add(source_key)
        sources_text += f"\n{source_count}. {rendered}"
        source_count += 1

    logger.info(
        f"Showing {len(shown_sources)} unique sources for legal query: {query[:50]}..."
    )
    return _append_user_document_sources(results, sources_text.rstrip())


if __name__ == "__main__":
    # Câu hỏi cho trước
    query = "Xin chào"

    # Chạy pipeline
    results = rag_answer_gemini(query)

    # In kết quả
    print("Câu hỏi:", query)
    print(format_answer(results))
    print(format_sources(results))
