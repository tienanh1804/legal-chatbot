import logging
import os
import pickle

import fasttext
import numpy as np

# Cấu hình logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

# Kiểm tra xem có thể sử dụng mô hình phân loại không
CLASSIFIER_AVAILABLE = True
FASTTEXT_AVAILABLE = True
try:
    # Kiểm tra xem có file mô hình phân loại không
    backend_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    classifier_path = os.path.join(
        backend_dir, "classification_models", "question_classifier.pkl"
    )
    vectorizer_path = os.path.join(
        backend_dir, "classification_models", "tfidf_vectorizer.pkl"
    )
    fasttext_path = os.path.join(
        backend_dir, "classification_models", "fasttext_model.bin"
    )

    if os.path.exists(classifier_path) and os.path.exists(vectorizer_path):
        logger.info("Classification model files found")
    else:
        logger.warning("Classification model files not found")
        CLASSIFIER_AVAILABLE = False

    if os.path.exists(fasttext_path):
        logger.info("FastText model file found")
    else:
        logger.warning("FastText model file not found")
        FASTTEXT_AVAILABLE = False

except Exception as e:
    logger.warning(f"Error checking classification models: {e}")
    CLASSIFIER_AVAILABLE = False
    FASTTEXT_AVAILABLE = False


if CLASSIFIER_AVAILABLE or FASTTEXT_AVAILABLE:
    # Đường dẫn đến các file mô hình
    backend_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    CLASSIFIER_PATH = os.path.join(
        backend_dir, "classification_models", "question_classifier.pkl"
    )
    VECTORIZER_PATH = os.path.join(
        backend_dir, "classification_models", "tfidf_vectorizer.pkl"
    )
    FASTTEXT_MODEL_PATH = os.path.join(
        backend_dir, "classification_models", "fasttext_model.bin"
    )


def prediction(query="Xin chào bạn có khỏe không?"):
    """Test dự đoán với một câu query cụ thể"""
    logger.info(f"\nTesting prediction for query: '{query}'")

    # Load các models đã train
    try:
        # Ưu tiên sử dụng FastText model nếu có
        if FASTTEXT_AVAILABLE:
            fasttext_model = fasttext.load_model(FASTTEXT_MODEL_PATH)
            ft_prediction = fasttext_model.predict(query)[0][0]  # Lấy label
            ft_prob = fasttext_model.predict(query)[1][0]  # Lấy xác suất
            prediction = ft_prediction.replace("__label__", "")
            confidence = ft_prob
            model_type = "FastText"
            logger.info(f"FastText prediction: {prediction}")
            logger.info(f"FastText probability: {confidence:.4f}")
        elif CLASSIFIER_AVAILABLE:
            # Load model chính (best model)
            with open(CLASSIFIER_PATH, "rb") as f:
                best_model = pickle.load(f)
            with open(VECTORIZER_PATH, "rb") as f:
                vectorizer = pickle.load(f)

            # Vectorize câu query
            query_vector = vectorizer.transform([query])

            # Dự đoán với best model
            prediction = best_model.predict(query_vector)[0]

            # Lấy xác suất dự đoán nếu model hỗ trợ
            confidence = None
            if hasattr(best_model, "predict_proba"):
                probabilities = best_model.predict_proba(query_vector)[0]
                confidence = probabilities[prediction]
            elif hasattr(best_model, "decision_function"):
                decision_scores = best_model.decision_function(query_vector)[0]
                if isinstance(decision_scores, np.ndarray):
                    score = decision_scores[0] if decision_scores.size > 0 else 0
                else:
                    score = decision_scores
                confidence = 0.5 + min(max(score / 2.0, 0), 0.5)

            model_type = type(best_model).__name__
            logger.info(f"Best model prediction: {prediction}")
            if confidence is not None:
                logger.info(f"Confidence: {confidence:.4f}")
        else:
            logger.error("No classification models available")
            return None

        # Kết luận cuối cùng
        is_legal = int(prediction) == 1 and (confidence is None or confidence > 0.7)
        logger.info(
            f"\nFinal conclusion: This query is{' ' if is_legal else ' NOT '}a legal question"
        )

        return {
            "query": query,
            "is_legal_question": is_legal,
            "prediction": int(prediction),
            "confidence": confidence,
            "model_type": model_type,
        }

    except Exception as e:
        logger.error(f"Error in prediction: {e}")
        return None


if __name__ == "__main__":
    prediction("câu hỏi trước của tôi là gì")
