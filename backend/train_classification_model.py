import json
import logging
import os
import pickle
import random

import fasttext
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    classification_report,
    f1_score,
    make_scorer,
    precision_score,
    recall_score,
)
from sklearn.model_selection import StratifiedKFold, cross_validate
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import SVC, LinearSVC

logger = logging.getLogger(__name__)

# Đường dẫn đến thư mục classification_models
backend_dir = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(backend_dir, "classification_models")
CLASSIFIER_PATH = os.path.join(MODEL_PATH, "question_classifier.pkl")
VECTORIZER_PATH = os.path.join(MODEL_PATH, "tfidf_vectorizer.pkl")
FASTTEXT_PATH = os.path.join(MODEL_PATH, "fasttext_model.bin")


def load_and_prepare_data():
    """Load and prepare training data from JSON files"""

    relevant_questions = []
    with open(
        os.path.join(backend_dir, "json_data", "val_data.json"),
        "r",
        encoding="utf-8",
    ) as f:
        data = json.load(f)
        relevant_questions.extend([item["question"] for item in data])

    with open(
        os.path.join(backend_dir, "json_data", "test_data.json"),
        "r",
        encoding="utf-8",
    ) as f:
        data = json.load(f)
        relevant_questions.extend([item["question"] for item in data])

    relevant_questions = random.sample(relevant_questions, 250)
    irrelevant_questions = []
    irrelevant_questions_path = os.path.join(
        backend_dir, "json_data", "irrelevant_questions.json"
    )
    with open(irrelevant_questions_path, "r", encoding="utf-8") as f:
        irrelevant_data = json.load(f)
        random.shuffle(irrelevant_data)
        irrelevant_questions.extend(irrelevant_data)

    X = relevant_questions + irrelevant_questions
    y = [1] * len(relevant_questions) + [0] * len(irrelevant_questions)

    return X, y


def prepare_fasttext_data(X, y, temp_file="temp_training.txt"):
    """Chuẩn bị dữ liệu cho FastText"""
    with open(temp_file, "w", encoding="utf-8") as f:
        for text, label in zip(X, y):
            # FastText yêu cầu label bắt đầu bằng '__label__'
            f.write(f"__label__{label} {text}\n")
    return temp_file


def train_and_evaluate_models():
    """Train và đánh giá nhiều mô hình khác nhau"""
    # Load và chuẩn bị dữ liệu
    X, y = load_and_prepare_data()

    # TF-IDF Vectorization
    vectorizer = TfidfVectorizer(
        max_features=5000, ngram_range=(1, 2), strip_accents="unicode"
    )
    X_vectorized = vectorizer.fit_transform(X)

    # Định nghĩa scoring metrics
    scoring = {
        "f1": make_scorer(f1_score),
        "precision": make_scorer(precision_score),
        "recall": make_scorer(recall_score),
    }

    # Định nghĩa cross-validation
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

    # Định nghĩa các mô hình
    models = {
        "SVM": SVC(kernel="rbf", C=1.0, class_weight="balanced", probability=True),
        "LinearSVC": LinearSVC(C=1.0, class_weight="balanced", max_iter=1000),
        "Naive Bayes": MultinomialNB(alpha=1.0),
        "Random Forest": RandomForestClassifier(
            n_estimators=50, max_depth=5, class_weight="balanced"
        ),
        "Logistic Regression": LogisticRegression(
            C=1.0, class_weight="balanced", max_iter=1000
        ),
    }

    # Dictionary để lưu kết quả
    results = {}

    # Đánh giá từng mô hình
    for name, model in models.items():
        print(f"\nEvaluating {name}...")
        try:
            scores = cross_validate(
                model,
                X_vectorized,
                y,
                cv=cv,
                scoring=scoring,
                return_train_score=True,
                n_jobs=-1,  # Sử dụng tất cả CPU cores
            )

            results[name] = {
                "f1": scores["test_f1"].mean(),
                "f1_std": scores["test_f1"].std(),
                "precision": scores["test_precision"].mean(),
                "recall": scores["test_recall"].mean(),
            }

            print(f"{name} Results:")
            print(
                f"F1-score: {results[name]['f1']:.4f} (+/- {results[name]['f1_std']*2:.4f})"
            )
            print(f"Precision: {results[name]['precision']:.4f}")
            print(f"Recall: {results[name]['recall']:.4f}")

        except Exception as e:
            print(f"Error evaluating {name}: {str(e)}")

    # Train FastText model separately (vì FastText có API khác)
    print("\nTraining FastText model...")
    try:
        # Chuẩn bị dữ liệu cho FastText
        temp_file = prepare_fasttext_data(X, y)

        # Train FastText model
        model = fasttext.train_supervised(
            input=temp_file, epoch=25, lr=0.1, wordNgrams=2, verbose=2, minCount=1
        )

        # Đánh giá FastText
        result = model.test(temp_file)
        print("\nFastText Results:")
        print(f"Precision: {result[1]:.4f}")
        print(f"Recall: {result[2]:.4f}")

        # Lưu model FastText
        model.save_model(FASTTEXT_PATH)
        logger.info(f"FastText model saved to: {FASTTEXT_PATH}")

        # Xóa file tạm
        os.remove(temp_file)

    except Exception as e:
        print(f"Error training FastText: {str(e)}")

    # Tìm mô hình tốt nhất (dựa trên F1-score)
    best_model_name = max(results, key=lambda k: results[k]["f1"])
    print(f"\nBest model: {best_model_name}")
    print(f"Best F1-score: {results[best_model_name]['f1']:.4f}")

    # Train mô hình tốt nhất trên toàn bộ dữ liệu
    best_model = models[best_model_name]
    best_model.fit(X_vectorized, y)

    # Lưu mô hình và vectorizer
    os.makedirs(MODEL_PATH, exist_ok=True)
    with open(CLASSIFIER_PATH, "wb") as f:
        pickle.dump(best_model, f)
    with open(VECTORIZER_PATH, "wb") as f:
        pickle.dump(vectorizer, f)

    logger.info(f"Best model ({best_model_name}) saved to: {CLASSIFIER_PATH}")
    logger.info(f"Vectorizer saved to: {VECTORIZER_PATH}")

    return best_model, vectorizer, results


def is_specific_legal_question(query):
    """Determine if a query is related to legal documents"""
    try:
        with open(CLASSIFIER_PATH, "rb") as f:
            classifier = pickle.load(f)
        with open(VECTORIZER_PATH, "rb") as f:
            vectorizer = pickle.load(f)
    except Exception as e:
        logger.error(f"Could not load classifier: {e}")
        return None

    try:
        query_vector = vectorizer.transform([query])
        prediction = classifier.predict(query_vector)[0]
        probabilities = classifier.predict_proba(query_vector)[0]
        confidence = probabilities[prediction]

        return prediction == 1 and confidence > 0.7
    except Exception as e:
        logger.error(f"Error in prediction: {e}")
        return None


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
    )

    logger.info("Starting model evaluation...")
    best_model, vectorizer, results = train_and_evaluate_models()

    # In kết quả chi tiết
    print("\nDetailed Results:")
    for model_name, metrics in results.items():
        print(f"\n{model_name}:")
        for metric_name, value in metrics.items():
            print(f"{metric_name}: {value:.4f}")
