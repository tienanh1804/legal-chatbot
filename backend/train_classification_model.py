import json
import logging
import os
import pickle
import random
import re

import numpy as np

try:
    import fasttext
except ImportError:
    fasttext = None  # optional; sklearn pipeline vẫn chạy (pip install fasttext nếu cần)
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    f1_score,
    precision_score,
    recall_score,
)
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import SVC, LinearSVC

logger = logging.getLogger(__name__)

# Đường dẫn đến thư mục classification_models
backend_dir = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(backend_dir, "classification_models")
CLASSIFIER_PATH = os.path.join(MODEL_PATH, "question_classifier.pkl")
VECTORIZER_PATH = os.path.join(MODEL_PATH, "tfidf_vectorizer.pkl")
FASTTEXT_PATH = os.path.join(MODEL_PATH, "fasttext_model.bin")


def _load_json_questions(path: str) -> list[str]:
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    out: list[str] = []
    if not isinstance(data, list):
        return out
    for item in data:
        if isinstance(item, dict) and "question" in item:
            q = str(item["question"]).strip()
            if q:
                out.append(q)
    return out


def _load_irrelevant_questions(path: str) -> list[str]:
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    out: list[str] = []
    if isinstance(data, list):
        for item in data:
            s = str(item).strip()
            if s:
                out.append(s)
    return out


def _normalize_text(text: str) -> str:
    return re.sub(r"\s+", " ", text).strip().lower()


def _dedup_keep_order(items: list[str]) -> list[str]:
    out: list[str] = []
    seen: set[str] = set()
    for item in items:
        key = _normalize_text(item)
        if not key or key in seen:
            continue
        seen.add(key)
        out.append(item.strip())
    return out


def _partition_negatives(
    irrelevant_all: list[str],
    train_count: int,
    val_count: int,
    test_count: int,
    rng: random.Random,
) -> tuple[list[str], list[str], list[str]]:
    pool = _dedup_keep_order(irrelevant_all)
    if len(pool) < (train_count + val_count + test_count):
        raise ValueError(
            "Not enough unique irrelevant samples to build disjoint train/val/test negatives. "
            f"Need {train_count + val_count + test_count}, got {len(pool)}."
        )
    rng.shuffle(pool)
    train_neg = pool[:train_count]
    val_neg = pool[train_count : train_count + val_count]
    test_neg = pool[train_count + val_count : train_count + val_count + test_count]
    return train_neg, val_neg, test_neg


def _build_balanced_split(
    relevant: list[str],
    irrelevant_split: list[str],
) -> tuple[list[str], list[int]]:
    """
    Build a balanced binary classification dataset:
    - label 1: relevant questions
    - label 0: sampled irrelevant questions (same count as relevant)
    """
    relevant = _dedup_keep_order([q for q in relevant if q])
    if not relevant:
        raise ValueError("Relevant split is empty.")

    relevant_keys = {_normalize_text(q) for q in relevant}
    irrelevant = [
        q for q in _dedup_keep_order(irrelevant_split) if _normalize_text(q) not in relevant_keys
    ]
    if len(irrelevant) < len(relevant):
        raise ValueError(
            "Irrelevant split has too few unique samples after filtering overlap with positives. "
            f"Need {len(relevant)}, got {len(irrelevant)}."
        )
    irrelevant = irrelevant[: len(relevant)]
    X = relevant + irrelevant
    y = [1] * len(relevant) + [0] * len(irrelevant)
    return X, y


def prepare_fasttext_data(X, y, temp_file="temp_training.txt"):
    """Chuẩn bị dữ liệu cho FastText"""
    with open(temp_file, "w", encoding="utf-8") as f:
        for text, label in zip(X, y):
            # FastText yêu cầu label bắt đầu bằng '__label__'
            f.write(f"__label__{label} {text}\n")
    return temp_file


def train_and_evaluate_models():
    """Train/Val/Test split pipeline (no leakage)."""
    rng = random.Random(42)

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

    # Load splits
    relevant_train = _load_json_questions(os.path.join(backend_dir, "json_data", "train_data.json"))
    relevant_val = _load_json_questions(os.path.join(backend_dir, "json_data", "val_data.json"))
    relevant_test = _load_json_questions(os.path.join(backend_dir, "json_data", "test_data.json"))

    irrelevant_all = _load_irrelevant_questions(
        os.path.join(backend_dir, "json_data", "irrelevant_questions.json")
    )

    relevant_train = _dedup_keep_order(relevant_train)
    relevant_val = _dedup_keep_order(relevant_val)
    relevant_test = _dedup_keep_order(relevant_test)

    train_neg, val_neg, test_neg = _partition_negatives(
        irrelevant_all=irrelevant_all,
        train_count=len(relevant_train),
        val_count=len(relevant_val),
        test_count=len(relevant_test),
        rng=rng,
    )

    X_train, y_train = _build_balanced_split(relevant_train, train_neg)
    X_val, y_val = _build_balanced_split(relevant_val, val_neg)
    X_test, y_test = _build_balanced_split(relevant_test, test_neg)

    # TF-IDF fit ONLY on train to avoid leakage
    vectorizer = TfidfVectorizer(
        max_features=5000, ngram_range=(1, 2), strip_accents="unicode"
    )
    X_train_vec = vectorizer.fit_transform(X_train)
    X_val_vec = vectorizer.transform(X_val)
    X_test_vec = vectorizer.transform(X_test)

    def _eval(model, X_vec, y_true) -> dict[str, float]:
        y_pred = model.predict(X_vec)
        return {
            "f1": float(f1_score(y_true, y_pred, pos_label=1, zero_division=0)),
            "precision": float(precision_score(y_true, y_pred, pos_label=1, zero_division=0)),
            "recall": float(recall_score(y_true, y_pred, pos_label=1, zero_division=0)),
        }

    # Evaluate each model on validation, choose best by val F1
    results: dict[str, dict[str, float]] = {}
    best_model_name = None
    best_val_f1 = -1.0
    for name, model in models.items():
        print(f"\nEvaluating {name} (Val)...")
        try:
            model.fit(X_train_vec, y_train)
            val_metrics = _eval(model, X_val_vec, y_val)
            results[name] = val_metrics

            print(
                f"{name} Val: F1={val_metrics['f1']:.4f} "
                f"Precision={val_metrics['precision']:.4f} Recall={val_metrics['recall']:.4f}"
            )
            if val_metrics["f1"] > best_val_f1:
                best_val_f1 = val_metrics["f1"]
                best_model_name = name
        except Exception as e:
            print(f"Error evaluating {name}: {e}")

    if best_model_name is None:
        raise RuntimeError("No model could be trained/evaluated.")

    print(f"\nBest model by Val F1: {best_model_name} (F1={best_val_f1:.4f})")

    # Retrain best model on train+val (still using vectorizer fit on train)
    best_model = models[best_model_name]
    X_trainval = X_train + X_val
    y_trainval = y_train + y_val
    X_trainval_vec = vectorizer.transform(X_trainval)
    best_model.fit(X_trainval_vec, y_trainval)

    # Final evaluation on test
    test_metrics = _eval(best_model, X_test_vec, y_test)
    results[best_model_name].update({f"test_{k}": v for k, v in test_metrics.items()})
    print(
        f"\n{best_model_name} Test: "
        f"F1={test_metrics['f1']:.4f} "
        f"Precision={test_metrics['precision']:.4f} "
        f"Recall={test_metrics['recall']:.4f}"
    )

    # Train FastText model separately (optional). Use the same train data split.
    if fasttext is None:
        print(
            "\nSkip FastText: package not available. "
            "Sklearn classifier vẫn được huấn luyện và lưu."
        )
    else:
        print("\nTraining FastText model (train split)...")
        try:
            temp_file = prepare_fasttext_data(X_train, y_train, temp_file="temp_training_fasttext.txt")
            ft_model = fasttext.train_supervised(
                input=temp_file,
                epoch=25,
                lr=0.1,
                wordNgrams=2,
                verbose=2,
                minCount=1,
            )
            ft_model.save_model(FASTTEXT_PATH)
            logger.info(f"FastText model saved to: {FASTTEXT_PATH}")
            os.remove(temp_file)
        except Exception as e:
            print(f"Error training FastText: {e}")

    # Save sklearn artifacts
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
        confidence = None
        if hasattr(classifier, "predict_proba"):
            probabilities = classifier.predict_proba(query_vector)[0]
            confidence = float(probabilities[prediction])
            return prediction == 1 and confidence > 0.7

        # Fallback for models without predict_proba (e.g., LinearSVC)
        if hasattr(classifier, "decision_function"):
            score = float(classifier.decision_function(query_vector)[0])
            # Map score -> [0,1] via sigmoid for a usable threshold.
            confidence = 1.0 / (1.0 + np.exp(-score))
            return prediction == 1 and confidence > 0.7

        # If no confidence available, rely on hard prediction only.
        return prediction == 1
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
