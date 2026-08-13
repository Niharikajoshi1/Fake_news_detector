import os
import pickle
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, classification_report
)

from preprocessing import clean_text

def train_and_evaluate():
    print("=" * 60)
    print(" FAKE NEWS DETECTION MODEL TRAINING & EVALUATION PIPELINE ")
    print("=" * 60)

    # 1. Load Datasets
    data_dir = "."
    fake_path = os.path.join(data_dir, "Fake.csv")
    true_path = os.path.join(data_dir, "True.csv")

    if not os.path.exists(fake_path) or not os.path.exists(true_path):
        raise FileNotFoundError("Fake.csv or True.csv dataset files not found!")

    print("\n[1/6] Loading datasets...")
    data_fake = pd.read_csv(fake_path)
    data_true = pd.read_csv(true_path)

    print(f" - Loaded Fake News samples: {len(data_fake):,}")
    print(f" - Loaded True News samples: {len(data_true):,}")

    # Assign class labels: 0 for Fake News, 1 for Authentic / True News
    data_fake["class"] = 0
    data_true["class"] = 1

    # Combine datasets
    df = pd.concat([data_fake, data_true], axis=0, ignore_index=True)

    # Fill NaN values in title/text
    df["title"] = df["title"].fillna("")
    df["text"] = df["text"].fillna("")

    # Combine title and text to ensure model learns both headline and body patterns
    print("\n[2/6] Combining Title + Text and applying NLP Preprocessing...")
    df["full_content"] = df["title"] + " " + df["text"]

    # Remove duplicates based on combined content
    df.drop_duplicates(subset=["full_content"], inplace=True)
    print(f" - Total unique dataset samples after deduplication: {len(df):,}")

    # Clean text
    df["cleaned_text"] = df["full_content"].apply(clean_text)

    # Remove samples that became empty after cleaning
    df = df[df["cleaned_text"].str.strip() != ""].reset_index(drop=True)
    print(f" - Valid non-empty samples for training: {len(df):,}")

    X = df["cleaned_text"]
    y = df["class"]

    # 2. Train-Test Split
    print("\n[3/6] Splitting dataset (80% Train, 20% Test)...")
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.20, random_state=42, stratify=y
    )

    print(f" - Training set size: {len(X_train):,}")
    print(f" - Testing set size : {len(X_test):,}")

    # 3. Feature Extraction (TF-IDF Vectorization)
    print("\n[4/6] Vectorizing text using TF-IDF...", flush=True)
    vectorizer = TfidfVectorizer(
        max_features=5000,
        ngram_range=(1, 2),
        stop_words="english",
        sublinear_tf=True
    )

    X_train_vec = vectorizer.fit_transform(X_train)
    X_test_vec = vectorizer.transform(X_test)
    print(f" - TF-IDF Vocabulary Size: {len(vectorizer.vocabulary_):,} features", flush=True)

    # 4. Model Training & Evaluation Setup
    models = {
        "Logistic Regression": LogisticRegression(
            C=1.0, max_iter=1000, solver="lbfgs", random_state=42
        ),
        "Decision Tree": DecisionTreeClassifier(
            max_depth=20, min_samples_split=5, random_state=42
        ),
        "Random Forest": RandomForestClassifier(
            n_estimators=50, max_depth=15, random_state=42, n_jobs=-1
        ),
        "Gradient Boosting": GradientBoostingClassifier(
            n_estimators=30, max_depth=3, subsample=0.8, learning_rate=0.1, random_state=42
        ),
    }

    metrics_results = {}

    print("\n[5/6] Training & Evaluating 4 ML Models...")
    print("-" * 60)

    for name, model in models.items():
        print(f"\nTraining {name}...")
        model.fit(X_train_vec, y_train)

        # Predictions
        train_preds = model.predict(X_train_vec)
        test_preds = model.predict(X_test_vec)

        # Metrics
        train_acc = accuracy_score(y_train, train_preds)
        test_acc = accuracy_score(y_test, test_preds)
        precision = precision_score(y_test, test_preds)
        recall = recall_score(y_test, test_preds)
        f1 = f1_score(y_test, test_preds)
        cm = confusion_matrix(y_test, test_preds)
        clf_report = classification_report(y_test, test_preds, target_names=["Fake", "Real"], digits=4)

        metrics_results[name] = {
            "model_object": model,
            "train_accuracy": train_acc,
            "accuracy": test_acc,
            "precision": precision,
            "recall": recall,
            "f1_score": f1,
            "confusion_matrix": cm.tolist(),
            "classification_report": clf_report
        }

        print(f"   - Training Accuracy : {train_acc * 100:.2f}%")
        print(f"   - Testing Accuracy  : {test_acc * 100:.2f}%")
        print(f"   - Precision         : {precision * 100:.2f}%")
        print(f"   - Recall            : {recall * 100:.2f}%")
        print(f"   - F1-Score          : {f1 * 100:.2f}%")
        print(f"   - Overfitting Check : Train ({train_acc * 100:.2f}%) vs Test ({test_acc * 100:.2f}%)")

    # 5. Model Comparison & Best Model Selection
    print("\n" + "=" * 60)
    print(" MODEL EVALUATION SUMMARY ")
    print("=" * 60)
    summary_data = []
    best_name = None
    best_f1 = -1.0

    for name, m in metrics_results.items():
        summary_data.append({
            "Model": name,
            "Train Acc (%)": round(m["train_accuracy"] * 100, 2),
            "Test Acc (%)": round(m["accuracy"] * 100, 2),
            "Precision (%)": round(m["precision"] * 100, 2),
            "Recall (%)": round(m["recall"] * 100, 2),
            "F1-Score (%)": round(m["f1_score"] * 100, 2),
        })
        if m["f1_score"] > best_f1:
            best_f1 = m["f1_score"]
            best_name = name

    summary_df = pd.DataFrame(summary_data)
    print(summary_df.to_string(index=False))
    print(f"\nBest Selected Model: {best_name} (F1-Score: {best_f1 * 100:.2f}%)")

    # 6. Save Vectorizer & Models
    print("\n[6/6] Saving Vectorizer & Models to disk...")
    os.makedirs("models", exist_ok=True)

    model_filenames = {
        "Logistic Regression": "logistic_regression.pkl",
        "Decision Tree": "decision_tree.pkl",
        "Random Forest": "random_forest.pkl",
        "Gradient Boosting": "gradient_boosting.pkl"
    }

    root_filenames = {
        "Logistic Regression": "lr_model.pkl",
        "Decision Tree": "dt_model.pkl",
        "Random Forest": "rfc_model.pkl",
        "Gradient Boosting": "gbc_model.pkl"
    }

    # Save TF-IDF Vectorizer
    with open(os.path.join("models", "tfidf_vectorizer.pkl"), "wb") as f:
        pickle.dump(vectorizer, f)
    with open("vectorizer.pkl", "wb") as f:
        pickle.dump(vectorizer, f)

    # Save Models
    for name, m in metrics_results.items():
        sub_path = os.path.join("models", model_filenames[name])
        root_path = root_filenames[name]
        
        with open(sub_path, "wb") as f:
            pickle.dump(m["model_object"], f)
        with open(root_path, "wb") as f:
            pickle.dump(m["model_object"], f)

    # Save Metrics dictionary for Streamlit App UI
    save_metrics = {}
    for name, m in metrics_results.items():
        save_metrics[name] = {
            "train_accuracy": m["train_accuracy"],
            "accuracy": m["accuracy"],
            "precision": m["precision"],
            "recall": m["recall"],
            "f1_score": m["f1_score"],
            "confusion_matrix": m["confusion_matrix"],
            "classification_report": m["classification_report"]
        }
    save_metrics["best_model_name"] = best_name

    with open(os.path.join("models", "model_metrics.pkl"), "wb") as f:
        pickle.dump(save_metrics, f)
    with open("model_metrics.pkl", "wb") as f:
        pickle.dump(save_metrics, f)

    print("\nAll models, vectorizer, and evaluation metrics saved successfully!")
    return save_metrics

if __name__ == "__main__":
    train_and_evaluate()
