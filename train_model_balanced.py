"""
Train models using BALANCED dataset (equal fake/real samples).
This removes source bias that was causing 57% confidence on fake news.
Run this after create_balanced_data.py
"""

import os
import pickle
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

def train_on_balanced_data():
    print("=" * 60)
    print(" TRAINING ON BALANCED DATA (50% Fake, 50% Real)")
    print("=" * 60)
    
    # Load balanced dataset
    print("\n[1/5] Loading balanced dataset...")
    if not os.path.exists('balanced_data.csv'):
        print("ERROR: balanced_data.csv not found!")
        print("Run: python create_balanced_data.py")
        return
    
    df = pd.read_csv('balanced_data.csv')
    print(f"  ✓ Loaded {len(df):,} samples")
    print(f"    - Fake (class 0): {len(df[df['class'] == 0]):,}")
    print(f"    - Real (class 1): {len(df[df['class'] == 1]):,}")
    
    X = df['cleaned']
    y = df['class']
    
    # Split
    print("\n[2/5] Splitting data (80% train, 20% test)...")
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.20, random_state=42, stratify=y
    )
    print(f"  ✓ Train: {len(X_train):,}, Test: {len(X_test):,}")
    
    # Vectorize
    print("\n[3/5] Vectorizing with TF-IDF...")
    vectorizer = TfidfVectorizer(
        max_features=5000,
        ngram_range=(1, 2),
        stop_words="english",
        sublinear_tf=True
    )
    X_train_vec = vectorizer.fit_transform(X_train)
    X_test_vec = vectorizer.transform(X_test)
    print(f"  ✓ Vocabulary size: {len(vectorizer.vocabulary_):,}")
    
    # Models
    models = {
        "Logistic Regression": LogisticRegression(max_iter=1000, random_state=42),
        "Decision Tree": DecisionTreeClassifier(max_depth=20, random_state=42),
        "Random Forest": RandomForestClassifier(n_estimators=50, max_depth=15, random_state=42, n_jobs=-1),
        "Gradient Boosting": GradientBoostingClassifier(n_estimators=30, max_depth=3, random_state=42),
    }
    
    # Train
    print("\n[4/5] Training models...")
    metrics = {}
    for name, model in models.items():
        print(f"\n  Training {name}...")
        model.fit(X_train_vec, y_train)
        
        # Metrics
        preds = model.predict(X_test_vec)
        acc = accuracy_score(y_test, preds)
        prec = precision_score(y_test, preds)
        rec = recall_score(y_test, preds)
        f1 = f1_score(y_test, preds)
        
        metrics[name] = {
            "accuracy": acc,
            "precision": prec,
            "recall": rec,
            "f1": f1
        }
        
        print(f"    Accuracy:  {acc*100:.2f}%")
        print(f"    Precision: {prec*100:.2f}%")
        print(f"    Recall:    {rec*100:.2f}%")
        print(f"    F1-Score:  {f1*100:.2f}%")
    
    # Summary
    print("\n[5/5] Model Comparison:")
    print("-" * 60)
    for name, m in sorted(metrics.items(), key=lambda x: x[1]['f1'], reverse=True):
        print(f"{name:20} | Acc: {m['accuracy']*100:6.2f}% | "
              f"F1: {m['f1']*100:6.2f}% | "
              f"Recall: {m['recall']*100:6.2f}%")
    
    best_model = max(metrics.items(), key=lambda x: x[1]['f1'])
    print(f"\n✓ Best Model: {best_model[0]} (F1: {best_model[1]['f1']*100:.2f}%)")
    
    # Save
    print("\nSaving models...")
    os.makedirs('models', exist_ok=True)
    
    for name, model in models.items():
        filename = {
            "Logistic Regression": "logistic_regression.pkl",
            "Decision Tree": "decision_tree.pkl",
            "Random Forest": "random_forest.pkl",
            "Gradient Boosting": "gradient_boosting.pkl",
        }[name]
        
        path = os.path.join('models', filename)
        with open(path, 'wb') as f:
            pickle.dump(model, f)
    
    # Save vectorizer
    with open(os.path.join('models', 'tfidf_vectorizer.pkl'), 'wb') as f:
        pickle.dump(vectorizer, f)
    
    # Save metrics
    metrics['best_model_name'] = best_model[0]
    with open(os.path.join('models', 'model_metrics.pkl'), 'wb') as f:
        pickle.dump(metrics, f)
    
    print("\n✓ All models saved to models/")
    print("\nNow restart your Streamlit app to see improvements!")
    print("The app will auto-load the new models.")

if __name__ == "__main__":
    train_on_balanced_data()
