import os
import pickle
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.calibration import CalibratedClassifierCV
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, classification_report

from preprocessing import clean_text


def load_datasets(fake_path='Fake.csv', true_path='True.csv'):
    if not os.path.exists(fake_path) or not os.path.exists(true_path):
        raise FileNotFoundError('Dataset files not found in workspace root.')
    df_fake = pd.read_csv(fake_path)
    df_true = pd.read_csv(true_path)
    df_fake['class'] = 0
    df_true['class'] = 1
    df = pd.concat([df_fake, df_true], ignore_index=True)
    df['title'] = df['title'].fillna('')
    df['text'] = df['text'].fillna('')
    df['full_content'] = df['title'] + ' ' + df['text']
    df.drop_duplicates(subset=['full_content'], inplace=True)
    df['cleaned_text'] = df['full_content'].apply(clean_text)
    df = df[df['cleaned_text'].str.strip() != ''].reset_index(drop=True)
    return df


def train_and_calibrate():
    print('Loading and preparing data...')
    df = load_datasets()
    X = df['cleaned_text']
    y = df['class']

    # Split into train_val and test (80/20)
    X_trainval, X_test, y_trainval, y_test = train_test_split(
        X, y, test_size=0.20, random_state=42, stratify=y
    )

    # From trainval, set aside calibration set (25% of trainval => 20% overall)
    X_train, X_calib, y_train, y_calib = train_test_split(
        X_trainval, y_trainval, test_size=0.25, random_state=42, stratify=y_trainval
    )

    print(f'Train size: {len(X_train):,}, Calib size: {len(X_calib):,}, Test size: {len(X_test):,}')

    # Vectorize using TF-IDF fit on training portion only
    vectorizer = TfidfVectorizer(max_features=5000, ngram_range=(1,2), stop_words='english', sublinear_tf=True)
    X_train_vec = vectorizer.fit_transform(X_train)
    X_calib_vec = vectorizer.transform(X_calib)
    X_test_vec = vectorizer.transform(X_test)

    models = {
        'Logistic Regression': LogisticRegression(C=1.0, max_iter=1000, solver='lbfgs', random_state=42),
        'Decision Tree': DecisionTreeClassifier(max_depth=20, min_samples_split=5, random_state=42),
        'Random Forest': RandomForestClassifier(n_estimators=50, max_depth=15, random_state=42, n_jobs=-1),
        'Gradient Boosting': GradientBoostingClassifier(n_estimators=30, max_depth=3, subsample=0.8, learning_rate=0.1, random_state=42),
    }

    results = {}

    for name, base in models.items():
        print('\n' + '='*60)
        print('Training base model:', name)
        base.fit(X_train_vec, y_train)

        # Calibrate using held-out calibration set
        print(' Calibrating probabilities (sigmoid)...')
        calib = CalibratedClassifierCV(base_estimator=base, method='sigmoid', cv='prefit')
        calib.fit(X_calib_vec, y_calib)

        # Evaluate on test set
        preds = calib.predict(X_test_vec)
        probas = calib.predict_proba(X_test_vec) if hasattr(calib, 'predict_proba') else None

        acc = accuracy_score(y_test, preds)
        prec = precision_score(y_test, preds)
        rec = recall_score(y_test, preds)
        f1 = f1_score(y_test, preds)
        cm = confusion_matrix(y_test, preds)
        cr = classification_report(y_test, preds, target_names=['Fake', 'Real'], digits=4)

        results[name] = {
            'model_object': calib,
            'train_samples': len(X_train),
            'calib_samples': len(X_calib),
            'test_samples': len(X_test),
            'accuracy': acc,
            'precision': prec,
            'recall': rec,
            'f1_score': f1,
            'confusion_matrix': cm.tolist(),
            'classification_report': cr
        }

        print(f" {name} Test Accuracy: {acc*100:.2f}%, F1: {f1*100:.2f}%")

    # Save vectorizer and calibrated models
    os.makedirs('models', exist_ok=True)

    with open(os.path.join('models', 'tfidf_vectorizer.pkl'), 'wb') as f:
        pickle.dump(vectorizer, f)
    with open('vectorizer.pkl', 'wb') as f:
        pickle.dump(vectorizer, f)

    filename_map = {
        'Logistic Regression': 'logistic_regression.pkl',
        'Decision Tree': 'decision_tree.pkl',
        'Random Forest': 'random_forest.pkl',
        'Gradient Boosting': 'gradient_boosting.pkl'
    }
    root_map = {
        'Logistic Regression': 'lr_model.pkl',
        'Decision Tree': 'dt_model.pkl',
        'Random Forest': 'rfc_model.pkl',
        'Gradient Boosting': 'gbc_model.pkl'
    }

    metrics_save = {}
    best_name = None
    best_f1 = -1.0

    for name, info in results.items():
        model_obj = info['model_object']
        sub_path = os.path.join('models', filename_map[name])
        root_path = root_map[name]
        with open(sub_path, 'wb') as f:
            pickle.dump(model_obj, f)
        with open(root_path, 'wb') as f:
            pickle.dump(model_obj, f)

        metrics_save[name] = {
            'train_accuracy': None,
            'accuracy': info['accuracy'],
            'precision': info['precision'],
            'recall': info['recall'],
            'f1_score': info['f1_score'],
            'confusion_matrix': info['confusion_matrix'],
            'classification_report': info['classification_report']
        }
        if info['f1_score'] > best_f1:
            best_f1 = info['f1_score']
            best_name = name

    metrics_save['best_model_name'] = best_name

    with open(os.path.join('models', 'model_metrics.pkl'), 'wb') as f:
        pickle.dump(metrics_save, f)
    with open('model_metrics.pkl', 'wb') as f:
        pickle.dump(metrics_save, f)

    print('\nCalibration and saving complete. Best model:', best_name)
    return metrics_save


if __name__ == '__main__':
    train_and_calibrate()
