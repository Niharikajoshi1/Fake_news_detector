import os
import pickle
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.calibration import CalibratedClassifierCV
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, classification_report
from scipy.stats import randint, uniform

from preprocessing import clean_text


def load_data():
    if not (os.path.exists('Fake.csv') and os.path.exists('True.csv')):
        raise FileNotFoundError('Dataset files not found.')
    df_fake = pd.read_csv('Fake.csv')
    df_true = pd.read_csv('True.csv')
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


def tune_model(name, estimator, param_dist, X_vec, y, n_iter=8):
    rnd = RandomizedSearchCV(
        estimator,
        param_distributions=param_dist,
        n_iter=n_iter,
        scoring='f1',
        cv=3,
        random_state=42,
        n_jobs=-1,
        verbose=0
    )
    rnd.fit(X_vec, y)
    print(f"{name} best params: {rnd.best_params_}, best score: {rnd.best_score_:.4f}")
    return rnd.best_estimator_



def main():
    print('Loading data...')
    df = load_data()
    X = df['cleaned_text']
    y = df['class']

    # split trainval/test
    X_trainval, X_test, y_trainval, y_test = train_test_split(X, y, test_size=0.20, random_state=42, stratify=y)
    # split train/calib (from trainval)
    X_train, X_calib, y_train, y_calib = train_test_split(X_trainval, y_trainval, test_size=0.20, random_state=42, stratify=y_trainval)

    print(f'Train: {len(X_train):,}, Calib: {len(X_calib):,}, Test: {len(X_test):,}')

    vectorizer = TfidfVectorizer(max_features=5000, ngram_range=(1,2), stop_words='english', sublinear_tf=True)
    X_train_vec = vectorizer.fit_transform(X_train)
    X_calib_vec = vectorizer.transform(X_calib)
    X_test_vec = vectorizer.transform(X_test)

    # Define models and parameter distributions (small search for speed)
    models = {
        'Logistic Regression': (
            LogisticRegression(max_iter=1000, solver='lbfgs', random_state=42),
            {'C': uniform(0.01, 10)}
        ),
        'Decision Tree': (
            DecisionTreeClassifier(random_state=42),
            {'max_depth': randint(5, 30), 'min_samples_split': randint(2, 10)}
        ),
        'Random Forest': (
            RandomForestClassifier(random_state=42, n_jobs=-1),
            {'n_estimators': randint(30, 150), 'max_depth': randint(5, 25)}
        ),
        'Gradient Boosting': (
            GradientBoostingClassifier(random_state=42),
            {'n_estimators': randint(20, 150), 'learning_rate': uniform(0.01, 0.3), 'max_depth': randint(2, 6)}
        )
    }

    trained = {}
    for name, (est, params) in models.items():
        print('\n' + '='*50)
        print('Tuning', name)
        best = tune_model(name, est, params, X_train_vec, y_train, n_iter=8)
        # fit on full trainval (train+calib) for final model, we'll calibrate next
        print('Fitting best estimator on train+calib...')
        best.fit(vectorizer.transform(pd.concat([X_train, X_calib])), pd.concat([y_train, y_calib]))

        # Calibrate using held-out calibration set (X_calib)
        print('Calibrating probabilities...')
        calib = CalibratedClassifierCV(estimator=best, method='sigmoid', cv='prefit')
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

        trained[name] = {
            'model': calib,
            'accuracy': acc,
            'precision': prec,
            'recall': rec,
            'f1_score': f1,
            'confusion_matrix': cm.tolist(),
            'classification_report': cr
        }

        print(f'{name} Test Acc: {acc*100:.2f}%, F1: {f1*100:.2f}%')

    # Save vectorizer and models
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

    metrics = {}
    best_name = None
    best_f1 = -1
    for name, info in trained.items():
        model_obj = info['model']
        fname = filename_map.get(name)
        if fname:
            with open(os.path.join('models', fname), 'wb') as f:
                pickle.dump(model_obj, f)
            with open(fname.replace('.pkl', '_root.pkl'), 'wb') as f:
                pickle.dump(model_obj, f)

        metrics[name] = {
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

    metrics['best_model_name'] = best_name
    with open(os.path.join('models', 'model_metrics.pkl'), 'wb') as f:
        pickle.dump(metrics, f)
    with open('model_metrics.pkl', 'wb') as f:
        pickle.dump(metrics, f)

    print('\nTuning and retraining complete. Best model:', best_name)


if __name__ == '__main__':
    main()
