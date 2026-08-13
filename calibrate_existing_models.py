import os
import pickle
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.calibration import CalibratedClassifierCV

from preprocessing import clean_text


def try_load(paths):
    for p in paths:
        if os.path.exists(p):
            with open(p, 'rb') as f:
                return pickle.load(f)
    return None


def main(sample_frac=None):
    vec = try_load(['models/tfidf_vectorizer.pkl', 'vectorizer.pkl'])
    if vec is None:
        print('Vectorizer not found — aborting calibration.')
        return

    # Load models
    model_paths = {
        'Logistic Regression': ['models/logistic_regression.pkl', 'lr_model.pkl'],
        'Decision Tree': ['models/decision_tree.pkl', 'dt_model.pkl'],
        'Random Forest': ['models/random_forest.pkl', 'rfc_model.pkl'],
        'Gradient Boosting': ['models/gradient_boosting.pkl', 'gbc_model.pkl'],
    }

    models = {}
    for name, paths in model_paths.items():
        m = try_load(paths)
        if m is None:
            print(f'{name}: model file not found (tried {paths})')
        else:
            models[name] = m

    if not models:
        print('No models to calibrate.')
        return

    # Load dataset and create calibration split (use same seed to avoid leakage)
    if not (os.path.exists('Fake.csv') and os.path.exists('True.csv')):
        print('Dataset files not found — cannot create calibration set.')
        return

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

    X = df['cleaned_text']
    y = df['class']

    # Create trainval/test split, then take calibration from trainval
    X_trainval, X_test, y_trainval, y_test = train_test_split(X, y, test_size=0.20, random_state=42, stratify=y)
    # Use 25% of trainval as calibration (~20% overall)
    X_train, X_calib, y_train, y_calib = train_test_split(X_trainval, y_trainval, test_size=0.25, random_state=42, stratify=y_trainval)

    # Optionally subsample calibration set to make it faster
    if sample_frac is not None and 0 < sample_frac < 1.0:
        calib_idx = X_calib.sample(frac=sample_frac, random_state=42).index
        X_calib = X_calib.loc[calib_idx]
        y_calib = y_calib.loc[calib_idx]

    X_calib_vec = vec.transform(X_calib)

    # Calibrate each loaded model in-place and save
    for name, model in models.items():
        print('\nCalibrating:', name)
        try:
            calib = CalibratedClassifierCV(estimator=model, method='sigmoid', cv='prefit')
            calib.fit(X_calib_vec, y_calib)

            # Save calibrated model overwriting previous
            sub_path = os.path.join('models', name.lower().replace(' ', '_') + '.pkl')
            # maintain existing filenames compatibility
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
            fname = filename_map.get(name)
            rootname = root_map.get(name)
            if fname:
                with open(os.path.join('models', fname), 'wb') as f:
                    pickle.dump(calib, f)
            if rootname:
                with open(rootname, 'wb') as f:
                    pickle.dump(calib, f)

            print(' Saved calibrated model for', name)
        except Exception as e:
            print(' Calibration failed for', name, '-', e)

    print('\nCalibration pass complete.')


if __name__ == '__main__':
    main(sample_frac=0.5)
