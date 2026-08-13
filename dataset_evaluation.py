import os
import pickle
import pandas as pd
import numpy as np
from sklearn.metrics import accuracy_score, brier_score_loss
from preprocessing import clean_text


def load_data():
    fake = pd.read_csv("Fake.csv")
    true = pd.read_csv("True.csv")
    fake['class'] = 0
    true['class'] = 1
    df = pd.concat([fake, true], ignore_index=True)
    df['title'] = df['title'].fillna("")
    df['text'] = df['text'].fillna("")
    df['full_content'] = df['title'] + " " + df['text']
    df.drop_duplicates(subset=['full_content'], inplace=True)
    df['cleaned_text'] = df['full_content'].apply(clean_text)
    df = df[df['cleaned_text'].str.strip() != ""].reset_index(drop=True)
    return df


def main():
    print("Loading data and models...")
    df = load_data()
    X = df['cleaned_text']
    y = df['class'].values

    # load vectorizer
    with open(os.path.join('models', 'tfidf_vectorizer.pkl'), 'rb') as f:
        vectorizer = pickle.load(f)
    X_vec = vectorizer.transform(X)

    model_files = {
        'Logistic Regression': os.path.join('models','logistic_regression.pkl'),
        'Decision Tree': os.path.join('models','decision_tree.pkl'),
        'Random Forest': os.path.join('models','random_forest.pkl'),
        'Gradient Boosting': os.path.join('models','gradient_boosting.pkl')
    }

    results = {}
    for name, path in model_files.items():
        if not os.path.exists(path):
            print(f"Model file missing: {path}, skipping {name}")
            continue
        with open(path, 'rb') as f:
            model = pickle.load(f)

        # Ensure model supports predict_proba
        if not hasattr(model, 'predict_proba'):
            print(f"Model {name} has no predict_proba, skipping calibration metrics.")
            preds = model.predict(X_vec)
            acc = accuracy_score(y, preds)
            results[name] = {'accuracy': acc, 'avg_confidence': None, 'brier_score': None}
            continue

        probs = model.predict_proba(X_vec)
        classes = list(model.classes_)
        # find index for real(1) and fake(0)
        idx_real = classes.index(1) if 1 in classes else None
        idx_fake = classes.index(0) if 0 in classes else None

        preds = model.predict(X_vec)
        # per-sample chosen-class probability
        chosen_probs = np.zeros(len(preds))
        prob_real = np.zeros(len(preds))
        for i, p in enumerate(preds):
            if p == 1 and idx_real is not None:
                chosen_probs[i] = probs[i, idx_real]
            elif p == 0 and idx_fake is not None:
                chosen_probs[i] = probs[i, idx_fake]
            else:
                # fallback: max prob
                chosen_probs[i] = probs[i].max()
            # prob for Brier (probability of positive class)
            prob_real[i] = probs[i, idx_real] if idx_real is not None else 0.0

        acc = accuracy_score(y, preds)
        avg_conf = float(np.mean(chosen_probs))
        brier = float(brier_score_loss(y, prob_real))

        results[name] = {
            'accuracy': acc,
            'avg_confidence': avg_conf,
            'brier_score': brier,
            'n_samples': len(y)
        }

    # print results
    print('\nPer-model dataset evaluation:')
    for name, r in results.items():
        print(f"- {name}: accuracy={r['accuracy']*100:.2f}%, avg_confidence={r['avg_confidence'] if r['avg_confidence'] is not None else 'N/A'}, brier={r['brier_score']:.4f}")

    # Save results
    with open('models/dataset_evaluation.pkl', 'wb') as f:
        pickle.dump(results, f)
    print('\nSaved evaluation to models/dataset_evaluation.pkl')


if __name__ == '__main__':
    main()
