import os
import pickle
import argparse

from preprocessing import clean_text


def try_load(paths):
    for p in paths:
        if os.path.exists(p):
            with open(p, "rb") as f:
                return pickle.load(f)
    return None


MODEL_FILES = {
    "Logistic Regression": ["models/logistic_regression.pkl", "lr_model.pkl"],
    "Decision Tree": ["models/decision_tree.pkl", "dt_model.pkl"],
    "Random Forest": ["models/random_forest.pkl", "rfc_model.pkl"],
    "Gradient Boosting": ["models/gradient_boosting.pkl", "gbc_model.pkl"],
}


def main(text):
    vec = try_load(["models/tfidf_vectorizer.pkl", "vectorizer.pkl"])
    if vec is None:
        print("ERROR: Vectorizer not found. Run train_model.py first.")
        return

    cleaned = clean_text(text)
    print("Input cleaned:", cleaned)
    X = vec.transform([cleaned])

    for name, paths in MODEL_FILES.items():
        model = try_load(paths)
        if model is None:
            print(f"{name}: model file not found (tried {paths})")
            continue

        print('\n' + '=' * 60)
        print(f"Model: {name}")
        try:
            classes = getattr(model, 'classes_', None)
            print(" classes_:", classes)

            pred = model.predict(X)[0]
            print(" predicted class:", pred)

            if hasattr(model, 'predict_proba'):
                proba = model.predict_proba(X)[0]
                print(" predict_proba:", proba)
                # Determine confidence using model's classes_
                if classes is not None:
                    # find index of predicted class in classes_
                    try:
                        idx = list(classes).index(pred)
                    except ValueError:
                        idx = 0
                else:
                    idx = pred
                confidence = proba[idx]
                print(f" chosen confidence: {confidence*100:.2f}%")
            else:
                print(" model has no predict_proba, confidence assumed 100%")
        except Exception as e:
            print(" Error during model inference:", e)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Analyze predictions for all saved models')
    parser.add_argument('--text', '-t', help='Input text to classify', required=False)
    args = parser.parse_args()

    sample = args.text or (
        "WASHINGTON (Reuters) - The US Senate passed the fiscal budget bill today after bipartisan discussion."
    )
    main(sample)
