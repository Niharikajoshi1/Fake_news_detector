import os
import pickle
import numpy as np
import pandas as pd
from preprocessing import clean_text

class FakeNewsPredictor:
    def __init__(self, model_dir="models"):
        self.model_dir = model_dir
        self.vectorizer = None
        self.models = {}
        self.best_model_name = "Logistic Regression"
        self.metrics = {}
        self.is_loaded = False

        self._load_components()

    def _load_components(self):
        try:
            # 1. Load Vectorizer (Try models/ first, then root)
            vec_path = os.path.join(self.model_dir, "tfidf_vectorizer.pkl")
            if not os.path.exists(vec_path):
                vec_path = "vectorizer.pkl"
            
            with open(vec_path, "rb") as f:
                self.vectorizer = pickle.load(f)

            # 2. Map of Model Names to File Paths
            model_files = {
                "Logistic Regression": ["models/logistic_regression.pkl", "lr_model.pkl"],
                "Decision Tree": ["models/decision_tree.pkl", "dt_model.pkl"],
                "Random Forest": ["models/random_forest.pkl", "rfc_model.pkl"],
                "Gradient Boosting": ["models/gradient_boosting.pkl", "gbc_model.pkl"]
            }

            for name, paths in model_files.items():
                loaded = False
                for p in paths:
                    if os.path.exists(p):
                        with open(p, "rb") as f:
                            self.models[name] = pickle.load(f)
                            loaded = True
                            break
                if not loaded:
                    raise FileNotFoundError(f"Model file for {name} not found.")

            # 3. Load Model Metrics if available
            metrics_path = os.path.join(self.model_dir, "model_metrics.pkl")
            if not os.path.exists(metrics_path):
                metrics_path = "model_metrics.pkl"

            if os.path.exists(metrics_path):
                with open(metrics_path, "rb") as f:
                    self.metrics = pickle.load(f)
                    self.best_model_name = self.metrics.get("best_model_name", "Logistic Regression")

            self.is_loaded = True
        except Exception as e:
            self.is_loaded = False
            self.load_error = str(e)

    def predict(self, text: str, model_name: str = "Best Model"):
        """
        Predicts authenticity for a single news text/headline string.
        Returns dictionary containing:
        - label: 'REAL NEWS' or 'FAKE NEWS'
        - confidence: Percentage confidence (float)
        - class_id: 1 for Real, 0 for Fake
        - proba_fake: Probability of Fake News
        - proba_real: Probability of Real News
        - model_used: Name of model utilized
        """
        if not self.is_loaded:
            raise RuntimeError(f"Models not loaded properly: {getattr(self, 'load_error', 'Unknown error')}")

        if not text or not str(text).strip():
            return {
                "error": "Input text is empty. Please enter a valid news article or headline."
            }

        cleaned = clean_text(text)
        if not cleaned:
            return {
                "error": "Input text contains no recognizable words after preprocessing."
            }

        vectorized = self.vectorizer.transform([cleaned])

        # Resolve Model Choice
        if model_name == "Best Model" or model_name not in self.models:
            selected_name = self.best_model_name if self.best_model_name in self.models else "Logistic Regression"
        else:
            selected_name = model_name

        model = self.models[selected_name]
        pred_class = model.predict(vectorized)[0]

        # Calculate Confidence Score using predict_proba if available
        if hasattr(model, "predict_proba"):
            probs = model.predict_proba(vectorized)[0]
            classes = getattr(model, "classes_", None)
            if classes is not None:
                prob_map = {cls: float(probs[i]) for i, cls in enumerate(classes)}
                proba_fake = float(prob_map.get(0, 0.0))
                proba_real = float(prob_map.get(1, 0.0))
            else:
                proba_fake = float(probs[0])
                proba_real = float(probs[1])
            confidence = proba_real if pred_class == 1 else proba_fake
        else:
            proba_fake = 1.0 if pred_class == 0 else 0.0
            proba_real = 1.0 if pred_class == 1 else 0.0
            confidence = 1.0

        label = "REAL NEWS" if pred_class == 1 else "FAKE NEWS"

        return {
            "label": label,
            "confidence": round(confidence * 100, 2),
            "class_id": int(pred_class),
            "proba_fake": round(proba_fake * 100, 2),
            "proba_real": round(proba_real * 100, 2),
            "model_used": selected_name,
            "cleaned_text": cleaned
        }

    def predict_batch(self, texts: list, model_name: str = "Best Model"):
        """
        Predicts authenticity for a list of news text strings.
        """
        if not self.is_loaded:
            raise RuntimeError("Models not loaded properly.")

        if model_name == "Best Model" or model_name not in self.models:
            selected_name = self.best_model_name if self.best_model_name in self.models else "Logistic Regression"
        else:
            selected_name = model_name

        model = self.models[selected_name]

        cleaned_texts = [clean_text(t) for t in texts]
        vectors = self.vectorizer.transform(cleaned_texts)

        preds = model.predict(vectors)

        if hasattr(model, "predict_proba"):
            probs = model.predict_proba(vectors)
            classes = getattr(model, "classes_", None)
            confidences = []
            if classes is not None:
                for i, pred in enumerate(preds):
                    # find index of predicted class label in classes_
                    try:
                        idx = list(classes).index(pred)
                        confidences.append(float(probs[i][idx]))
                    except ValueError:
                        confidences.append(0.0)
            else:
                confidences = [float(probs[i][pred]) for i, pred in enumerate(preds)]
        else:
            confidences = [1.0] * len(preds)

        results = []
        for i in range(len(texts)):
            label = "REAL NEWS" if preds[i] == 1 else "FAKE NEWS"
            results.append({
                "original_text": texts[i],
                "cleaned_text": cleaned_texts[i],
                "label": label,
                "confidence": round(confidences[i] * 100, 2),
                "model_used": selected_name
            })

        return results


if __name__ == "__main__":
    predictor = FakeNewsPredictor()
    if predictor.is_loaded:
        sample_real = "WASHINGTON (Reuters) - The US Senate passed the fiscal budget bill today after bipartisan discussion."
        sample_fake = "BREAKING: Secret Alien Spaceship Discovered in Sahara Desert! Shocking Photos Inside!"
        
        print("Test Real Sample:", predictor.predict(sample_real))
        print("Test Fake Sample:", predictor.predict(sample_fake))
    else:
        print("Models not loaded:", getattr(predictor, 'load_error', 'Unknown'))
