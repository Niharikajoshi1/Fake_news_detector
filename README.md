# TruthGuard AI — Fake News Detection

A B.Tech CSE-AI project using Natural Language Processing and Machine Learning to classify news articles as Real or Fake.

## Project Structure

```
Fake news detector/
├── app.py                  # Streamlit web application
├── train_model.py          # Full ML training pipeline
├── predict.py              # Standalone inference module
├── preprocessing.py        # Shared NLP preprocessing
├── requirements.txt        # Python dependencies
├── Fake.csv                # Fake news dataset (23,481 articles)
├── True.csv                # Real news dataset (21,417 articles)
├── vectorizer.pkl          # Saved TF-IDF vectorizer
├── lr_model.pkl            # Logistic Regression model
├── dt_model.pkl            # Decision Tree model
├── rfc_model.pkl           # Random Forest model
├── gbc_model.pkl           # Gradient Boosting model
├── model_metrics.pkl       # Evaluation metrics for UI
└── models/                 # Backup copies of all saved models
    ├── tfidf_vectorizer.pkl
    ├── logistic_regression.pkl
    ├── decision_tree.pkl
    ├── random_forest.pkl
    ├── gradient_boosting.pkl
    └── model_metrics.pkl
```

## Model Performance (Test Set — 20% holdout)

| Model               | Test Accuracy | Precision | Recall | F1 Score |
|---------------------|:-------------:|:---------:|:------:|:--------:|
| **Logistic Regression** (Best) | **99.04%** | **98.74%** | **99.50%** | **99.12%** |
| Random Forest       | 97.46%        | 95.85%    | 99.62% | 97.70%   |
| Decision Tree       | 96.36%        | 96.67%    | 96.60% | 96.64%   |
| Gradient Boosting   | 95.64%        | 96.52%    | 95.40% | 95.95%   |

## Root Cause of Previous LR Performance Drop

The Logistic Regression model was previously performing at 40-50% on user inputs because:

1. **Title vs. Full Text Mismatch** — The original pipeline trained only on `text` column body paragraphs. When users input news *headlines* (short text), the TF-IDF feature vector had almost no matches, causing near-random 50/50 predictions.

2. **Publisher Signature Data Leakage** — 99.8% of authentic articles began with `CITY (Reuters) -`. Models overfit to the word "reuters" as a discriminative feature. When articles without this prefix were tested, accuracy dropped severely.

**Fix Applied:** The new pipeline:
- Combines `Title + " " + Text` before cleaning, so models evaluate both headline and body
- Strips publisher source prefixes (`CITY (Reuters) -`) during preprocessing
- Uses TF-IDF with `max_features=5000`, `ngram_range=(1,2)`, `stop_words='english'`, `sublinear_tf=True`

## Setup & Installation

```bash
pip install -r requirements.txt
```

## Training Models

```bash
python train_model.py
```

## Running the Streamlit App

```bash
streamlit run app.py
```

## Running in Google Colab

1. Upload all project files to Colab or mount Google Drive.
2. Install dependencies:
   ```python
   !pip install streamlit pandas numpy scikit-learn
   ```
3. Train models:
   ```python
   !python train_model.py
   ```
4. For Colab frontend, use `localtunnel` or `ngrok`:
   ```python
   !streamlit run app.py &
   !npx localtunnel --port 8501
   ```

## Dataset Sources

- **Fake.csv** — 23,481 fake/misleading news articles (various topics)
- **True.csv** — 21,417 authentic Reuters news articles

**Note:** Models perform best on standard English news articles. Very short inputs (< 5 words) may produce less reliable predictions.
