# Fake_news_detector
# 📰 Fake News Detection Using Machine Learning

## 📌 Project Overview

This project is a Machine Learning-based Fake News Detection system that classifies news articles as **Fake** or **Real**.

The project uses Natural Language Processing (NLP) techniques to convert news text into numerical features and applies multiple Machine Learning classification algorithms to identify whether a news article is fake or genuine.

## 🎯 Objectives

- Detect fake and real news automatically.
- Apply Natural Language Processing techniques to textual data.
- Train and compare multiple Machine Learning models.
- Evaluate model performance using different evaluation metrics.
- Visualize model performance using a confusion matrix.

## 🤖 Machine Learning Models Used

The following classification algorithms were implemented:

- Logistic Regression
- Decision Tree Classifier
- Random Forest Classifier
- Gradient Boosting Classifier

## 🛠️ Technologies & Libraries

- Python
- Pandas
- NumPy
- Scikit-learn
- Matplotlib
- Natural Language Processing (NLP)
- Google Colab
- Jupyter Notebook

## 📊 Model Evaluation

The models were evaluated using:

- Accuracy
- Precision
- Recall
- F1 Score
- Confusion Matrix

An F1 Score of **1.0** was obtained for the evaluated model.

> Note: Model performance can vary depending on the dataset, preprocessing, train-test split, and evaluation methodology.

## 📂 Dataset

The project uses two datasets:

- `Fake.csv` – Fake news articles
- `True.csv` – Real news articles

The original datasets are relatively large and therefore are not included directly in this GitHub repository.

The complete dataset can be loaded separately when running the notebook.

## 🔄 Project Workflow

1. Load the news datasets
2. Combine and label the data
3. Clean and preprocess the text
4. Split the dataset into training and testing sets
5. Convert text into numerical features
6. Train Machine Learning models
7. Make predictions
8. Evaluate model performance
9. Generate a confusion matrix
10. Compare model results

## 📈 Results

The trained Machine Learning models demonstrated strong performance in distinguishing between fake and real news.

The notebook contains the complete implementation, model evaluation, and visualizations.

## ▶️ How to Run

### Using Google Colab

1. Download or clone this repository.
2. Open `Fake_News_detector.ipynb` in Google Colab.
3. Upload the required `Fake.csv` and `True.csv` datasets.
4. Run the notebook cells sequentially.
5. View the model results and evaluation metrics.

## 📁 Project Structure

```text
Fake_news_detector/
│
├── Fake_News_detector.ipynb
├── README.md
└── .gitignore
├── README.md
└── .gitignore
