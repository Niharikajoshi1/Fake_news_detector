import pandas as pd
import pickle
from preprocessing import clean_text

# Load data
df_fake = pd.read_csv('Fake.csv')
df_true = pd.read_csv('True.csv')

print(f"Fake.csv rows: {len(df_fake)}")
print(f"True.csv rows: {len(df_true)}")

print("\n=== FAKE NEWS SAMPLES ===")
for i in range(3):
    title = df_fake['title'].iloc[i][:80]
    text = df_fake['text'].iloc[i][:80]
    print(f"Sample {i+1}: {title}...")

print("\n=== TRUE NEWS SAMPLES ===")
for i in range(3):
    title = df_true['title'].iloc[i][:80]
    text = df_true['text'].iloc[i][:80]
    print(f"Sample {i+1}: {title}...")

# Load model and vectorizer
with open('models/tfidf_vectorizer.pkl', 'rb') as f:
    vec = pickle.load(f)
with open('models/logistic_regression.pkl', 'rb') as f:
    model = pickle.load(f)

# Test one fake and one true
fake_text = (df_fake['title'].iloc[0] + " " + df_fake['text'].iloc[0])
true_text = (df_true['title'].iloc[0] + " " + df_true['text'].iloc[0])

print("\n=== TESTING ===")
print(f"Fake news prediction: {model.predict(vec.transform([clean_text(fake_text)]))[0]}")
print(f"True news prediction: {model.predict(vec.transform([clean_text(true_text)]))[0]}")
print("\nExpected: Fake=0, True=1")
