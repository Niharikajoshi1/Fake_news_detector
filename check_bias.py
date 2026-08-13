import pandas as pd

fake = pd.read_csv('Fake.csv')
true = pd.read_csv('True.csv')

# Check for Trump mentions
trump_in_fake = fake['title'].str.lower().str.contains('trump', na=False).sum()
trump_in_true = true['title'].str.lower().str.contains('trump', na=False).sum()

print(f"Trump mentions in Fake.csv: {trump_in_fake} / {len(fake)} ({trump_in_fake/len(fake)*100:.1f}%)")
print(f"Trump mentions in True.csv: {trump_in_true} / {len(true)} ({trump_in_true/len(true)*100:.1f}%)")

# Check for Reuters mentions
reuters_fake = fake['text'].str.lower().str.contains('reuters', na=False).sum()
reuters_true = true['text'].str.lower().str.contains('reuters', na=False).sum()

print(f"\nReuters mentions in Fake.csv: {reuters_fake} / {len(fake)} ({reuters_fake/len(fake)*100:.1f}%)")
print(f"Reuters mentions in True.csv: {reuters_true} / {len(true)} ({reuters_true/len(true)*100:.1f}%)")

# Check top words in titles
print("\n=== TOP 10 WORDS IN FAKE NEWS TITLES ===")
all_fake_titles = ' '.join(fake['title'].fillna('').str.lower()).split()
fake_words = pd.Series(all_fake_titles).value_counts().head(10)
print(fake_words)

print("\n=== TOP 10 WORDS IN TRUE NEWS TITLES ===")
all_true_titles = ' '.join(true['title'].fillna('').str.lower()).split()
true_words = pd.Series(all_true_titles).value_counts().head(10)
print(true_words)
