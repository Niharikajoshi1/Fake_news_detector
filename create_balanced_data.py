import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from preprocessing import clean_text

def analyze_and_fix_data_bias():
    """
    Analyzes and fixes source/topic bias in the training data.
    Removes artificial signals (Reuters, specific publishers) so model
    learns from actual content quality, not source.
    """
    
    print("=" * 70)
    print(" ANALYZING DATA BIAS & CREATING BALANCED DATASET")
    print("=" * 70)
    
    # Load data
    fake = pd.read_csv('Fake.csv')
    true = pd.read_csv('True.csv')
    
    fake['class'] = 0
    true['class'] = 1
    
    print(f"\nOriginal dataset sizes:")
    print(f"  Fake: {len(fake)}, True: {len(true)}")
    
    # --- FIX 1: Remove Reuters dependency ---
    print("\n[1/3] Removing publisher source bias...")
    
    # For fake news: convert "WASHINGTON (Reuters) -" to just content
    fake_before = len(fake)
    fake['title'] = fake['title'].str.replace(r'^\(reuters\)', '', regex=True, case=False)
    fake['text'] = fake['text'].str.replace(r'^[a-z\s]+\(reuters\)', '', regex=True, case=False)
    
    # For true news: same treatment
    true_before = len(true)
    true['title'] = true['title'].str.replace(r'^\(reuters\)', '', regex=True, case=False)
    true['text'] = true['text'].str.replace(r'^[a-z\s]+\(reuters\)', '', regex=True, case=False)
    
    print(f"  ✓ Removed broadcaster location prefixes")
    print(f"  - Fake: {len(fake)} (unchanged)")
    print(f"  - True: {len(true)} (unchanged)")
    
    # --- FIX 2: Balance the dataset ---
    print("\n[2/3] Balancing dataset...")
    
    # Use stratified sampling to create balanced subsets
    min_size = min(len(fake), len(true))
    print(f"  Target size: {min_size} per class")
    
    fake_balanced = fake.sample(n=min_size, random_state=42)
    true_balanced = true.sample(n=min_size, random_state=42)
    
    balanced_df = pd.concat([fake_balanced, true_balanced], ignore_index=True)
    balanced_df = balanced_df.sample(frac=1, random_state=42).reset_index(drop=True)
    
    print(f"  ✓ Created balanced dataset: {len(balanced_df)} total samples")
    print(f"    - Fake: {len(fake_balanced)} (class 0)")
    print(f"    - True: {len(true_balanced)} (class 1)")
    
    # --- FIX 3: Data quality check ---
    print("\n[3/3] Analyzing cleaned text quality...")
    
    balanced_df['full_content'] = balanced_df['title'] + " " + balanced_df['text']
    balanced_df['cleaned'] = balanced_df['full_content'].apply(clean_text)
    balanced_df = balanced_df[balanced_df['cleaned'].str.len() > 50].reset_index(drop=True)
    
    print(f"  ✓ Removed very short articles: {len(balanced_df)} remaining")
    
    # Analyze top words per class
    fake_data = balanced_df[balanced_df['class'] == 0]
    true_data = balanced_df[balanced_df['class'] == 1]
    
    print(f"\n  Fake news patterns:")
    fake_text = ' '.join(fake_data['cleaned'].str.lower()).split()
    fake_freq = pd.Series(fake_text).value_counts().head(5)
    for word, count in fake_freq.items():
        print(f"    - '{word}': {count}")
    
    print(f"\n  True news patterns:")
    true_text = ' '.join(true_data['cleaned'].str.lower()).split()
    true_freq = pd.Series(true_text).value_counts().head(5)
    for word, count in true_freq.items():
        print(f"    - '{word}': {count}")
    
    # Save balanced dataset
    output_file = 'balanced_data.csv'
    balanced_df.to_csv(output_file, index=False)
    print(f"\n✓ Saved balanced dataset to '{output_file}'")
    
    return balanced_df

if __name__ == "__main__":
    analyze_and_fix_data_bias()
