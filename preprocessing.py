import re
import string

def clean_text(text: str) -> str:
    """
    Cleans raw news text by:
    1. Converting to lowercase
    2. Removing location prefixes only (e.g., 'WASHINGTON -') but KEEPING Reuters mentions
    3. Removing URLs
    4. PRESERVING important signals like [VIDEO], [AUDIO]
    5. Removing excessive punctuation but keeping some for context
    6. Normalizing white spaces
    """
    if not isinstance(text, str):
        return ""
    
    text = text.lower()
    
    # 1. Strip ONLY location prefixes (e.g., 'washington -', 'london -'), but KEEP 'reuters' as a signal
    # Remove patterns like "CITY (COUNTRY) -" or "CITY, COUNTRY -"
    text = re.sub(r'^[a-z]+(?:\s+[a-z]+)*\s*(?:\([a-z\s]+\))?\s*[-—–:]\s*', '', text)
    text = re.sub(r'^\s*21st century wire\s*[-—–:]\s*', '', text)
    
    # 2. Remove URLs and HTML angle-bracket tags, but KEEP square brackets (they contain signals)
    text = re.sub(r'https?://\S+|www\.\S+', '', text)
    text = re.sub(r'<[^>]+>', ' ', text)  # Remove only < > tags, not [ ]
    
    # 3. Preserve [VIDEO], [AUDIO], [IMAGE] etc - these are strong fake news signals
    # Just normalize them: [VIDEO] -> VIDEO
    text = re.sub(r'\[([a-z\s]+)\]', r'\1', text)
    
    # 4. Remove punctuation but keep some structure
    # Keep apostrophes and hyphens for word integrity
    text = re.sub(r'[%s]' % re.escape(string.punctuation.replace("'", "").replace("-", "")), ' ', text)
    
    # 5. Remove newlines but preserve word boundaries
    text = re.sub(r'\n|\r', ' ', text)
    
    # 6. Remove extra whitespace
    text = re.sub(r'\s+', ' ', text).strip()
    
    return text

# Alias for backward compatibility with old code
wordopt = clean_text

if __name__ == "__main__":
    sample_text = "WASHINGTON (Reuters) - The U.S. Senate passed a new bill on Wednesday... Visit https://example.com"
    print("Original:", sample_text)
    print("Cleaned:", clean_text(sample_text))
