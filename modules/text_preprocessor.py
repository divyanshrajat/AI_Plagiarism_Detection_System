"""
Enhanced text preprocessor for NLP operations.

This module provides text preprocessing including:
- Tokenization
- Stop word removal
- Lemmatization
- Text statistics for analysis
"""

import re
from typing import List, Dict

# Try to import NLTK components
try:
    import nltk
    from nltk.corpus import stopwords
    from nltk.stem import WordNetLemmatizer
    
    # Download required NLTK data
    try:
        nltk.data.find('tokenizers/punkt')
    except LookupError:
        nltk.download('punkt', quiet=True)
    
    try:
        nltk.data.find('corpora/stopwords')
    except LookupError:
        nltk.download('stopwords', quiet=True)
    
    try:
        nltk.data.find('corpora/wordnet')
    except LookupError:
        nltk.download('wordnet', quiet=True)
    
    _STOPWORDS = set(stopwords.words('english'))
    _LEMMATIZER = WordNetLemmatizer()
    _NLTK_AVAILABLE = True
    
except (ImportError, ModuleNotFoundError):
    _STOPWORDS = set()
    _LEMMATIZER = None
    _NLTK_AVAILABLE = False


def clean_text(text: str) -> str:
    """Basic text cleaning - lowercase and remove special characters."""
    text = text.lower()
    text = re.sub(r"[^a-z0-9\s\.,;:!?'-]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def tokenize(text: str) -> List[str]:
    """Tokenize input into words."""
    return re.findall(r"\b\w+\b", text.lower())


def remove_stop_words(tokens: List[str]) -> List[str]:
    """Remove common English stop words."""
    if _NLTK_AVAILABLE and _STOPWORDS:
        return [t for t in tokens if t not in _STOPWORDS]
    return tokens


def lemmatize(tokens: List[str]) -> List[str]:
    """Reduce words to their base form using lemmatization."""
    if not _NLTK_AVAILABLE or _LEMMATIZER is None:
        return tokens
    
    result = []
    for token in tokens:
        lemma = _LEMMATIZER.lemmatize(token)
        result.append(lemma)
    return result


def preprocess_for_similarity(text: str) -> str:
    """
    Full preprocessing pipeline optimized for similarity detection.
    Returns cleaned, normalized text ready for embedding.
    """
    cleaned = clean_text(text)
    tokens = tokenize(cleaned)
    filtered_tokens = [t for t in tokens if len(t) > 1]
    return ' '.join(filtered_tokens).strip()


def get_word_count_stats(text: str) -> Dict[str, float]:
    """
    Get statistics about word usage in document.
    Useful for detecting AI-generated content patterns.
    """
    cleaned = text.lower()
    all_words = re.findall(r"\b\w+\b", cleaned)
    
    if not all_words:
        return {
            'total_words': 0,
            'unique_words': 0,
            'unique_ratio': 0.0,
            'avg_word_length': 0.0
        }
    
    unique_words = set(all_words)
    
    stats = {
        'total_words': len(all_words),
        'unique_words': len(unique_words),
        'unique_ratio': round(len(unique_words) / len(all_words), 4),
        'avg_word_length': round(sum(len(w) for w in all_words) / len(all_words), 2)
    }
    return stats


def chunk_text(text: str, chunk_size: int = 100, overlap: int = 50) -> List[str]:
    """
    Split text into overlapping chunks of a specific number of words.
    Smaller chunks (100 words) ensure localized matches aren't diluted.
    """
    words = text.split()
    if not words:
        return []
    
    chunks = []
    for i in range(0, len(words), chunk_size - overlap):
        chunk = " ".join(words[i:i + chunk_size])
        if chunk:
            chunks.append(chunk)
        if i + chunk_size >= len(words):
            break
            
    return chunks


# Expose main functions at module level
__all__ = [
    'clean_text',
    'tokenize',
    'remove_stop_words',
    'lemmatize',
    'preprocess_for_similarity',
    'get_word_count_stats'
]
