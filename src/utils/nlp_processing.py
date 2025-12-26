"""
Advanced NLP post-processing for OCR text cleanup.
Improves accuracy for Hindi, Marathi, and English text.
"""
from __future__ import annotations

import re
from typing import List, Tuple
from functools import lru_cache

import numpy as np
from scipy import sparse
from scipy.spatial.distance import cosine
from sklearn.feature_extraction.text import TfidfVectorizer

from src.core.logger import logger


# Common OCR error patterns for Devanagari
DEVANAGARI_CORRECTIONS = {
    # Common character confusions
    "।।": "।",  # Double danda to single
    "0": "०",   # Zero to Devanagari zero in numeric contexts
    "l": "।",   # Lowercase L often misread as danda
}

# Common English OCR errors
ENGLISH_CORRECTIONS = {
    "rn": "m",   # Common misread
    "cl": "d",   # Common misread
    "vv": "w",   # Common misread
}


def clean_ocr_text(text: str, language: str = "en") -> str:
    """
    Apply language-specific NLP cleaning to OCR output.
    
    Args:
        text: Raw OCR text
        language: Language code ('en', 'mr', 'hi')
    
    Returns:
        Cleaned text with improved accuracy
    """
    if not text or not text.strip():
        return ""
    
    # Step 1: Basic whitespace normalization
    text = normalize_whitespace(text)
    
    # Step 2: Remove control characters
    text = remove_control_chars(text)
    
    # Step 3: Language-specific corrections
    if language in ["mr", "hi"]:
        text = fix_devanagari_errors(text)
    else:
        text = fix_english_errors(text)
    
    # Step 4: Fix common punctuation issues
    text = fix_punctuation(text)
    
    # Step 5: Remove isolated single characters (likely noise)
    text = remove_noise_chars(text, language)
    
    return text.strip()


def normalize_whitespace(text: str) -> str:
    """Normalize various whitespace characters."""
    # Replace multiple spaces with single space
    text = re.sub(r' +', ' ', text)
    # Replace multiple newlines with double newline (paragraph break)
    text = re.sub(r'\n{3,}', '\n\n', text)
    # Remove spaces before punctuation
    text = re.sub(r' +([।,.!?:;])', r'\1', text)
    # Add space after punctuation if missing
    text = re.sub(r'([।,.!?:;])([^\s\n।,.!?:;0-9])', r'\1 \2', text)
    return text


def remove_control_chars(text: str) -> str:
    """Remove non-printable control characters except newlines and tabs."""
    # Keep newlines, tabs, and normal printable characters
    cleaned = []
    for char in text:
        if char in '\n\t' or (ord(char) >= 32 and ord(char) != 127):
            cleaned.append(char)
        elif ord(char) >= 0x0900:  # Keep all Devanagari and other Unicode
            cleaned.append(char)
    return ''.join(cleaned)


def fix_devanagari_errors(text: str) -> str:
    """Fix common OCR errors in Devanagari text."""
    for wrong, correct in DEVANAGARI_CORRECTIONS.items():
        text = text.replace(wrong, correct)
    
    # Fix common nukta issues
    # Sometimes nukta (़) gets separated from base character
    text = re.sub(r'(\u093C)\s+', r'\1', text)  # Remove space after nukta
    
    # Fix chandrabindu spacing
    text = re.sub(r'\s+(\u0901)', r'\1', text)  # Remove space before chandrabindu
    
    # Fix anusvara spacing
    text = re.sub(r'\s+(\u0902)', r'\1', text)  # Remove space before anusvara
    
    # Fix visarga spacing
    text = re.sub(r'\s+(\u0903)', r'\1', text)  # Remove space before visarga
    
    return text


def fix_english_errors(text: str) -> str:
    """Fix common OCR errors in English text."""
    # Apply common corrections
    for wrong, correct in ENGLISH_CORRECTIONS.items():
        # Only apply in word context to avoid false positives
        text = re.sub(rf'\b(\w*){wrong}(\w*)\b', rf'\1{correct}\2', text)
    
    # Fix common I/l/1 confusion in specific contexts
    text = re.sub(r'\bl\b', 'I', text)  # Standalone 'l' is usually 'I'
    
    # Fix O/0 confusion
    text = re.sub(r'\b0([a-zA-Z])', r'O\1', text)  # 0 before letters is O
    
    return text


def fix_punctuation(text: str) -> str:
    """Fix common punctuation issues."""
    # Fix multiple periods
    text = re.sub(r'\.{2,}', '...', text)
    # Fix multiple commas
    text = re.sub(r',{2,}', ',', text)
    # Fix space before period/comma
    text = re.sub(r'\s+([.,])', r'\1', text)
    return text


def remove_noise_chars(text: str, language: str) -> str:
    """Remove isolated single characters that are likely OCR noise."""
    lines = text.split('\n')
    cleaned_lines = []
    
    for line in lines:
        words = line.split()
        cleaned_words = []
        
        for word in words:
            # Keep the word if it's not a single character noise
            if len(word) > 1:
                cleaned_words.append(word)
            elif language in ["mr", "hi"]:
                # For Devanagari, keep valid single characters
                if is_valid_devanagari_single(word):
                    cleaned_words.append(word)
            else:
                # For English, keep valid single letters (I, a, etc.)
                if word.lower() in ['i', 'a', 'o'] or word.isdigit():
                    cleaned_words.append(word)
        
        cleaned_lines.append(' '.join(cleaned_words))
    
    return '\n'.join(cleaned_lines)


def is_valid_devanagari_single(char: str) -> bool:
    """Check if a single Devanagari character is valid standalone."""
    if len(char) != 1:
        return True
    
    # Valid single characters in Devanagari
    valid_singles = {
        '।', '॥',  # Dandas
        'अ', 'आ', 'इ', 'ई', 'उ', 'ऊ', 'ए', 'ऐ', 'ओ', 'औ',  # Vowels
        'क', 'ख', 'ग', 'घ',  # Some consonants used as abbreviations
        'व', 'य',  # Common single consonants
    }
    return char in valid_singles


def calculate_text_similarity(text1: str, text2: str) -> float:
    """
    Calculate similarity between two texts using TF-IDF and cosine similarity.
    Uses scipy for efficient sparse matrix operations.
    
    Args:
        text1: First text
        text2: Second text
    
    Returns:
        Similarity score between 0 and 1
    """
    if not text1 or not text2:
        return 0.0
    
    try:
        vectorizer = TfidfVectorizer()
        tfidf_matrix = vectorizer.fit_transform([text1, text2])
        
        # Use scipy sparse dot product for efficiency
        similarity = 1 - cosine(
            tfidf_matrix[0].toarray().flatten(),
            tfidf_matrix[1].toarray().flatten()
        )
        return float(similarity) if not np.isnan(similarity) else 0.0
    except Exception as e:
        logger.warning(f"Similarity calculation failed: {e}")
        return 0.0


def merge_ocr_results(results: List[Tuple[str, float, str]]) -> Tuple[str, float, str]:
    """
    Merge multiple OCR results by selecting the best one based on confidence
    and text quality metrics.
    
    Args:
        results: List of (text, confidence, language) tuples
    
    Returns:
        Best (text, confidence, language) tuple
    """
    if not results:
        return ("", 0.0, "unknown")
    
    if len(results) == 1:
        return results[0]
    
    scored_results = []
    for text, confidence, language in results:
        # Calculate quality score
        quality = calculate_text_quality(text, language)
        combined_score = 0.6 * confidence + 0.4 * quality
        scored_results.append((text, combined_score, language, confidence))
    
    # Sort by combined score
    scored_results.sort(key=lambda x: x[1], reverse=True)
    best = scored_results[0]
    
    return (best[0], best[3], best[2])  # Return original confidence


def calculate_text_quality(text: str, language: str) -> float:
    """
    Calculate text quality score based on various heuristics.
    
    Returns a score between 0 and 1.
    """
    if not text:
        return 0.0
    
    score = 1.0
    
    # Penalize very short text
    if len(text) < 10:
        score *= 0.5
    
    # Penalize excessive special characters
    special_ratio = sum(1 for c in text if not c.isalnum() and not c.isspace()) / max(len(text), 1)
    if special_ratio > 0.3:
        score *= (1 - special_ratio)
    
    # Penalize excessive single characters
    words = text.split()
    if words:
        single_char_ratio = sum(1 for w in words if len(w) == 1) / len(words)
        if single_char_ratio > 0.5:
            score *= (1 - single_char_ratio * 0.5)
    
    # Bonus for proper sentence structure
    if re.search(r'[.।]$', text.strip()):
        score *= 1.1
    
    return min(score, 1.0)


@lru_cache(maxsize=100)
def get_stopwords(language: str) -> set:
    """Get stopwords for a language (cached)."""
    english_stopwords = {
        'the', 'a', 'an', 'is', 'are', 'was', 'were', 'be', 'been', 'being',
        'have', 'has', 'had', 'do', 'does', 'did', 'will', 'would', 'could',
        'should', 'may', 'might', 'must', 'shall', 'can', 'of', 'at', 'by',
        'for', 'with', 'about', 'against', 'between', 'into', 'through',
        'during', 'before', 'after', 'above', 'below', 'to', 'from', 'up',
        'down', 'in', 'out', 'on', 'off', 'over', 'under', 'again', 'further',
        'then', 'once', 'here', 'there', 'when', 'where', 'why', 'how', 'all',
        'each', 'few', 'more', 'most', 'other', 'some', 'such', 'no', 'nor',
        'not', 'only', 'own', 'same', 'so', 'than', 'too', 'very', 'just',
    }
    
    hindi_marathi_stopwords = {
        'का', 'के', 'की', 'को', 'से', 'में', 'पर', 'है', 'हैं', 'था', 'थे',
        'थी', 'और', 'या', 'एक', 'यह', 'वह', 'जो', 'कि', 'इस', 'उस', 'को',
        'ने', 'भी', 'तो', 'हो', 'जा', 'आ', 'कर', 'ला', 'ले', 'दे', 'रहा',
        'रही', 'रहे', 'गया', 'गयी', 'गये', 'होता', 'होती', 'होते',
        # Marathi specific
        'आहे', 'आहेत', 'होता', 'होती', 'होते', 'आणि', 'किंवा', 'हा', 'ही',
        'हे', 'त्या', 'या', 'ते', 'ती', 'तो', 'च', 'पण', 'तर', 'म्हणून',
    }
    
    if language in ['mr', 'hi']:
        return hindi_marathi_stopwords
    return english_stopwords


__all__ = [
    "clean_ocr_text",
    "normalize_whitespace",
    "fix_devanagari_errors",
    "fix_english_errors",
    "calculate_text_similarity",
    "merge_ocr_results",
    "calculate_text_quality",
]
