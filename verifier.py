"""
verifier.py
Deterministic Python verification engine for interview quotes.
Performs substring, normalized whitespace, and sliding-window fuzzy verification.
"""

import re
import unicodedata
import difflib
from typing import Dict, Any


def normalize_text(text: str) -> str:
    """Normalizes whitespace, Unicode quotation marks, and hyphens."""
    if not text:
        return ""
    text = unicodedata.normalize("NFKD", text)
    text = text.replace('“', '"').replace('”', '"').replace('’', "'").replace('‘', "'")
    text = text.replace('—', '-').replace('–', '-')
    text = re.sub(r'\s+', ' ', text)
    return text.strip()


def verify_quote(quote: str, raw_source: str) -> Dict[str, Any]:
    """
    Deterministically audits whether `quote` exists in `raw_source`.
    Returns verification boolean, match type, and similarity metrics.
    """
    cleaned_quote = quote.strip().strip('"').strip("'")
    if not cleaned_quote:
        return {
            "verified": False,
            "match_type": "EMPTY",
            "similarity_score": 0.0,
            "details": "Candidate string is empty"
        }

    # 1. Exact raw character match
    if cleaned_quote in raw_source:
        return {
            "verified": True,
            "match_type": "EXACT",
            "similarity_score": 1.0,
            "details": "Verbatim substring match in raw source"
        }

    # 2. Normalized match (handles line wraps, smart quotes, extra spaces)
    norm_quote = normalize_text(cleaned_quote)
    norm_source = normalize_text(raw_source)

    if norm_quote.lower() in norm_source.lower():
        return {
            "verified": True,
            "match_type": "NORMALIZED_EXACT",
            "similarity_score": 1.0,
            "details": "Exact match verified after whitespace/punctuation normalization"
        }

    # 3. Sliding-window fuzzy match for stitched multi-line statements
    quote_len = len(norm_quote)
    source_len = len(norm_source)
    best_ratio = 0.0
    step = max(1, quote_len // 6)

    for i in range(0, max(1, source_len - quote_len + 1), step):
        window = norm_source[i : i + quote_len + 15]
        ratio = difflib.SequenceMatcher(None, norm_quote.lower(), window.lower()).ratio()
        if ratio > best_ratio:
            best_ratio = ratio

    is_verified = best_ratio >= 0.88

    return {
        "verified": is_verified,
        "match_type": "SLIDING_WINDOW_MATCH" if is_verified else "FAILED",
        "similarity_score": round(best_ratio, 3),
        "details": f"Similarity ratio: {round(best_ratio * 100, 1)}%"
    }
