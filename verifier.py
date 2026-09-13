"""
verifier.py
Deterministic Python verification engine for interview quotes.
"""

import re
import unicodedata
from typing import Dict, Any
import Levenshtein


def normalize_text(text: str) -> str:
    if not text:
        return ""
    text = unicodedata.normalize("NFKD", text)
    text = text.replace('“', '"').replace('”', '"').replace('’', "'").replace('‘', "'")
    text = text.replace('—', '-').replace('–', '-')
    text = re.sub(r'\s+', ' ', text)
    return text.strip()


def verify_quote(quote: str, raw_source: str) -> Dict[str, Any]:
    cleaned_quote = quote.strip().strip('"').strip("'")
    if not cleaned_quote:
        return {
            "verified": False,
            "match_type": "EMPTY",
            "similarity_score": 0.0,
            "details": "Quote string was empty"
        }

    # 1. Exact raw match
    if cleaned_quote in raw_source:
        return {
            "verified": True,
            "match_type": "EXACT",
            "similarity_score": 1.0,
            "details": "Verbatim substring match in raw document"
        }

    # 2. Normalized match (ignores smart quotes, trailing line breaks, extra spaces)
    norm_quote = normalize_text(cleaned_quote)
    norm_source = normalize_text(raw_source)

    if norm_quote.lower() in norm_source.lower():
        return {
            "verified": True,
            "match_type": "NORMALIZED_EXACT",
            "similarity_score": 1.0,
            "details": "Match confirmed after normalizing whitespace and punctuation"
        }

    # 3. Best window sliding fuzzy match
    quote_len = len(norm_quote)
    best_ratio = 0.0
    source_len = len(norm_source)
    step = max(1, quote_len // 5)

    for i in range(0, max(1, source_len - quote_len + 1), step):
        window = norm_source[i : i + quote_len + 10]
        ratio = Levenshtein.ratio(norm_quote.lower(), window.lower())
        if ratio > best_ratio:
            best_ratio = ratio

    is_fuzzy_verified = best_ratio >= 0.92

    return {
        "verified": is_fuzzy_verified,
        "match_type": "FUZZY_MATCH" if is_fuzzy_verified else "FAILED",
        "similarity_score": round(best_ratio, 3),
        "details": f"Closest match similarity: {round(best_ratio * 100, 1)}%"
    }
