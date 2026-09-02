"""
Text cleaning utilities.
Pure functions — no I/O, no external calls.
"""

import re
import unicodedata


def normalize(text: str) -> str:
    """
    Full normalization pipeline:
    1. Unicode normalization (NFKC)
    2. Lowercase
    3. Collapse whitespace
    4. Strip leading/trailing
    """
    text = unicodedata.normalize("NFKC", text)
    text = text.lower()
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def clean_abstract(text: str) -> str:
    """
    Remove common PubMed boilerplate and HTML artifacts from abstracts.
    """
    # Strip HTML-like tags (rare but present in some records)
    text = re.sub(r"<[^>]+>", " ", text)
    # Remove copyright/license lines
    text = re.sub(
        r"(copyright|©|all rights reserved|published by elsevier)[^\n]*",
        "",
        text,
        flags=re.IGNORECASE,
    )
    # Collapse whitespace again after removals
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def truncate(text: str, max_chars: int = 1500) -> str:
    """Truncate to max_chars, preserving word boundaries."""
    if len(text) <= max_chars:
        return text
    truncated = text[:max_chars]
    last_space = truncated.rfind(" ")
    return truncated[:last_space] if last_space > 0 else truncated
