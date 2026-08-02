"""Router and sufficiency parsing utilities for the Adaptive CRAG pipeline.

This module provides parsing functions for the query router and sufficiency
evaluator LLM responses, as well as context preparation for sufficiency checks.
"""

from typing import List

CATEGORIES = ["IDENTIFIER", "SEMANTIC", "MULTI_HOP", "CONFLICTING", "OUT_OF_SCOPE"]


def parse_route_response(response: str) -> str:
    """Parse LLM response to extract query category.

    Searches for known category keywords in the response text (case-insensitive).
    Returns the first matching category, or "SEMANTIC" as the default fallback.

    Args:
        response: Raw LLM response text from the router prompt.

    Returns:
        One of the CATEGORIES strings, or "SEMANTIC" if no category is found.
    """
    upper = response.strip().upper()
    for cat in CATEGORIES:
        if cat in upper:
            return cat
    return "SEMANTIC"


def parse_sufficiency_response(response: str) -> str:
    """Parse sufficiency evaluator LLM response.

    CRITICAL: Checks for "INSUFFICIENT" before "SUFFICIENT" to avoid the
    substring matching bug where "INSUFFICIENT" contains "SUFFICIENT".
    Defaults to "INSUFFICIENT" if neither keyword is found.

    Args:
        response: Raw LLM response text from the sufficiency evaluator.

    Returns:
        Either "INSUFFICIENT" or "SUFFICIENT".
    """
    upper = response.strip().upper()
    if "INSUFFICIENT" in upper:
        return "INSUFFICIENT"
    if "SUFFICIENT" in upper:
        return "SUFFICIENT"
    return "INSUFFICIENT"


def prepare_sufficiency_context(documents: List, max_chars: int = 1500) -> str:
    """Concatenate top-3 documents' full page_content, truncated to max_chars.

    Takes the first 3 documents (assumed to be ranked by relevance), joins
    their full page_content with double newlines, and truncates the combined
    text to the specified character limit.

    Args:
        documents: List of document objects with a `page_content` attribute.
            Only the first 3 are used.
        max_chars: Maximum character length of the returned context string.
            Defaults to 1500.

    Returns:
        Combined text from top-3 documents, truncated to max_chars.
    """
    texts = [doc.page_content for doc in documents[:3]]
    combined = "\n\n".join(texts)
    return combined[:max_chars]
