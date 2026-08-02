"""Compression with fallback logic for RAG pipeline.

This module provides robust document compression with automatic fallback
to full text when compression produces empty, minimal, or irrelevant output.
"""

from typing import Callable, List, Tuple

MIN_COMPRESSED_LENGTH = 50


def compress_with_fallback(
    query: str,
    documents: List,
    compress_fn: Callable[[str, str], str],
    min_length: int = MIN_COMPRESSED_LENGTH,
) -> Tuple[str, bool]:
    """Compress documents with fallback to full context if compression fails.

    For each document, attempts compression via compress_fn. If the compressed
    output is empty, too short, or indicates no relevant content, falls back to
    the document's full text.

    Args:
        query: The user query to compress against.
        documents: List of document objects with a `page_content` attribute.
        compress_fn: Callable(query, doc_text) -> compressed_text.
        min_length: Minimum acceptable length for compressed output.

    Returns:
        Tuple of (context_text, fallback_triggered) where fallback_triggered
        is True if any per-document or full fallback was applied.

    Logic:
        1. For each doc, call compress_fn(query, doc.page_content)
        2. If result contains "NO_RELEVANT_CONTENT" -> exclude doc
        3. If result is empty/whitespace or < min_length chars -> use full doc text (per-doc fallback)
        4. If ALL docs excluded/failed -> use first doc's full text
        5. Final context must be >= min_length chars
    """
    if not documents:
        return ("", False)

    compressed_parts: List[str] = []
    fallback_triggered = False

    for doc in documents:
        doc_text = doc.page_content

        try:
            compressed = compress_fn(query, doc_text)
        except Exception:
            # Treat LLM call failure as compression failure -> use full text
            compressed_parts.append(doc_text)
            fallback_triggered = True
            continue

        # Check for NO_RELEVANT_CONTENT -> exclude this document
        if compressed and "NO_RELEVANT_CONTENT" in compressed:
            continue

        # Check for empty/whitespace or too short -> per-doc fallback
        if not compressed or not compressed.strip() or len(compressed.strip()) < min_length:
            compressed_parts.append(doc_text)
            fallback_triggered = True
        else:
            compressed_parts.append(compressed.strip())

    # If ALL docs were excluded (no parts collected), use first doc's full text
    if not compressed_parts:
        compressed_parts.append(documents[0].page_content)
        fallback_triggered = True

    context_text = "\n\n".join(compressed_parts)

    # Final safety check: ensure context meets minimum length
    if len(context_text.strip()) < min_length:
        context_text = documents[0].page_content
        fallback_triggered = True

    return (context_text, fallback_triggered)
