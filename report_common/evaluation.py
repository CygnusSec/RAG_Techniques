"""
Evaluation metrics for RAG experiments.

Includes retrieval metrics (hit_rate, precision, recall, MRR, nDCG)
and generation metrics (faithfulness, answer_relevancy placeholders).
"""

import bisect
import math
import re
from typing import List, Optional, Set, Tuple

# --- Source ID patterns ---
SOURCE_ID_PATTERN = re.compile(r'^(.+)::page_(\d+)::chunk_(\d+)$')
PAGE_ID_PATTERN = re.compile(r'^(.+)::page_(\d+)$')


def format_page_id(filename: str, page_number: int) -> str:
    """Format a page-level ID: {filename}::page_{N}"""
    return f"{filename}::page_{page_number}"


def format_chunk_id(filename: str, page_number: int, chunk_index: int) -> str:
    """Format a chunk-level ID: {filename}::page_{N}::chunk_{M}"""
    return f"{filename}::page_{page_number}::chunk_{chunk_index}"


def extract_page_id(chunk_id: str) -> Optional[str]:
    """Extract page-level prefix from chunk ID. Returns None if format unrecognized."""
    match = SOURCE_ID_PATTERN.match(chunk_id)
    if match:
        return f"{match.group(1)}::page_{match.group(2)}"
    # Check if already a page-level ID
    if PAGE_ID_PATTERN.match(chunk_id):
        return chunk_id
    return None


def hit_rate_at_k(retrieved_ids: List[str], relevant_ids: Set[str], k: int = 5) -> float:
    """1 if any relevant doc is in top-k, else 0."""
    top_k = retrieved_ids[:k]
    return 1.0 if any(doc_id in relevant_ids for doc_id in top_k) else 0.0


def precision_at_k(retrieved_ids: List[str], relevant_ids: Set[str], k: int = 5) -> float:
    """Fraction of top-k that are relevant."""
    top_k = retrieved_ids[:k]
    if not top_k:
        return 0.0
    relevant_in_top_k = sum(1 for doc_id in top_k if doc_id in relevant_ids)
    return relevant_in_top_k / len(top_k)


def recall_at_k(retrieved_ids: List[str], relevant_ids: Set[str], k: int = 5) -> float:
    """Fraction of relevant docs found in top-k."""
    if not relevant_ids:
        return 0.0
    top_k = retrieved_ids[:k]
    found = sum(1 for doc_id in top_k if doc_id in relevant_ids)
    return found / len(relevant_ids)


def mrr(retrieved_ids: List[str], relevant_ids: Set[str]) -> float:
    """Mean Reciprocal Rank: 1/rank of first relevant doc."""
    for i, doc_id in enumerate(retrieved_ids, start=1):
        if doc_id in relevant_ids:
            return 1.0 / i
    return 0.0


def ndcg_at_k(retrieved_ids: List[str], relevant_ids: Set[str], k: int = 5) -> float:
    """Normalized Discounted Cumulative Gain at k (binary relevance)."""
    top_k = retrieved_ids[:k]

    # DCG
    dcg = 0.0
    for i, doc_id in enumerate(top_k, start=1):
        if doc_id in relevant_ids:
            dcg += 1.0 / math.log2(i + 1)

    # Ideal DCG
    ideal_count = min(len(relevant_ids), k)
    idcg = sum(1.0 / math.log2(i + 1) for i in range(1, ideal_count + 1))

    if idcg == 0:
        return 0.0
    return dcg / idcg


def compute_retrieval_metrics(
    retrieved_ids: List[str],
    relevant_ids: List[str],
    k: int = 5,
    granularity: str = "auto",
    relevant_chunks: List[str] = None,
) -> dict:
    """
    Compute retrieval metrics with configurable granularity.

    Args:
        retrieved_ids: Source IDs of retrieved chunks.
        relevant_ids: Ground-truth page-level IDs (e.g. from relevant_documents).
        k: Cutoff for @k metrics.
        granularity: One of "chunk", "page", or "auto".
        relevant_chunks: Ground-truth chunk-level IDs (optional).

    Granularity behavior:
        - "chunk": exact match against relevant_chunks (fallback to page with warning
          if relevant_chunks is empty).
        - "page": extract page prefix from retrieved, match against relevant_ids.
        - "auto": use chunk if relevant_chunks non-empty; if relevant_chunks is None,
          fall back to direct matching against relevant_ids (legacy file-level);
          if relevant_chunks is an empty list, use page-level matching.

    Returns:
        dict with metric values and optional "granularity_fallback_warning" flag.

    Raises:
        ValueError: If granularity is not one of "chunk", "page", "auto".
    """
    valid_granularities = ("chunk", "page", "auto")
    if granularity not in valid_granularities:
        raise ValueError(
            f"Invalid granularity '{granularity}'. Must be one of: {', '.join(valid_granularities)}"
        )

    # Track whether a fallback warning should be emitted
    fallback_warning = False

    # Determine matching mode: "chunk_exact", "page_level", or "direct"
    if granularity == "chunk":
        if relevant_chunks:
            mode = "chunk_exact"
        else:
            # Chunk granularity requested but no chunk ground truth available;
            # fall back to page-level with warning (Requirement 2.5)
            mode = "page_level"
            fallback_warning = True
    elif granularity == "page":
        mode = "page_level"
    else:  # granularity == "auto"
        if relevant_chunks:
            # Non-empty relevant_chunks: use exact chunk matching
            mode = "chunk_exact"
        elif relevant_chunks is None:
            # relevant_chunks not provided at all: backward-compatible direct matching
            mode = "direct"
        else:
            # relevant_chunks is an empty list: fall back to page-level matching
            mode = "page_level"

    # Build effective retrieved IDs and relevant set based on mode
    if mode == "chunk_exact":
        rel_set = set(relevant_chunks)
        effective_retrieved = retrieved_ids
    elif mode == "page_level":
        # Page-level: extract page prefix from each retrieved_id,
        # match against relevant_ids (treated as page-level IDs).
        # IDs that don't parse are treated as non-matching (kept as-is, won't match).
        rel_set = set(relevant_ids)
        effective_retrieved = []
        for rid in retrieved_ids:
            page_id = extract_page_id(rid)
            # If extraction returns None, use original (won't match page-level IDs)
            effective_retrieved.append(page_id if page_id is not None else rid)
    else:
        # Direct mode: legacy behavior, match retrieved_ids against relevant_ids as-is
        rel_set = set(relevant_ids)
        effective_retrieved = retrieved_ids

    result = {
        f"hit_rate_at_{k}": hit_rate_at_k(effective_retrieved, rel_set, k),
        f"precision_at_{k}": precision_at_k(effective_retrieved, rel_set, k),
        f"recall_at_{k}": recall_at_k(effective_retrieved, rel_set, k),
        "mrr": mrr(effective_retrieved, rel_set),
        f"ndcg_at_{k}": ndcg_at_k(effective_retrieved, rel_set, k),
    }

    if fallback_warning:
        result["granularity_fallback_warning"] = True

    return result


def compute_page_offsets(pages: List, separator: str = "\n\n") -> Tuple[str, List[int]]:
    """
    Concatenate page texts with separator, return (full_text, offsets).
    offsets[i] = starting character index of page i in full_text.

    Args:
        pages: List of document objects with page_content attribute.
        separator: String inserted between pages. Defaults to "\n\n".

    Returns:
        (full_text, offsets) where offsets[i] is the start position of page i.
    """
    offsets = []
    current_offset = 0
    texts = []
    for i, page in enumerate(pages):
        offsets.append(current_offset)
        texts.append(page.page_content)
        current_offset += len(page.page_content)
        if i < len(pages) - 1:
            current_offset += len(separator)
    full_text = separator.join(texts)
    return full_text, offsets


def find_page_for_position(position: int, offsets: List[int], page_lengths: List[int]) -> int:
    """
    Given a character position in concatenated text, find which page it belongs to.
    Uses binary search on offsets.

    Args:
        position: Character position in the concatenated string.
        offsets: List of page start positions (from compute_page_offsets).
        page_lengths: List of page content lengths.

    Returns:
        Page index (0-based) that contains the given position.
    """
    # bisect_right gives us the insert point; -1 gives the page index
    page_idx = bisect.bisect_right(offsets, position) - 1
    return max(0, page_idx)


def attach_metadata_to_semantic_chunks(
    pages: List,
    semantic_chunks: List,
    filename: str,
    separator: str = "\n\n",
) -> List:
    """
    Assign page metadata to semantic chunks via character offset tracking.

    Args:
        pages: Original page documents with .page_content
        semantic_chunks: Chunks produced by SemanticChunker (may lack metadata)
        filename: PDF filename without path (e.g., "Understanding_Climate_Change.pdf")
        separator: Separator used when concatenating pages

    Returns:
        List of chunks with metadata set: source, page, chunk_index, source_id
        Empty/whitespace chunks are discarded.
    """
    full_text, offsets = compute_page_offsets(pages, separator)
    page_lengths = [len(p.page_content) for p in pages]

    result = []
    chunk_index = 0

    for chunk in semantic_chunks:
        # Discard empty/whitespace chunks
        if not chunk.page_content or not chunk.page_content.strip():
            continue

        # Find the starting position of this chunk in the concatenated text
        chunk_text = chunk.page_content
        pos = full_text.find(chunk_text)
        if pos == -1:
            # If exact match not found, try first 50 chars
            pos = full_text.find(chunk_text[:50])
        if pos == -1:
            pos = 0  # fallback to first page

        # Determine which page this position belongs to
        page_idx = find_page_for_position(pos, offsets, page_lengths)

        # Set metadata
        chunk.metadata = {
            "source": filename,
            "page": page_idx,
            "chunk_index": chunk_index,
        }
        chunk.metadata["source_id"] = format_chunk_id(filename, page_idx, chunk_index)

        result.append(chunk)
        chunk_index += 1

    return result


# --- Generation Metrics (LLM-as-judge) ---

import json


def _get_default_llm():
    """Construct LLM from experiment.yaml config when not provided."""
    from report_common.config import load_config
    from report_common.models import build_llm

    config = load_config()
    return build_llm(config)


def _decompose_claims(answer: str, context: str, llm) -> Tuple[int, int]:
    """
    Use LLM to decompose answer into atomic claims and check support against context.

    Returns:
        (supported_claims, total_claims)

    Raises:
        RuntimeError: If the LLM call fails or returns unparseable output.
    """
    prompt = (
        "You are an evaluation assistant. Your task is to:\n"
        "1. Decompose the ANSWER into individual atomic claims (simple factual statements).\n"
        "2. For each claim, determine if it is supported by the CONTEXT.\n\n"
        "CONTEXT:\n"
        f"{context}\n\n"
        "ANSWER:\n"
        f"{answer}\n\n"
        "Respond ONLY with a JSON object in this exact format:\n"
        '{"claims": [{"claim": "<claim text>", "supported": true/false}]}\n\n'
        "Rules:\n"
        "- A claim is supported if the context contains information that directly supports it.\n"
        "- If a claim cannot be verified from the context, mark it as not supported.\n"
        "- Decompose into the smallest possible atomic claims.\n"
    )
    try:
        response = llm.invoke(prompt)
        content = response.content if hasattr(response, "content") else str(response)
    except Exception as e:
        raise RuntimeError(f"LLM call failed during claim decomposition: {e}") from e

    try:
        # Try to extract JSON from the response
        content = content.strip()
        # Handle cases where response has markdown code block
        if content.startswith("```"):
            lines = content.split("\n")
            # Remove first and last lines (```json and ```)
            json_lines = []
            in_block = False
            for line in lines:
                if line.strip().startswith("```") and not in_block:
                    in_block = True
                    continue
                elif line.strip() == "```" and in_block:
                    break
                elif in_block:
                    json_lines.append(line)
            content = "\n".join(json_lines)

        parsed = json.loads(content)
        claims = parsed.get("claims", [])
        if not claims:
            # If no claims extracted, treat answer as single unsupported claim
            return (0, 1)
        total = len(claims)
        supported = sum(1 for c in claims if c.get("supported", False))
        return (supported, total)
    except (json.JSONDecodeError, KeyError, TypeError) as e:
        raise RuntimeError(
            f"Failed to parse LLM response for claim decomposition: {e}. "
            f"Response was: {content[:200]}"
        ) from e


def compute_faithfulness(answer: str, context: str, llm=None) -> float:
    """
    Decompose answer into atomic claims, check each against context.
    Returns: supported_claims / total_claims (0.0-1.0)

    Args:
        answer: The generated answer text.
        context: The retrieved context text.
        llm: Optional LLM instance. If None, constructs from experiment.yaml config.

    Returns:
        Float score between 0.0 and 1.0.

    Edge cases:
        - Empty answer → return 0.0 (no LLM call)
        - Empty context → return 0.0 (no LLM call)
        - LLM failure → raise RuntimeError

    Raises:
        RuntimeError: If the LLM call fails or returns unparseable response.
    """
    if not answer or not answer.strip():
        return 0.0
    if not context or not context.strip():
        return 0.0

    if llm is None:
        llm = _get_default_llm()

    supported, total = _decompose_claims(answer, context, llm)
    if total == 0:
        return 0.0
    return supported / total


def compute_relevancy(answer: str, question: str, llm=None) -> float:
    """
    Judge whether answer addresses the question.
    Returns: 0.0 (irrelevant) to 1.0 (fully addresses).

    Args:
        answer: The generated answer text.
        question: The user question.
        llm: Optional LLM instance. If None, constructs from experiment.yaml config.

    Returns:
        Float score between 0.0 and 1.0.

    Edge cases:
        - Empty answer → return 0.0 (no LLM call)
        - LLM failure → raise RuntimeError

    Raises:
        RuntimeError: If the LLM call fails or returns unparseable response.
    """
    if not answer or not answer.strip():
        return 0.0

    if llm is None:
        llm = _get_default_llm()

    prompt = (
        "You are an evaluation assistant. Judge how well the ANSWER addresses the QUESTION.\n\n"
        "QUESTION:\n"
        f"{question}\n\n"
        "ANSWER:\n"
        f"{answer}\n\n"
        "Respond ONLY with a JSON object in this exact format:\n"
        '{"score": <float between 0.0 and 1.0>, "reasoning": "<brief explanation>"}\n\n'
        "Scoring guide:\n"
        "- 1.0: The answer fully and directly addresses the question.\n"
        "- 0.7-0.9: The answer mostly addresses the question but may miss minor aspects.\n"
        "- 0.4-0.6: The answer partially addresses the question.\n"
        "- 0.1-0.3: The answer is only tangentially related to the question.\n"
        "- 0.0: The answer is completely irrelevant to the question.\n"
    )
    try:
        response = llm.invoke(prompt)
        content = response.content if hasattr(response, "content") else str(response)
    except Exception as e:
        raise RuntimeError(f"LLM call failed during relevancy scoring: {e}") from e

    try:
        content = content.strip()
        # Handle cases where response has markdown code block
        if content.startswith("```"):
            lines = content.split("\n")
            json_lines = []
            in_block = False
            for line in lines:
                if line.strip().startswith("```") and not in_block:
                    in_block = True
                    continue
                elif line.strip() == "```" and in_block:
                    break
                elif in_block:
                    json_lines.append(line)
            content = "\n".join(json_lines)

        parsed = json.loads(content)
        score = float(parsed.get("score", 0.0))
        # Clamp to valid range
        return max(0.0, min(1.0, score))
    except (json.JSONDecodeError, KeyError, TypeError, ValueError) as e:
        raise RuntimeError(
            f"Failed to parse LLM response for relevancy scoring: {e}. "
            f"Response was: {content[:200]}"
        ) from e


def compute_hallucination_rate(answer: str, context: str, llm=None) -> float:
    """
    Decompose answer into atomic claims, count unsupported.
    Returns: unsupported_claims / total_claims (0.0-1.0)

    Invariant: faithfulness + hallucination_rate ≈ 1.0 (tolerance 0.01)

    Args:
        answer: The generated answer text.
        context: The retrieved context text.
        llm: Optional LLM instance. If None, constructs from experiment.yaml config.

    Returns:
        Float score between 0.0 and 1.0.

    Edge cases:
        - Empty answer → return 0.0 (no LLM call)
        - Empty context → return 1.0 (no LLM call)
        - LLM failure → raise RuntimeError

    Raises:
        RuntimeError: If the LLM call fails or returns unparseable response.
    """
    if not answer or not answer.strip():
        return 0.0
    if not context or not context.strip():
        return 1.0

    if llm is None:
        llm = _get_default_llm()

    supported, total = _decompose_claims(answer, context, llm)
    if total == 0:
        return 0.0
    return (total - supported) / total


def compute_generation_metrics(
    answer: str, question: str, context: str, llm=None
) -> dict:
    """
    Convenience wrapper that computes all generation metrics.

    Args:
        answer: The generated answer text.
        question: The user question.
        context: The retrieved context text.
        llm: Optional LLM instance. If None, constructs from experiment.yaml config.

    Returns:
        dict with keys: faithfulness, relevancy, hallucination_rate

    Edge cases:
        - Empty answer → all metrics 0.0
        - Empty context → faithfulness=0.0, hallucination_rate=1.0
        - LLM failure → raise RuntimeError
    """
    if llm is None:
        llm = _get_default_llm()

    # Compute faithfulness and hallucination_rate together (same claim decomposition)
    # to ensure complementarity
    if not answer or not answer.strip():
        return {
            "faithfulness": 0.0,
            "relevancy": 0.0,
            "hallucination_rate": 0.0,
        }

    if not context or not context.strip():
        faithfulness = 0.0
        hallucination_rate = 1.0
    else:
        # Use single claim decomposition for both faithfulness and hallucination_rate
        supported, total = _decompose_claims(answer, context, llm)
        if total == 0:
            faithfulness = 0.0
            hallucination_rate = 0.0
        else:
            faithfulness = supported / total
            hallucination_rate = (total - supported) / total

    relevancy = compute_relevancy(answer, question, llm)

    return {
        "faithfulness": faithfulness,
        "relevancy": relevancy,
        "hallucination_rate": hallucination_rate,
    }
