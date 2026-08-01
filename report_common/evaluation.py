"""
Evaluation metrics for RAG experiments.

Includes retrieval metrics (hit_rate, precision, recall, MRR, nDCG)
and generation metrics (faithfulness, answer_relevancy placeholders).
"""

import math
from typing import List, Set


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
) -> dict:
    """Compute all retrieval metrics at once."""
    rel_set = set(relevant_ids)
    return {
        f"hit_rate_at_{k}": hit_rate_at_k(retrieved_ids, rel_set, k),
        f"precision_at_{k}": precision_at_k(retrieved_ids, rel_set, k),
        f"recall_at_{k}": recall_at_k(retrieved_ids, rel_set, k),
        "mrr": mrr(retrieved_ids, rel_set),
        f"ndcg_at_{k}": ndcg_at_k(retrieved_ids, rel_set, k),
    }
