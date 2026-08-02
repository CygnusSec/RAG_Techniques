"""
Model factory functions for experiments.

All notebooks MUST use these functions instead of instantiating models directly.
This ensures consistent configuration across all experiments.
"""

from typing import Any, Dict, List, Tuple

from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from sentence_transformers import CrossEncoder


def build_llm(config: Dict[str, Any], **overrides) -> ChatOpenAI:
    """
    Build LLM instance from experiment config.

    Args:
        config: Loaded experiment config dict.
        **overrides: Override any LLM parameter (temperature, max_tokens, etc.)

    Returns:
        ChatOpenAI instance pointing to configured endpoint.
    """
    llm_cfg = config["llm"]
    params = {
        "base_url": llm_cfg["base_url"],
        "api_key": llm_cfg.get("api_key", "not-needed"),
        "model": llm_cfg["model"],
        "temperature": llm_cfg.get("temperature", 0),
        "max_tokens": llm_cfg.get("max_tokens", 1024),
    }
    params.update(overrides)
    return ChatOpenAI(**params)


def build_embeddings(config: Dict[str, Any], **overrides) -> OpenAIEmbeddings:
    """
    Build Embeddings instance from experiment config.

    Args:
        config: Loaded experiment config dict.
        **overrides: Override any embedding parameter.

    Returns:
        OpenAIEmbeddings instance pointing to configured endpoint.
    """
    emb_cfg = config["embedding"]
    params = {
        "base_url": emb_cfg["base_url"],
        "api_key": emb_cfg.get("api_key", "not-needed"),
        "model": emb_cfg["model"],
    }
    params.update(overrides)
    return OpenAIEmbeddings(**params)


def build_cross_encoder(config: Dict[str, Any]) -> CrossEncoder:
    """
    Load cross-encoder model from config['reranker']['model_name'].

    Args:
        config: Loaded experiment config dict. Expected to contain a 'reranker'
                section with 'model_name' key.

    Returns:
        CrossEncoder instance ready for scoring.

    Raises:
        RuntimeError: If the model cannot be loaded or downloaded.
    """
    model_name = config.get("reranker", {}).get(
        "model_name", "cross-encoder/ms-marco-MiniLM-L-6-v2"
    )
    try:
        return CrossEncoder(model_name)
    except Exception as e:
        raise RuntimeError(
            f"Failed to load cross-encoder model '{model_name}': {e}"
        ) from e


def cross_encoder_rerank(
    query: str,
    documents: List,
    cross_encoder: CrossEncoder,
    top_k: int = 5,
) -> List[Tuple[Any, float]]:
    """
    Score all (query, doc) pairs in a single batch predict call.

    Args:
        query: The search query string.
        documents: List of document objects with a `page_content` attribute.
        cross_encoder: A loaded CrossEncoder instance.
        top_k: Number of top-scoring documents to return.

    Returns:
        List of (document, score) tuples sorted by descending score,
        truncated to top_k entries.
    """
    if not documents:
        return []
    pairs = [(query, doc.page_content[:512]) for doc in documents]
    scores = cross_encoder.predict(pairs)
    scored = list(zip(documents, [float(s) for s in scores]))
    scored.sort(key=lambda x: x[1], reverse=True)
    return scored[:top_k]
