"""
Model factory functions for experiments.

All notebooks MUST use these functions instead of instantiating models directly.
This ensures consistent configuration across all experiments.
"""

from typing import Any, Dict

from langchain_openai import ChatOpenAI, OpenAIEmbeddings


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
