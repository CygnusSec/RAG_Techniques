"""
Centralized LLM & Embedding Configuration
==========================================

All notebooks in this project import from this file.
Change the config here to apply across the entire project.

Usage in notebooks:
    import sys
    sys.path.append(str(Path(os.getcwd()).parent))
    from config import get_llm, get_embeddings, LLM_CONFIG, EMBEDDING_CONFIG
"""

from langchain_openai import ChatOpenAI, OpenAIEmbeddings


# ============================================================
# LLM CONFIGURATION
# ============================================================
LLM_CONFIG = {
    "base_url": "http://chatbot.cygnussec.tech:23113/v1",
    "model": "cyankiwi/gemma-4-31B-it-AWQ-8bit",
    "api_key": "not-needed",  # Set API key if server requires one
}

# ============================================================
# EMBEDDING CONFIGURATION
# ============================================================
EMBEDDING_CONFIG = {
    "base_url": "http://chatbot.cygnussec.tech:23012/v1",
    "model": "bge-m3",
    "api_key": "not-needed",  # Set API key if server requires one
}


# ============================================================
# FACTORY FUNCTIONS
# ============================================================

def get_llm(temperature=0, max_tokens=4000, **kwargs):
    """
    Create a ChatOpenAI-compatible LLM instance.
    
    Supports any OpenAI-compatible endpoint (vLLM, Ollama, LM Studio, etc.)
    
    Args:
        temperature: Sampling temperature (0 = deterministic)
        max_tokens: Maximum tokens in response
        **kwargs: Override any LLM_CONFIG parameter
        
    Returns:
        ChatOpenAI instance
    """
    config = {**LLM_CONFIG, **kwargs}
    return ChatOpenAI(
        base_url=config["base_url"],
        api_key=config["api_key"],
        model=config["model"],
        temperature=temperature,
        max_tokens=max_tokens,
    )


def get_embeddings(**kwargs):
    """
    Create an OpenAIEmbeddings-compatible instance.
    
    Supports any OpenAI-compatible embedding endpoint.
    
    Args:
        **kwargs: Override any EMBEDDING_CONFIG parameter
        
    Returns:
        OpenAIEmbeddings instance
    """
    config = {**EMBEDDING_CONFIG, **kwargs}
    return OpenAIEmbeddings(
        base_url=config["base_url"],
        api_key=config["api_key"],
        model=config["model"],
    )
