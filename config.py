"""
Centralized LLM & Embedding Configuration
==========================================

All notebooks in this project import from this file.
Change the config here to apply across the entire project.

Usage in notebooks:
    import sys
    sys.path.append(str(Path(os.getcwd()).parent))
    from config import get_llm, get_embeddings, LLM_CONFIG, EMBEDDING_CONFIG, check_connections
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


# ============================================================
# CONNECTION CHECK
# ============================================================

def check_connections():
    """
    Test connectivity to both LLM and Embedding endpoints.
    Logs config details and verifies each service responds.
    
    Returns:
        dict with 'llm' and 'embedding' status (True/False)
    """
    import requests

    print("=" * 60)
    print("CONFIG CHECK")
    print("=" * 60)
    print(f"  LLM base_url : {LLM_CONFIG['base_url']}")
    print(f"  LLM model    : {LLM_CONFIG['model']}")
    print(f"  LLM api_key  : {'***' if LLM_CONFIG['api_key'] not in ('not-needed', '') else '(none)'}")
    print(f"  EMB base_url : {EMBEDDING_CONFIG['base_url']}")
    print(f"  EMB model    : {EMBEDDING_CONFIG['model']}")
    print(f"  EMB api_key  : {'***' if EMBEDDING_CONFIG['api_key'] not in ('not-needed', '') else '(none)'}")
    print("-" * 60)

    results = {"llm": False, "embedding": False}

    # Check LLM
    print("\n[1/2] Testing LLM connection...")
    try:
        llm = get_llm(max_tokens=20)
        response = llm.invoke("Say 'hello' in one word.")
        print(f"  ✅ LLM OK — response: {response.content.strip()[:80]}")
        results["llm"] = True
    except Exception as e:
        print(f"  ❌ LLM FAILED — {type(e).__name__}: {e}")

    # Check Embedding
    print("\n[2/2] Testing Embedding connection...")
    try:
        emb = get_embeddings()
        vector = emb.embed_query("test")
        print(f"  ✅ Embedding OK — vector dim: {len(vector)}")
        results["embedding"] = True
    except Exception as e:
        print(f"  ❌ Embedding FAILED — {type(e).__name__}: {e}")

    print("\n" + "=" * 60)
    if results["llm"] and results["embedding"]:
        print("ALL CONNECTIONS OK ✅")
    else:
        failed = [k for k, v in results.items() if not v]
        print(f"FAILED: {', '.join(failed)} ❌")
    print("=" * 60)

    return results
