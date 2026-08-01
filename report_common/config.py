"""
Experiment configuration loader.

Reads .env and config/experiment.yaml, validates required variables,
fixes random seed, computes config_hash, and saves config snapshot with results.
"""

import hashlib
import json
import os
import random
from pathlib import Path
from typing import Any, Dict

import numpy as np
import yaml
from dotenv import load_dotenv


PROJECT_ROOT = Path(__file__).resolve().parent.parent
CONFIG_PATH = PROJECT_ROOT / "config" / "experiment.yaml"


def _resolve_env_vars(obj: Any) -> Any:
    """Recursively resolve ${VAR} placeholders in config values."""
    if isinstance(obj, str) and obj.startswith("${") and obj.endswith("}"):
        var_name = obj[2:-1]
        value = os.environ.get(var_name)
        if value is None:
            raise EnvironmentError(f"Required env var '{var_name}' is not set.")
        return value
    elif isinstance(obj, dict):
        return {k: _resolve_env_vars(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [_resolve_env_vars(v) for v in obj]
    return obj


def load_config(env_path: Path = None) -> Dict[str, Any]:
    """
    Load experiment configuration.

    1. Load .env file
    2. Read YAML config
    3. Resolve environment variables
    4. Validate required fields
    5. Fix random seed
    6. Compute config_hash

    Returns:
        dict with all experiment parameters
    """
    # Load .env
    env_file = env_path or PROJECT_ROOT / ".env"
    if env_file.exists():
        load_dotenv(env_file)
    else:
        load_dotenv()  # Try default locations

    # Read YAML
    if not CONFIG_PATH.exists():
        raise FileNotFoundError(f"Config file not found: {CONFIG_PATH}")

    with open(CONFIG_PATH, "r", encoding="utf-8") as f:
        raw_config = yaml.safe_load(f)

    # Resolve env vars
    config = _resolve_env_vars(raw_config)

    # Validate required
    required_keys = [
        ("llm", "base_url"),
        ("llm", "model"),
        ("embedding", "base_url"),
        ("embedding", "model"),
    ]
    for section, key in required_keys:
        val = config.get(section, {}).get(key)
        if not val or val.startswith("${"):
            raise ValueError(
                f"Config '{section}.{key}' is missing or unresolved. "
                f"Check .env and config/experiment.yaml."
            )

    # Fix random seed
    seed = config.get("seed", 42)
    random.seed(seed)
    np.random.seed(seed)

    # Compute config hash for reproducibility tracking
    config_str = json.dumps(config, sort_keys=True, ensure_ascii=False)
    config["_config_hash"] = hashlib.sha256(config_str.encode()).hexdigest()[:12]
    config["_project_root"] = str(PROJECT_ROOT)

    return config


def save_config_snapshot(config: Dict[str, Any], output_dir: Path) -> Path:
    """Save a config snapshot alongside results for reproducibility."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    snapshot_path = output_dir / "config_snapshot.json"
    # Remove internal keys
    saveable = {k: v for k, v in config.items() if not k.startswith("_")}
    saveable["_config_hash"] = config.get("_config_hash", "unknown")
    with open(snapshot_path, "w", encoding="utf-8") as f:
        json.dump(saveable, f, indent=2, ensure_ascii=False)
    return snapshot_path


def print_config_summary(config: Dict[str, Any]) -> None:
    """Print a concise config summary for notebook display."""
    print("=" * 60)
    print("EXPERIMENT CONFIGURATION")
    print("=" * 60)
    print(f"  Config hash  : {config.get('_config_hash', 'N/A')}")
    print(f"  Seed         : {config.get('seed')}")
    print(f"  LLM          : {config['llm']['model']} @ {config['llm']['base_url']}")
    print(f"  Embedding    : {config['embedding']['model']} @ {config['embedding']['base_url']}")
    print(f"  Chunk size   : {config['baseline']['chunk_size']} {config['baseline']['chunk_unit']}")
    print(f"  Chunk overlap: {config['baseline']['chunk_overlap']}")
    print(f"  Top-K        : {config['baseline']['top_k']}")
    print(f"  Results dir  : {config['paths']['results']}")
    print("=" * 60)
