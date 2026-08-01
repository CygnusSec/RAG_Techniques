"""
I/O utilities for saving experiment results.

Handles JSONL per-question output, CSV summaries, and prevents accidental overwrites.
"""

import csv
import json
import time
from pathlib import Path
from typing import Any, Dict, List


def save_jsonl(records: List[Dict[str, Any]], output_path: Path) -> Path:
    """Save list of result records as JSONL (one JSON object per line)."""
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        for record in records:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")
    print(f"Saved {len(records)} records to {output_path}")
    return output_path


def load_jsonl(path: Path) -> List[Dict[str, Any]]:
    """Load JSONL file into list of dicts."""
    records = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                records.append(json.loads(line))
    return records


def save_csv_summary(records: List[Dict[str, Any]], output_path: Path, fields: List[str] = None) -> Path:
    """Save summary CSV from records."""
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    if not records:
        print("No records to save.")
        return output_path

    if fields is None:
        fields = list(records[0].keys())

    with open(output_path, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        for record in records:
            writer.writerow(record)

    print(f"Saved CSV summary to {output_path}")
    return output_path


def build_result_record(
    experiment_id: str,
    notebook: str,
    config_hash: str,
    seed: int,
    question_id: str,
    question: str,
    retrieved_documents: List[Dict] = None,
    answer: str = "",
    citations: List[str] = None,
    predicted_abstain: bool = False,
    latency: Dict[str, float] = None,
    usage: Dict[str, int] = None,
    metrics: Dict[str, Any] = None,
    error: str = None,
    **extra
) -> Dict[str, Any]:
    """Build a standardized result record matching the report schema."""
    return {
        "experiment_id": experiment_id,
        "notebook": notebook,
        "config_hash": config_hash,
        "seed": seed,
        "question_id": question_id,
        "question": question,
        "retrieved_documents": retrieved_documents or [],
        "answer": answer,
        "citations": citations or [],
        "predicted_abstain": predicted_abstain,
        "latency": latency or {},
        "usage": usage or {},
        "metrics": metrics or {},
        "error": error,
        **extra,
    }


class Timer:
    """Simple context manager for measuring elapsed time."""

    def __init__(self):
        self.elapsed = 0.0

    def __enter__(self):
        self._start = time.perf_counter()
        return self

    def __exit__(self, *args):
        self.elapsed = time.perf_counter() - self._start
