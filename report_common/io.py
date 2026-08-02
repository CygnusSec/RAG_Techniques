"""
I/O utilities for saving experiment results.

Handles JSONL per-question output, CSV summaries, prevents accidental overwrites,
and provides experiment discovery and comparison functions.
"""

import csv
import json
import time
from pathlib import Path
from typing import Any, Dict, List

import numpy as np
import pandas as pd


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


# ---------------------------------------------------------------------------
# Experiment Discovery and Comparison
# ---------------------------------------------------------------------------


def discover_experiments(results_dir: Path) -> Dict[str, List[Dict[str, Any]]]:
    """
    Discover all A*.jsonl files in results_dir, sorted alphanumerically.

    Returns dict mapping experiment_name (file stem) -> list of records.
    Files that cannot be parsed are skipped with a warning.
    """
    results_dir = Path(results_dir)
    experiments: Dict[str, List[Dict[str, Any]]] = {}
    for jsonl_file in sorted(results_dir.glob("A*.jsonl")):
        try:
            records = load_jsonl(jsonl_file)
            if records:
                experiments[jsonl_file.stem] = records
            else:
                print(f"Warning: {jsonl_file.name} has no parseable records, skipping.")
        except Exception as e:
            print(f"Warning: could not load {jsonl_file.name}: {e}")
    return experiments


def build_comparison_table(experiments: Dict[str, List[Dict[str, Any]]]) -> pd.DataFrame:
    """
    Build a comparison table with one row per experiment.

    Columns include: experiment, n_questions, mean of each numeric metric found
    in records' 'metrics' dictionary, mean_latency, and p95_latency.
    """
    rows = []
    for name, records in experiments.items():
        row: Dict[str, Any] = {"experiment": name, "n_questions": len(records)}

        # Collect all numeric metric values across records
        metric_values: Dict[str, List[float]] = {}
        latencies: List[float] = []

        for rec in records:
            metrics = rec.get("metrics", {})
            if not isinstance(metrics, dict):
                continue
            for key, val in metrics.items():
                if isinstance(val, (int, float)) and not isinstance(val, bool):
                    metric_values.setdefault(key, []).append(float(val))

            # Extract total latency
            latency = rec.get("latency", {})
            if isinstance(latency, dict):
                total = latency.get("total_seconds")
                if isinstance(total, (int, float)):
                    latencies.append(float(total))
            elif isinstance(latency, (int, float)):
                latencies.append(float(latency))

        # Compute means for each metric
        for key, vals in sorted(metric_values.items()):
            row[key] = float(np.mean(vals))

        # Latency statistics
        if latencies:
            row["mean_latency"] = float(np.mean(latencies))
            row["p95_latency"] = float(np.percentile(latencies, 95))
        else:
            row["mean_latency"] = np.nan
            row["p95_latency"] = np.nan

        rows.append(row)

    return pd.DataFrame(rows)


def build_category_breakdown(
    experiment_records: List[Dict[str, Any]],
    questions: List[Dict[str, Any]],
) -> pd.DataFrame:
    """
    Per-category metric averages for one experiment.

    Joins experiment records with questions (via question_id) to get the category,
    then computes the mean of each numeric metric grouped by category.
    """
    # Build question_id -> category lookup
    category_map: Dict[str, str] = {}
    for q in questions:
        qid = q.get("question_id", "")
        cat = q.get("category", "unknown")
        if qid:
            category_map[qid] = cat

    # Flatten records into rows with category and metrics
    flat_rows = []
    for rec in experiment_records:
        qid = rec.get("question_id", "")
        category = category_map.get(qid, "unknown")
        metrics = rec.get("metrics", {})
        if not isinstance(metrics, dict):
            continue
        row: Dict[str, Any] = {"category": category}
        for key, val in metrics.items():
            if isinstance(val, (int, float)) and not isinstance(val, bool):
                row[key] = float(val)
        flat_rows.append(row)

    if not flat_rows:
        return pd.DataFrame()

    df = pd.DataFrame(flat_rows)
    # Group by category and compute means for numeric columns
    numeric_cols = [c for c in df.columns if c != "category"]
    return df.groupby("category")[numeric_cols].mean().reset_index()


def build_pairwise_deltas(
    experiments: Dict[str, List[Dict[str, Any]]],
    baseline_name: str = "A0",
) -> pd.DataFrame:
    """
    Delta of each experiment vs baseline for all mean metrics.

    Computes the comparison table first, then subtracts the baseline row's
    metric values from each other experiment's row.
    Returns a DataFrame with one row per non-baseline experiment showing deltas.
    """
    comparison = build_comparison_table(experiments)
    if comparison.empty or baseline_name not in comparison["experiment"].values:
        return pd.DataFrame()

    baseline_row = comparison[comparison["experiment"] == baseline_name].iloc[0]

    # Identify numeric metric columns (exclude experiment name, n_questions)
    metric_cols = [
        c for c in comparison.columns
        if c not in ("experiment", "n_questions")
        and pd.api.types.is_numeric_dtype(comparison[c])
    ]

    rows = []
    for _, row in comparison.iterrows():
        if row["experiment"] == baseline_name:
            continue
        delta_row: Dict[str, Any] = {"experiment": row["experiment"]}
        for col in metric_cols:
            baseline_val = baseline_row[col]
            current_val = row[col]
            if pd.notna(baseline_val) and pd.notna(current_val):
                delta_row[col] = float(current_val - baseline_val)
            else:
                delta_row[col] = np.nan
        rows.append(delta_row)

    return pd.DataFrame(rows)
