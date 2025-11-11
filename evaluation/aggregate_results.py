"""Aggregate evaluation statistics for POLIS-Bench experiments."""

from __future__ import annotations

import argparse
import json
import re
from math import sqrt
from pathlib import Path

import numpy as np
import pandas as pd


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_RESULTS_DIR = REPO_ROOT / "results"


def _read_jsonl(path: Path) -> list[dict]:
    rows = []
    with path.open("r", encoding="utf-8") as fin:
        for line in fin:
            line = line.strip()
            if not line:
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError:
                continue
    return rows


def _extract_llm_label(text: str | None) -> str:
    if text is None:
        return "unknown"
    match = re.search(r"\[([^\[\]]+)\]\s*$", str(text))
    return match.group(1).strip().lower() if match else "unknown"


def _extract_language(id_str: str | None) -> str:
    if not id_str:
        return "unknown"
    id_str = id_str.strip().lower()
    return id_str[:2] if id_str.startswith("en") or id_str.startswith("cn") else "unknown"


def _extract_task_type(id_str: str | None) -> str:
    if not id_str:
        return "unknown"
    match = re.search(r"(\d{2})$", id_str.strip())
    return match.group(1) if match else "unknown"


def _aggregate(group: pd.DataFrame) -> pd.Series:
    sim = group["similarity"].dropna()
    n_sim = sim.size
    sim_mean = sim.mean() if n_sim > 0 else np.nan
    sim_se = sim.std(ddof=1) / sqrt(n_sim) if n_sim > 1 else (0.0 if n_sim == 1 else np.nan)

    acc = group["is_correct"].dropna()
    n_acc = acc.size
    acc_rate = acc.mean() if n_acc > 0 else np.nan
    acc_se = sqrt(acc_rate * (1 - acc_rate) / n_acc) if n_acc > 0 else np.nan

    return pd.Series(
        {
            "n_total": len(group),
            "similarity_mean": sim_mean,
            "similarity_se": sim_se,
            "accuracy": acc_rate,
            "accuracy_se": acc_se,
        }
    )


def aggregate_file(file_path: Path) -> dict[str, pd.DataFrame]:
    records = _read_jsonl(file_path)
    df = pd.DataFrame(records)
    if df.empty:
        raise ValueError(f"No valid rows found in {file_path}")

    df["llm_label"] = df.get("LLMJudge result", pd.Series([None] * len(df))).apply(
        _extract_llm_label
    )
    df["is_correct"] = df["llm_label"] == "correct"

    if "similarity" in df.columns:
        df["similarity"] = pd.to_numeric(df["similarity"], errors="coerce")
    elif "strict_semantic_score" in df.columns:
        df["similarity"] = pd.to_numeric(df["strict_semantic_score"], errors="coerce")
    else:
        df["similarity"] = np.nan

    df["language"] = df.get("id").apply(_extract_language)
    df["task_type"] = df.get("id").apply(_extract_task_type)

    return {
        "by_model_language": df.groupby(["model", "language"]).apply(
            _aggregate, include_groups=False
        ),
        "by_model_language_id": df.groupby(["model", "language", "id"]).apply(
            _aggregate, include_groups=False
        ),
        "by_model": df.groupby("model").apply(_aggregate, include_groups=False),
        "by_model_task": df.groupby(["model", "task_type"]).apply(
            _aggregate, include_groups=False
        ),
    }


def save_tables(tables: dict[str, pd.DataFrame], output_dir: Path, prefix: str) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    for name, table in tables.items():
        path = output_dir / f"{prefix}_{name}.csv"
        table.reset_index().to_csv(path, index=False)


def process_folder(input_dir: Path, output_dir: Path) -> None:
    for jsonl_path in sorted(input_dir.glob("*.jsonl")):
        prefix = jsonl_path.stem
        tables = aggregate_file(jsonl_path)
        save_tables(tables, output_dir, prefix)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Aggregate POLIS-Bench evaluation results.")
    parser.add_argument(
        "--input",
        type=Path,
        default=DEFAULT_RESULTS_DIR,
        help="Folder containing evaluation JSONL files.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=DEFAULT_RESULTS_DIR / "aggregated",
        help="Where to write aggregated CSVs.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    process_folder(args.input, args.output)


if __name__ == "__main__":
    main()


