#!/usr/bin/env python3
"""
Generate leaderboard JSON from eval results directory.

Usage:
    python generate_leaderboard.py <eval_dir> [--model_name NAME] [--output OUTPUT_PATH]

Examples:
    python generate_leaderboard.py /eval_dir/e5-omni-7B
    python generate_leaderboard.py /eval_dir/e5-omni-7B --model_name "E5-Omni-7B" --output ./e5-omni-7B.json

The script scans <eval_dir>/<modality>/*_score.json and assembles them into
a single JSON file compatible with the MMEB leaderboard format.

Primary metric per modality:
    visdoc, text  -> ndcg_linear@5
    all others    -> hit@1
"""

import argparse
import json
import glob
import os
from datetime import datetime


# Dataset name fixes: eval output name -> leaderboard canonical name
NAME_FIXES = {
    "MMLongBench-page": "MMLongBench-page-fixed",
    "ViDoSeek-page": "ViDoSeek-page-fixed",
}

# Primary metric per modality
MODALITY_METRIC = {
    "visdoc": "ndcg_linear@5",
    "text":   "ndcg_linear@5",
}
DEFAULT_METRIC = "hit@1"


def collect_scores(eval_dir):
    """Scan eval_dir/<modality>/*_score.json and return nested dict."""
    metrics = {}

    for mod_dir in sorted(os.listdir(eval_dir)):
        mod_path = os.path.join(eval_dir, mod_dir)
        if not os.path.isdir(mod_path) or mod_dir.startswith("_") or mod_dir == "run.log":
            continue

        datasets = {}
        for f in sorted(glob.glob(os.path.join(mod_path, "**", "*_score.json"), recursive=True)):
            ds_name = os.path.basename(f).replace("_score.json", "")
            ds_name = NAME_FIXES.get(ds_name, ds_name)

            if ds_name in datasets:
                continue

            with open(f) as fh:
                datasets[ds_name] = json.load(fh)

        if datasets:
            metrics[mod_dir] = datasets

    return metrics


def print_summary(metrics):
    """Print per-modality averages using the correct primary metric."""
    print("\n  Per-modality averages:")
    for mod in sorted(metrics):
        metric_key = MODALITY_METRIC.get(mod, DEFAULT_METRIC)
        vals = []
        missing = []
        for ds, scores in metrics[mod].items():
            v = scores.get(metric_key)
            if isinstance(v, (int, float)):
                vals.append(v)
            else:
                missing.append(ds)
        if vals:
            avg = sum(vals) / len(vals)
            print(f"    {mod:<10} ({metric_key:<15}): {avg*100:.1f}  ({len(vals)} datasets)")
        else:
            print(f"    {mod:<10}: no valid scores")
        if missing:
            print(f"      missing metric: {missing}")


def main():
    parser = argparse.ArgumentParser(description="Generate leaderboard JSON from eval results")
    parser.add_argument("eval_dir", help="Path to eval results directory (e.g. eval_standard/e5-omni-7B)")
    parser.add_argument("--model_name", default=None, help="Model name for metadata (default: dirname)")
    parser.add_argument("--model_size", type=float, default=0, help="Model size in billions (e.g. 3.1)")
    parser.add_argument("--embedding_dim", type=int, default=0, help="Embedding dimension (e.g. 2048)")
    parser.add_argument("--url", default="", help="Model URL (e.g. HuggingFace link)")
    parser.add_argument("--output", default=None, help="Output JSON path (default: <leaderboard_dir>/<model_name>.json)")
    args = parser.parse_args()

    eval_dir = args.eval_dir.rstrip("/")
    model_name = args.model_name or os.path.basename(eval_dir)
    output_dir = os.path.dirname(os.path.abspath(__file__))
    output_path = args.output or os.path.join(output_dir, f"{model_name}.json")

    if not os.path.isdir(eval_dir):
        print(f"Error: {eval_dir} is not a directory")
        return 1

    print(f"Scanning: {eval_dir}")
    metrics = collect_scores(eval_dir)

    if not metrics:
        print("Error: no score files found")
        return 1

    result = {
        "metadata": {
            "model_name": model_name,
            "model_size": args.model_size,
            "embedding_dimension": args.embedding_dim,
            "url": args.url,
            "data_source": "Reproduced",
            "report_generated_date": datetime.now().isoformat(),
        },
        "metrics": metrics,
    }

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(result, f, indent=4)

    total = sum(len(ds) for ds in metrics.values())
    print(f"\nOutput: {output_path}")
    print(f"Total: {total} datasets across {len(metrics)} modalities")
    for mod in sorted(metrics):
        print(f"  {mod}: {len(metrics[mod])}")

    print_summary(metrics)

    return 0


if __name__ == "__main__":
    exit(main())
