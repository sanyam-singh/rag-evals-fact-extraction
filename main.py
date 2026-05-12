"""
RAG Evals — Fact Extraction Evaluation Pipeline

Usage:
    # Step 1: Run inference (generate predicted facts from your model)
    python main.py infer --data test.jsonl --model ft:gpt-4.1-mini-2025-04-14:... --output predictions.csv

    # Step 2: Run evaluation (score predictions against ground truth)
    python main.py eval --data predictions.csv --output detailed_results.csv

    # Step 3: Compute metrics (Precision, Recall, F1/F2/F3)
    python main.py metrics --data detailed_results.csv
"""

import argparse
import os
import pandas as pd

from utils import load_jsonl
from inference import run_inference
from pipeline import run_enhanced_evaluation_pipeline_optimized
from metrics import compute_metrics_from_file


def cmd_infer(args):
    openai_api_key = args.api_key or os.environ.get("OPENAI_API_KEY")
    if not openai_api_key:
        raise ValueError("Provide --api-key or set OPENAI_API_KEY env var")

    df = load_jsonl(args.data)
    df = df.head(args.limit) if args.limit else df
    print(f"Loaded {len(df)} samples from {args.data}")

    run_inference(
        df=df,
        model_id=args.model,
        openai_api_key=openai_api_key,
        max_workers=args.workers,
        output_path=args.output
    )


def cmd_eval(args):
    openai_api_key = args.api_key or os.environ.get("OPENAI_API_KEY")
    if not openai_api_key:
        raise ValueError("Provide --api-key or set OPENAI_API_KEY env var")

    df = pd.read_csv(args.data)
    print(f"Loaded {len(df)} rows from {args.data}")

    run_enhanced_evaluation_pipeline_optimized(
        df=df,
        openai_api_key=openai_api_key,
        output_filename=args.output,
        max_workers=args.workers,
        use_batching=args.batched,
        batch_size=args.batch_size
    )


def cmd_metrics(args):
    compute_metrics_from_file(args.data)


def main():
    parser = argparse.ArgumentParser(description="RAG Fact Extraction Evaluator")
    sub = parser.add_subparsers(dest="command", required=True)

    # infer
    p_infer = sub.add_parser("infer", help="Run model inference on a JSONL dataset")
    p_infer.add_argument("--data", required=True, help="Path to input JSONL file")
    p_infer.add_argument("--model", required=True, help="OpenAI model ID")
    p_infer.add_argument("--output", required=True, help="Output CSV path")
    p_infer.add_argument("--api-key", default=None)
    p_infer.add_argument("--workers", type=int, default=10)
    p_infer.add_argument("--limit", type=int, default=None, help="Limit number of rows")

    # eval
    p_eval = sub.add_parser("eval", help="Evaluate predictions against ground truth")
    p_eval.add_argument("--data", required=True, help="CSV with FT_response and gt_facts columns")
    p_eval.add_argument("--output", default="detailed_results.csv")
    p_eval.add_argument("--api-key", default=None)
    p_eval.add_argument("--workers", type=int, default=40)
    p_eval.add_argument("--batched", action="store_true")
    p_eval.add_argument("--batch-size", type=int, default=100)

    # metrics
    p_metrics = sub.add_parser("metrics", help="Compute Precision/Recall/F1/F2/F3 from results CSV")
    p_metrics.add_argument("--data", required=True, help="Detailed results CSV")

    args = parser.parse_args()

    if args.command == "infer":
        cmd_infer(args)
    elif args.command == "eval":
        cmd_eval(args)
    elif args.command == "metrics":
        cmd_metrics(args)


if __name__ == "__main__":
    main()
