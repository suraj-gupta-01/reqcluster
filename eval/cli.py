"""CLI interface for the ReqCluster Evaluation Harness."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import logging
from pathlib import Path
import sys
from typing import List, Optional

from .baselines import (
    evaluate_dataset_gating,
    get_current_git_commit,
    load_baseline,
    save_baseline,
)
from .config import RegistryConfig, resolve_datasets
from .drift import append_drift_entry, format_drift_table
from .pipeline import run_dataset_pipeline
from .report import build_markdown_report, format_terminal_report, save_report
from .runner import hermetic_client


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger("eval")


# Ensure stdout/stderr handles unicode without crashing on Windows cp1252
if hasattr(sys.stdout, "reconfigure"):
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass
if hasattr(sys.stderr, "reconfigure"):
    try:
        sys.stderr.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass


def run_eval_command(args: argparse.Namespace) -> int:
    registry = RegistryConfig.from_yaml()
    datasets = resolve_datasets(registry, suite=args.suite, dataset_name=args.dataset)

    commit = get_current_git_commit()
    timestamp = datetime.now(timezone.utc).isoformat()
    run_meta = {
        "timestamp": timestamp,
        "commit": commit,
        "model": args.model,
        "provider": args.provider,
    }

    print(f"\n[RUN] Running ReqCluster Evaluation Harness")
    print(f"      Suite: {args.suite} | Model: {args.model} | Provider: {args.provider} | Commit: {commit}\n")

    dataset_results = []
    has_failure = False

    with hermetic_client(model_mode=args.model, provider=args.provider) as client:
        for ds in datasets:
            print(f"[*] Evaluating dataset: {ds.name} ... ", end="", flush=True)
            try:
                pipe_res = run_dataset_pipeline(
                    client=client,
                    dataset_config=ds,
                    provider=args.provider,
                )
                actual_metrics = pipe_res.get("metrics", {})
                baseline_data = load_baseline(ds.name, provider=args.provider)

                gating_rows = evaluate_dataset_gating(ds, actual_metrics, baseline_data)

                # Check if this dataset failed any gates
                ds_failed = any(r.get("verdict") == "FAIL" for r in gating_rows)
                if ds_failed:
                    has_failure = True
                    print("[FAIL]")
                else:
                    print("[PASS]")

                # Append drift
                append_drift_entry(
                    dataset=ds.name,
                    provider=args.provider,
                    model=args.model,
                    metrics=actual_metrics,
                    commit=commit,
                )

                dataset_results.append({
                    "dataset": ds.name,
                    "status": "success",
                    "metrics": actual_metrics,
                    "gating_rows": gating_rows,
                })

            except Exception as exc:
                has_failure = True
                print(f"[ERROR] {exc}")
                logger.exception("Error evaluating dataset %s", ds.name)
                dataset_results.append({
                    "dataset": ds.name,
                    "status": "error",
                    "error": str(exc),
                    "metrics": {},
                    "gating_rows": [],
                })

    drift_summary = format_drift_table(n=5)
    report_md = build_markdown_report(run_meta, dataset_results, drift_summary)
    report_path = save_report(
        report_content=report_md,
        run_meta=run_meta,
        dataset_results=dataset_results,
    )

    terminal_summary = format_terminal_report(run_meta, dataset_results)
    print(terminal_summary)
    print(drift_summary)

    print("=" * 96)
    print(f" Evaluation Verdict: {'[FAILED]' if has_failure else '[ALL PASSED]'}")
    print(f" Detailed Markdown Report: {report_path}")
    print("=" * 96 + "\n")

    return 1 if has_failure else 0


def update_baselines_command(args: argparse.Namespace) -> int:
    registry = RegistryConfig.from_yaml()
    datasets = resolve_datasets(registry, suite=args.suite, dataset_name=args.dataset)

    commit = get_current_git_commit()
    print(f"\n[BASELINES] Updating Golden Baselines for Suite: {args.suite} (Provider: {args.provider})")

    with hermetic_client(model_mode=args.model, provider=args.provider) as client:
        for ds in datasets:
            print(f"    Running pipeline for {ds.name} ... ", end="", flush=True)
            try:
                pipe_res = run_dataset_pipeline(
                    client=client,
                    dataset_config=ds,
                    provider=args.provider,
                )
                metrics = pipe_res.get("metrics", {})
                saved_path = save_baseline(
                    dataset_name=ds.name,
                    provider=args.provider,
                    metrics=metrics,
                    commit=commit,
                )
                print(f"Saved -> {saved_path.name}")
            except Exception as exc:
                print(f"FAILED: {exc}")
                logger.exception("Failed to update baseline for %s", ds.name)

    print("\n[OK] Baselines update complete.\n")
    return 0


def drift_command(args: argparse.Namespace) -> int:
    summary = format_drift_table(dataset_name=args.dataset, n=args.n)
    print(summary)
    return 0


def report_command(args: argparse.Namespace) -> int:
    import json
    results_dir = Path(__file__).resolve().parent / "results"
    if not results_dir.exists():
        print("No reports found.")
        return 0

    subdirs = sorted(
        [d for d in results_dir.iterdir() if d.is_dir()],
        key=lambda d: d.name,
        reverse=True,
    )
    if not subdirs:
        print("No report directories found.")
        return 0

    latest_dir = subdirs[0]
    json_path = latest_dir / "report.json"
    if json_path.exists():
        try:
            with open(json_path, "r", encoding="utf-8") as f:
                data = json.load(f)
            meta = data.get("meta", {})
            datasets = data.get("datasets", [])
            print(format_terminal_report(meta, datasets))
            print(format_drift_table(n=5))
            print(f"Report file: {latest_dir / 'report.md'}\n")
            return 0
        except Exception:
            pass

    # Fallback to reading markdown file
    latest_report = latest_dir / "report.md"
    if latest_report.exists():
        with open(latest_report, "r", encoding="utf-8") as f:
            print("\n" + f.read() + "\n")

    return 0


def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(
        prog="python -m eval",
        description="ReqCluster Evaluation Harness & Regression Guard Suite",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    # 1. run
    run_parser = subparsers.add_parser("run", help="Run evaluation and gate on baselines")
    run_parser.add_argument(
        "--model",
        choices=["real", "mock"],
        default="real",
        help="Model mode: real SBERT model or mock dummy encoder (default: real)",
    )
    run_parser.add_argument(
        "--provider",
        choices=["mock", "local", "openai_compatible"],
        default="mock",
        help="LLM provider mode (default: mock)",
    )
    run_parser.add_argument(
        "--suite",
        choices=["fast", "stress", "all"],
        default="fast",
        help="Dataset suite to evaluate (default: fast)",
    )
    run_parser.add_argument(
        "--dataset",
        type=str,
        default=None,
        help="Evaluate only a specific dataset by name",
    )

    # 2. update-baselines
    base_parser = subparsers.add_parser(
        "update-baselines",
        help="Update committed golden baselines",
    )
    base_parser.add_argument(
        "--model",
        choices=["real", "mock"],
        default="real",
        help="Model mode (default: real)",
    )
    base_parser.add_argument(
        "--provider",
        choices=["mock", "local", "openai_compatible"],
        default="mock",
        help="LLM provider mode (default: mock)",
    )
    base_parser.add_argument(
        "--suite",
        choices=["fast", "stress", "all"],
        default="fast",
        help="Dataset suite (default: fast)",
    )
    base_parser.add_argument(
        "--dataset",
        type=str,
        default=None,
        help="Update a specific dataset by name",
    )

    # 3. drift
    drift_parser = subparsers.add_parser("drift", help="View append-only metric drift history")
    drift_parser.add_argument("--n", type=int, default=10, help="Number of past runs to display")
    drift_parser.add_argument("--dataset", type=str, default=None, help="Filter by dataset name")

    # 4. report
    report_parser = subparsers.add_parser("report", help="View latest evaluation report")
    report_parser.add_argument("--latest", action="store_true", default=True, help="Show latest report")

    args = parser.parse_args(argv)

    if args.command == "run":
        return run_eval_command(args)
    elif args.command == "update-baselines":
        return update_baselines_command(args)
    elif args.command == "drift":
        return drift_command(args)
    elif args.command == "report":
        return report_command(args)

    return 0


if __name__ == "__main__":
    sys.exit(main())
