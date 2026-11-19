"""Report generation for evaluation harness runs (Markdown + Tabular Terminal views)."""

from __future__ import annotations

from datetime import datetime, UTC
import json
from pathlib import Path
from typing import Any


REPO_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_RESULTS_DIR = REPO_ROOT / "eval" / "results"


def format_terminal_report(
    run_meta: dict[str, Any],
    dataset_results: list[dict[str, Any]],
) -> str:
    """Format a clean, aligned tabular report optimized for terminal viewing."""
    raw_ts = str(run_meta.get("timestamp", ""))
    timestamp = raw_ts[:19].replace("T", " ") if "T" in raw_ts else raw_ts
    commit = run_meta.get("commit", "unknown")
    model_mode = run_meta.get("model", "real")
    provider = run_meta.get("provider", "mock")

    overall_pass = True
    for ds in dataset_results:
        for row in ds.get("gating_rows", []):
            if row.get("verdict") == "FAIL":
                overall_pass = False
                break

    verdict_str = "[ALL PASSED]" if overall_pass else "[FAILED]"
    width = 96

    lines = [
        "",
        "=" * width,
        "                      ReqCluster Evaluation Benchmark Report",
        "=" * width,
        f" Run Time: {timestamp} UTC  |  Commit: {commit}  |  Model: {model_mode}  |  Provider: {provider}",
        f" Overall Suite Verdict: {verdict_str}",
        "=" * width,
    ]

    for ds in dataset_results:
        ds_name = ds.get("dataset", "unknown")
        ds_status = ds.get("status", "success")
        metrics = ds.get("metrics", {})
        n = metrics.get("n", "-")
        n_clusters = metrics.get("n_clusters", "-")

        lines.append(f"\n[DATASET] {ds_name} (N={n}, Clusters={n_clusters})")
        lines.append("-" * width)
        lines.append(
            f" {'Metric':<32} {'Baseline':>12} {'Actual':>12} {'Delta':>10} {'Verdict':^9} {'Notes'}"
        )
        lines.append("-" * width)

        if ds_status != "success":
            lines.append(f" ERROR: {ds.get('error', 'Execution failed')}")
            lines.append("-" * width)
            continue

        gating_rows = ds.get("gating_rows", [])
        gated_rows = [r for r in gating_rows if r.get("verdict") in ("PASS", "FAIL")]
        info_rows = [r for r in gating_rows if r.get("verdict") not in ("PASS", "FAIL")]

        for row in gated_rows + info_rows:
            metric = row.get("metric", "")
            base = f"{row.get('baseline'):.4g}" if row.get("baseline") is not None else "-"
            actual = f"{row.get('actual'):.4g}" if row.get("actual") is not None else "-"
            delta_val = row.get("delta")
            if delta_val is not None:
                delta_str = f"{'+' if delta_val > 0 else ''}{delta_val:.4g}"
            else:
                delta_str = "-"

            verdict = row.get("verdict", "INFO")
            reason = row.get("reason", "")

            lines.append(
                f" {metric:<32} {base:>12} {actual:>12} {delta_str:>10} [{verdict:^5}] {reason}"
            )

        lines.append("-" * width)

    lines.append("")
    return "\n".join(lines)


def build_markdown_report(
    run_meta: dict[str, Any],
    dataset_results: list[dict[str, Any]],
    drift_summary: str = "",
) -> str:
    """Construct a readable Markdown report from evaluation results."""
    timestamp = run_meta.get("timestamp", datetime.now(UTC).isoformat())
    commit = run_meta.get("commit", "unknown")
    model_mode = run_meta.get("model", "real")
    provider = run_meta.get("provider", "mock")

    overall_pass = True
    for ds in dataset_results:
        for row in ds.get("gating_rows", []):
            if row.get("verdict") == "FAIL":
                overall_pass = False
                break

    verdict_badge = "✅ PASS" if overall_pass else "❌ FAIL"

    lines = [
        "# ReqCluster Evaluation Report",
        "",
        f"- **Timestamp:** `{timestamp}`",
        f"- **Commit:** `{commit}`",
        f"- **Model Mode:** `{model_mode}`",
        f"- **LLM Provider:** `{provider}`",
        f"- **Overall Suite Verdict:** **{verdict_badge}**",
        "",
        "---",
    ]

    for ds in dataset_results:
        ds_name = ds.get("dataset", "unknown")
        ds_status = ds.get("status", "success")
        lines.append(f"\n## Dataset: `{ds_name}`")

        if ds_status != "success":
            lines.append(f"\n> ⚠️ **Run Error:** {ds.get('error', 'Execution failed')}\n")
            continue

        lines.append("")
        lines.append("| Metric | Baseline | Actual | Delta | Verdict | Notes |")
        lines.append("|---|---:|---:|---:|:---:|---|")

        for row in ds.get("gating_rows", []):
            metric = row.get("metric", "")
            base = f"{row.get('baseline'):.4g}" if row.get("baseline") is not None else "-"
            actual = f"{row.get('actual'):.4g}" if row.get("actual") is not None else "-"
            delta_val = row.get("delta")
            if delta_val is not None:
                delta_str = f"{'+' if delta_val > 0 else ''}{delta_val:.4g}"
            else:
                delta_str = "-"

            verdict = row.get("verdict", "INFO")
            verdict_icon = "PASS" if verdict == "PASS" else ("FAIL" if verdict == "FAIL" else "INFO")
            reason = row.get("reason", "")

            lines.append(f"| `{metric}` | {base} | {actual} | {delta_str} | **{verdict_icon}** | {reason} |")

    if drift_summary:
        lines.append("\n---\n")
        lines.append(drift_summary)

    lines.append("")
    return "\n".join(lines)


def save_report(
    report_content: str,
    run_meta: dict[str, Any] | None = None,
    dataset_results: list[dict[str, Any]] | None = None,
    timestamp_str: str | None = None,
    out_dir: Path | str | None = None,
) -> Path:
    """Save markdown and structured JSON report into eval/results/<timestamp>/."""
    base_dir = Path(out_dir).resolve() if out_dir else DEFAULT_RESULTS_DIR
    ts = timestamp_str or datetime.now(UTC).strftime("%Y%m%d_%H%M%S")
    target_dir = base_dir / ts
    target_dir.mkdir(parents=True, exist_ok=True)

    report_path = target_dir / "report.md"
    with open(report_path, "w", encoding="utf-8") as f:
        f.write(report_content)

    if run_meta and dataset_results:
        json_path = target_dir / "report.json"
        with open(json_path, "w", encoding="utf-8") as f:
            json.dump({
                "meta": run_meta,
                "datasets": dataset_results,
            }, f, indent=2)

    return report_path
