"""Markdown report generator for evaluation harness runs."""

from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional


REPO_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_RESULTS_DIR = REPO_ROOT / "eval" / "results"


def build_markdown_report(
    run_meta: Dict[str, Any],
    dataset_results: List[Dict[str, Any]],
    drift_summary: str = "",
) -> str:
    """Construct a readable Markdown report from evaluation results."""
    timestamp = run_meta.get("timestamp", datetime.now(timezone.utc).isoformat())
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
    timestamp_str: Optional[str] = None,
    out_dir: Optional[Path | str] = None,
) -> Path:
    """Save markdown report into eval/results/<timestamp>/report.md."""
    base_dir = Path(out_dir).resolve() if out_dir else DEFAULT_RESULTS_DIR
    ts = timestamp_str or datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    target_dir = base_dir / ts
    target_dir.mkdir(parents=True, exist_ok=True)

    report_path = target_dir / "report.md"
    with open(report_path, "w", encoding="utf-8") as f:
        f.write(report_content)

    return report_path
