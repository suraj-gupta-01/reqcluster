"""Append-only drift tracking for monitoring metric decay over time."""

from __future__ import annotations

from datetime import datetime, timezone
import json
from pathlib import Path
from typing import Any, Dict, List, Optional


REPO_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_DRIFT_FILE = REPO_ROOT / "eval" / "drift.json"


def load_drift_history(file_path: Optional[Path | str] = None) -> List[Dict[str, Any]]:
    path = Path(file_path).resolve() if file_path else DEFAULT_DRIFT_FILE
    if not path.exists():
        return []
    try:
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
            return data.get("runs", [])
    except Exception:
        return []


def append_drift_entry(
    dataset: str,
    provider: str,
    model: str,
    metrics: Dict[str, Any],
    commit: str = "",
    file_path: Optional[Path | str] = None,
) -> None:
    path = Path(file_path).resolve() if file_path else DEFAULT_DRIFT_FILE
    path.parent.mkdir(parents=True, exist_ok=True)

    runs = load_drift_history(path)
    entry = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "commit": commit or "unknown",
        "dataset": dataset,
        "provider": provider,
        "model": model,
        "metrics": {
            k: v for k, v in metrics.items()
            if isinstance(v, (int, float, str, bool))
        },
    }
    runs.append(entry)

    with open(path, "w", encoding="utf-8") as f:
        json.dump({"runs": runs}, f, indent=2)


def format_drift_table(
    dataset_name: Optional[str] = None,
    n: int = 10,
    file_path: Optional[Path | str] = None,
) -> str:
    runs = load_drift_history(file_path)
    if not runs:
        return "No drift history recorded yet."

    filtered = [
        r for r in runs
        if not dataset_name or r.get("dataset") == dataset_name
    ]

    if not filtered:
        return f"No drift history found for dataset '{dataset_name}'."

    recent = filtered[-n:]
    lines = [f"### Drift History (Last {len(recent)} runs)"]

    for r in recent:
        ts = r.get("timestamp", "")[:19]
        ds = r.get("dataset", "")
        commit = r.get("commit", "")[:7]
        prov = r.get("provider", "")
        m = r.get("metrics", {})
        acc = m.get("accuracy_pct", "-")
        ari = m.get("ari", "-")
        noise = m.get("noise_pct", "-")
        ref_rec = m.get("reference_recall", "-")
        parse_suc = m.get("parse_success_rate", "-")

        lines.append(
            f"- [{ts}] {ds} ({prov}, {commit}): "
            f"acc={acc}%, ari={ari}, noise={noise}%, ref_recall={ref_rec}, parse={parse_suc}"
        )

    return "\n".join(lines)
