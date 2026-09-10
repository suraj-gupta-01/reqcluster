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
    """Format drift history into an aligned ASCII table for terminal inspection."""
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
    header = (
        f" {'Timestamp':<19}  {'Dataset':<14}  {'Commit':<7}  {'Model':<5}  "
        f"{'Accuracy':>9}  {'ARI':>7}  {'Noise':>7}  {'Ref Recall':>11}  {'Parse Rate':>11}"
    )
    width = len(header) + 2
    divider = "-" * width

    lines = [
        "",
        "=" * width,
        "                          Metric Drift History",
        "=" * width,
        header,
        divider,
    ]

    for r in recent:
        raw_ts = str(r.get("timestamp", ""))
        ts = raw_ts[:19].replace("T", " ")
        ds = str(r.get("dataset", ""))[:14]
        commit = str(r.get("commit", ""))[:7]
        model = str(r.get("model", "real"))[:5]
        m = r.get("metrics", {})

        def _fmt(val, is_pct=False, decimals=2):
            if val is None or val == "":
                return "-"
            try:
                fval = float(val)
                if is_pct:
                    if fval <= 1.0 and fval > 0.0:
                        fval = fval * 100.0
                    return f"{fval:.{decimals}f}%"
                return f"{fval:.{decimals}f}"
            except Exception:
                return str(val)

        acc = _fmt(m.get("accuracy_pct"), is_pct=True, decimals=2)
        ari = _fmt(m.get("ari"), decimals=4)
        noise = _fmt(m.get("noise_pct"), is_pct=True, decimals=2)
        ref_rec = _fmt(m.get("reference_recall"), is_pct=True, decimals=1)
        parse_suc = _fmt(m.get("parse_success_rate"), is_pct=True, decimals=1)

        lines.append(
            f" {ts:<19}  {ds:<14}  {commit:<7}  {model:<5}  "
            f"{acc:>9}  {ari:>7}  {noise:>7}  {ref_rec:>11}  {parse_suc:>11}"
        )

    lines.append("=" * width)
    lines.append("")
    return "\n".join(lines)
