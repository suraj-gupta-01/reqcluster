"""Golden baselines loading, persistence, and regression gating logic."""

from __future__ import annotations

from datetime import datetime, UTC
import json
from pathlib import Path
import subprocess
from typing import Any

from .config import DatasetConfig, ToleranceConfig


REPO_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_BASELINES_DIR = REPO_ROOT / "eval" / "baselines"


def get_current_git_commit() -> str:
    try:
        res = subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"],
            cwd=str(REPO_ROOT),
            capture_output=True,
            text=True,
            check=True,
        )
        return res.stdout.strip()
    except Exception:
        return "unknown"


def get_baseline_path(
    dataset_name: str,
    provider: str = "mock",
    dir_path: Path | str | None = None,
) -> Path:
    base_dir = Path(dir_path).resolve() if dir_path else DEFAULT_BASELINES_DIR
    # Standard format: <dataset>_<provider>.json
    return base_dir / f"{dataset_name}_{provider}.json"


def load_baseline(
    dataset_name: str,
    provider: str = "mock",
    dir_path: Path | str | None = None,
) -> dict[str, Any] | None:
    """Load baseline dictionary if it exists."""
    path = get_baseline_path(dataset_name, provider, dir_path)
    if not path.exists():
        # Fallback to <dataset>.json if provider is mock
        fallback = path.parent / f"{dataset_name}.json"
        if fallback.exists():
            path = fallback
        else:
            return None

    try:
        with open(path, encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return None


def save_baseline(
    dataset_name: str,
    provider: str,
    metrics: dict[str, Any],
    commit: str | None = None,
    dir_path: Path | str | None = None,
) -> Path:
    """Write metrics as a golden baseline for a dataset and provider."""
    path = get_baseline_path(dataset_name, provider, dir_path)
    path.parent.mkdir(parents=True, exist_ok=True)

    data = {
        "dataset": dataset_name,
        "provider": provider,
        "created_at": datetime.now(UTC).isoformat(),
        "commit": commit or get_current_git_commit(),
        "metrics": metrics,
    }

    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2)

    return path


def evaluate_metric_gate(
    metric_name: str,
    actual: float | int | None,
    baseline: float | int | None,
    tol: ToleranceConfig,
) -> dict[str, Any]:
    """Check a single metric against floor and delta tolerances."""
    if actual is None:
        return {
            "metric": metric_name,
            "baseline": baseline,
            "actual": None,
            "delta": None,
            "verdict": "FAIL",
            "reason": "Metric value was not produced in run",
        }

    actual_val = float(actual)
    base_val = float(baseline) if baseline is not None else None
    delta = round(actual_val - base_val, 4) if base_val is not None else None

    # Missing baseline on a gated metric
    if base_val is None and (tol.delta is not None or tol.floor is None):
        return {
            "metric": metric_name,
            "baseline": None,
            "actual": actual_val,
            "delta": None,
            "verdict": "FAIL",
            "reason": "Missing baseline for gated metric",
        }

    verdict = "PASS"
    fail_reasons = []

    # Floor check
    if tol.floor is not None:
        if tol.lower_is_better:
            if actual_val > tol.floor:
                verdict = "FAIL"
                fail_reasons.append(f"actual {actual_val} > floor {tol.floor}")
        else:
            if actual_val < tol.floor:
                verdict = "FAIL"
                fail_reasons.append(f"actual {actual_val} < floor {tol.floor}")

    # Delta check
    if tol.delta is not None and base_val is not None:
        threshold = base_val + tol.delta
        if tol.lower_is_better:
            if actual_val > threshold:
                verdict = "FAIL"
                fail_reasons.append(
                    f"actual {actual_val} exceeded baseline allowance ({threshold})"
                )
        else:
            if actual_val < threshold:
                verdict = "FAIL"
                fail_reasons.append(
                    f"actual {actual_val} dropped below baseline allowance ({threshold})"
                )

    return {
        "metric": metric_name,
        "baseline": base_val,
        "actual": actual_val,
        "delta": delta,
        "verdict": verdict,
        "reason": "; ".join(fail_reasons) if fail_reasons else "Within tolerances",
    }


def evaluate_dataset_gating(
    dataset_config: DatasetConfig,
    actual_metrics: dict[str, Any],
    baseline_payload: dict[str, Any] | None,
) -> list[dict[str, Any]]:
    """Evaluate all metrics for a dataset against registry tolerances."""
    baseline_metrics = baseline_payload.get("metrics", {}) if baseline_payload else {}
    results = []

    # 1. Gated metrics
    for metric_name, tol in dataset_config.tolerances.items():
        actual = actual_metrics.get(metric_name)
        base = baseline_metrics.get(metric_name)
        gate_res = evaluate_metric_gate(metric_name, actual, base, tol)
        results.append(gate_res)

    # 2. Informational metrics (present in actual_metrics but not in tolerances)
    for metric_name, actual in actual_metrics.items():
        if metric_name in dataset_config.tolerances:
            continue
        if isinstance(actual, (int, float)) and not isinstance(actual, bool):
            base = baseline_metrics.get(metric_name)
            base_val = float(base) if base is not None else None
            actual_val = float(actual)
            delta = round(actual_val - base_val, 4) if base_val is not None else None
            results.append({
                "metric": metric_name,
                "baseline": base_val,
                "actual": actual_val,
                "delta": delta,
                "verdict": "INFO",
                "reason": "Informational (not gated)",
            })

    return results
