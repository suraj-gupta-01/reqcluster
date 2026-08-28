"""Configuration and registry loader for the evaluation harness."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Literal, Optional
import yaml


REPO_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_REGISTRY_PATH = REPO_ROOT / "eval" / "datasets.yaml"


@dataclass
class ToleranceConfig:
    floor: Optional[float] = None
    delta: Optional[float] = None
    lower_is_better: bool = False

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> ToleranceConfig:
        return cls(
            floor=float(data["floor"]) if "floor" in data and data["floor"] is not None else None,
            delta=float(data["delta"]) if "delta" in data and data["delta"] is not None else None,
            lower_is_better=bool(data.get("lower_is_better", False)),
        )


@dataclass
class DatasetConfig:
    name: str
    path: Path
    suite: Literal["fast", "stress"] = "fast"
    ground_truth: str = "module"
    edges: Optional[Path] = None
    tolerances: Dict[str, ToleranceConfig] = field(default_factory=dict)

    @classmethod
    def from_dict(cls, name: str, data: Dict[str, Any], root: Path = REPO_ROOT) -> DatasetConfig:
        raw_path = Path(data["path"])
        path = raw_path if raw_path.is_absolute() else (root / raw_path).resolve()

        edges_path = None
        if "edges" in data and data["edges"]:
            raw_edges = Path(data["edges"])
            edges_path = raw_edges if raw_edges.is_absolute() else (root / raw_edges).resolve()

        tolerances = {}
        for metric_name, tol_data in (data.get("tolerances") or {}).items():
            tolerances[metric_name] = ToleranceConfig.from_dict(tol_data or {})

        return cls(
            name=name,
            path=path,
            suite=data.get("suite", "fast"),
            ground_truth=data.get("ground_truth", "module"),
            edges=edges_path,
            tolerances=tolerances,
        )


@dataclass
class RegistryConfig:
    datasets: Dict[str, DatasetConfig]

    @classmethod
    def from_yaml(cls, path: Path | str = DEFAULT_REGISTRY_PATH) -> RegistryConfig:
        file_path = Path(path).resolve()
        if not file_path.exists():
            raise FileNotFoundError(f"Dataset registry not found at: {file_path}")

        with open(file_path, "r", encoding="utf-8") as f:
            data = yaml.safe_load(f) or {}

        root = file_path.parent.parent
        datasets = {}
        for name, item_data in (data.get("datasets") or {}).items():
            datasets[name] = DatasetConfig.from_dict(name, item_data, root=root)

        return cls(datasets=datasets)


def resolve_datasets(
    registry: RegistryConfig,
    suite: str = "fast",
    dataset_name: Optional[str] = None,
) -> List[DatasetConfig]:
    """Filter registry datasets by suite or exact name."""
    if dataset_name:
        if dataset_name not in registry.datasets:
            raise KeyError(
                f"Dataset '{dataset_name}' not found in registry. "
                f"Available: {list(registry.datasets.keys())}"
            )
        return [registry.datasets[dataset_name]]

    suite_lower = suite.lower()
    if suite_lower == "all":
        return list(registry.datasets.values())

    return [d for d in registry.datasets.values() if d.suite == suite_lower]
