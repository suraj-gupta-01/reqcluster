"""Offline unit tests for the evaluation harness components."""

from __future__ import annotations

from pathlib import Path
import pytest

from eval.baselines import evaluate_metric_gate, load_baseline, save_baseline
from eval.config import DatasetConfig, RegistryConfig, ToleranceConfig, resolve_datasets
from eval.deps_metrics import (
    compute_precision_recall_f1,
    extract_expected_reference_edges,
)
from eval.drift import append_drift_entry, load_drift_history
from eval.enrich_metrics import evaluate_enrichment_metrics
from eval.metrics import extract_clustering_metrics
from eval.refine_metrics import (
    build_cluster_label_distribution,
    evaluate_refinement_metrics,
)
from eval.report import build_markdown_report


def test_tolerance_config_parsing():
    tol = ToleranceConfig.from_dict({
        "floor": 95.0,
        "delta": -1.5,
        "lower_is_better": False,
    })
    assert tol.floor == 95.0
    assert tol.delta == -1.5
    assert tol.lower_is_better is False

    tol_lower = ToleranceConfig.from_dict({
        "floor": 10.0,
        "delta": 2.0,
        "lower_is_better": True,
    })
    assert tol_lower.floor == 10.0
    assert tol_lower.delta == 2.0
    assert tol_lower.lower_is_better is True


def test_registry_loading_and_filtering(tmp_path):
    yaml_content = """
datasets:
  sample_fast:
    path: data/sample.csv
    suite: fast
    ground_truth: module
    tolerances:
      accuracy_pct:
        floor: 90.0
        delta: -2.0
  sample_stress:
    path: data/stress.csv
    suite: stress
"""
    yaml_file = tmp_path / "datasets.yaml"
    yaml_file.write_text(yaml_content, encoding="utf-8")

    registry = RegistryConfig.from_yaml(yaml_file)
    assert len(registry.datasets) == 2
    assert "sample_fast" in registry.datasets

    fast_suite = resolve_datasets(registry, suite="fast")
    assert len(fast_suite) == 1
    assert fast_suite[0].name == "sample_fast"

    stress_suite = resolve_datasets(registry, suite="stress")
    assert len(stress_suite) == 1
    assert stress_suite[0].name == "sample_stress"

    exact_ds = resolve_datasets(registry, dataset_name="sample_fast")
    assert len(exact_ds) == 1
    assert exact_ds[0].name == "sample_fast"


def test_gating_logic_higher_is_better():
    tol = ToleranceConfig(floor=90.0, delta=-2.0, lower_is_better=False)

    # 1. Passes within delta and floor
    res = evaluate_metric_gate("accuracy_pct", actual=92.0, baseline=93.0, tol=tol)
    assert res["verdict"] == "PASS"
    assert res["delta"] == -1.0

    # 2. Fails floor
    res_floor_fail = evaluate_metric_gate("accuracy_pct", actual=88.0, baseline=89.0, tol=tol)
    assert res_floor_fail["verdict"] == "FAIL"

    # 3. Fails delta
    res_delta_fail = evaluate_metric_gate("accuracy_pct", actual=91.0, baseline=95.0, tol=tol)
    assert res_delta_fail["verdict"] == "FAIL"

    # 4. Missing baseline -> FAIL
    res_no_base = evaluate_metric_gate("accuracy_pct", actual=95.0, baseline=None, tol=tol)
    assert res_no_base["verdict"] == "FAIL"


def test_gating_logic_lower_is_better():
    tol = ToleranceConfig(floor=10.0, delta=2.0, lower_is_better=True)

    # 1. Passes
    res = evaluate_metric_gate("noise_pct", actual=5.0, baseline=4.0, tol=tol)
    assert res["verdict"] == "PASS"
    assert res["delta"] == 1.0

    # 2. Fails floor
    res_floor_fail = evaluate_metric_gate("noise_pct", actual=12.0, baseline=8.0, tol=tol)
    assert res_floor_fail["verdict"] == "FAIL"

    # 3. Fails delta (allowed rise +2.0 -> base 4.0 + 2.0 = 6.0; actual 7.0 > 6.0)
    res_delta_fail = evaluate_metric_gate("noise_pct", actual=7.0, baseline=4.0, tol=tol)
    assert res_delta_fail["verdict"] == "FAIL"


def test_baseline_save_and_load(tmp_path):
    metrics = {"accuracy_pct": 98.3, "ari": 0.956}
    save_path = save_baseline("test_ds", "mock", metrics, commit="abc1234", dir_path=tmp_path)
    assert save_path.exists()

    loaded = load_baseline("test_ds", "mock", dir_path=tmp_path)
    assert loaded is not None
    assert loaded["dataset"] == "test_ds"
    assert loaded["commit"] == "abc1234"
    assert loaded["metrics"]["accuracy_pct"] == 98.3


def test_drift_append_and_history(tmp_path):
    drift_file = tmp_path / "drift.json"
    append_drift_entry(
        dataset="ds1",
        provider="mock",
        model="real",
        metrics={"accuracy_pct": 95.0},
        commit="c1",
        file_path=drift_file,
    )
    append_drift_entry(
        dataset="ds1",
        provider="mock",
        model="real",
        metrics={"accuracy_pct": 95.5},
        commit="c2",
        file_path=drift_file,
    )

    history = load_drift_history(drift_file)
    assert len(history) == 2
    assert history[0]["commit"] == "c1"
    assert history[1]["metrics"]["accuracy_pct"] == 95.5


def test_deps_metrics_reference_extraction():
    reqs = [
        {"req_id": "REQ-001", "text": "Base sensor reading module."},
        {"req_id": "REQ-002", "text": "Controller relies on data from REQ-001."},
        {"req_id": "REQ-003", "text": "Display shows values according to REQ_002 and REQ-001."},
        {"req_id": "REQ-004", "text": "Independent telemetry system."},
    ]

    expected = extract_expected_reference_edges(reqs)
    assert ("REQ-001", "REQ-002") in expected
    assert ("REQ-002", "REQ-003") in expected
    assert ("REQ-001", "REQ-003") in expected
    assert len(expected) == 3

    # Inferred edges test
    inferred = {("REQ-001", "REQ-002"), ("REQ-002", "REQ-003"), ("REQ-004", "REQ-001")}
    scores = compute_precision_recall_f1(inferred, expected)
    # TP = 2, FP = 1, FN = 1
    # Precision = 2/3 = 0.6667, Recall = 2/3 = 0.6667
    assert scores["precision"] == pytest.approx(0.6667, abs=1e-3)
    assert scores["recall"] == pytest.approx(0.6667, abs=1e-3)


def test_refinement_metrics_judgment():
    reqs = [
        {"req_id": "R1", "cluster_id": 1, "module": "Power"},
        {"req_id": "R2", "cluster_id": 1, "module": "Power"},
        {"req_id": "R3", "cluster_id": 2, "module": "Power"},  # Cluster 2 has same dominant label as 1 -> good merge
        {"req_id": "R4", "cluster_id": 3, "module": "Thermal"},
        {"req_id": "R5", "cluster_id": 4, "module": "Thermal"},
        {"req_id": "R6", "cluster_id": 4, "module": "Avionics"}, # Cluster 4 is impure (50/50) -> good split
    ]

    cluster_dist = build_cluster_label_distribution(reqs)
    assert cluster_dist[1]["Power"] == 2
    assert cluster_dist[4]["Thermal"] == 1
    assert cluster_dist[4]["Avionics"] == 1

    suggestions = {
        "merge_suggestions": [
            {"cluster_a_id": 1, "cluster_b_id": 2},  # Valid (both Power)
            {"cluster_a_id": 1, "cluster_b_id": 3},  # Invalid (Power vs Thermal)
        ],
        "split_suggestions": [
            {"cluster_id": 4},  # Valid (impure cluster)
            {"cluster_id": 1},  # Invalid (pure cluster)
        ],
    }

    metrics = evaluate_refinement_metrics(reqs, suggestions, impurity_threshold=0.90)
    assert metrics["n_merge_suggestions"] == 2
    assert metrics["correct_merges"] == 1
    assert metrics["merge_precision"] == 0.5

    assert metrics["n_split_suggestions"] == 2
    assert metrics["correct_splits"] == 1
    assert metrics["split_precision"] == 0.5
    assert metrics["suggestion_precision"] == 0.5


def test_enrichment_metrics_aggregation():
    results = [
        {
            "status": "succeeded",
            "quality_report": {
                "hallucination_risk_score": 0.05,
                "adjusted_confidence_score": 0.95,
                "lexical_overlap": 0.85,
                "domain_term_coverage": 1.0,
                "warnings": [],
            },
        },
        {
            "status": "succeeded",
            "quality_report": {
                "hallucination_risk_score": 0.15,
                "adjusted_confidence_score": 0.85,
                "lexical_overlap": 0.75,
                "domain_term_coverage": 0.80,
                "warnings": ["Warning 1"],
            },
        },
        {
            "status": "failed",
            "quality_report": None,
        },
    ]

    metrics = evaluate_enrichment_metrics(results)
    assert metrics["total_enrichment_rows"] == 3
    assert metrics["succeeded_enrichment_rows"] == 2
    assert metrics["parse_success_rate"] == pytest.approx(0.6667, abs=1e-3)
    assert metrics["mean_hallucination_risk_score"] == pytest.approx(0.10, abs=1e-3)
    assert metrics["warning_rate"] == 0.5


def test_markdown_report_builder():
    run_meta = {
        "timestamp": "2026-08-28T12:00:00Z",
        "commit": "test1234",
        "model": "real",
        "provider": "mock",
    }
    ds_results = [
        {
            "dataset": "sample_ds",
            "status": "success",
            "gating_rows": [
                {
                    "metric": "accuracy_pct",
                    "baseline": 95.0,
                    "actual": 96.0,
                    "delta": 1.0,
                    "verdict": "PASS",
                    "reason": "Within tolerances",
                }
            ],
        }
    ]

    md = build_markdown_report(run_meta, ds_results, drift_summary="Drift OK")
    assert "# ReqCluster Evaluation Report" in md
    assert "sample_ds" in md
    assert "accuracy_pct" in md
    assert "PASS" in md
    assert "Drift OK" in md


def _mini_csv(path: Path) -> Path:
    rows = ["id,text,module"]
    for i in range(30):
        rows.append(f"R{i},The power unit shall supply regulated voltage level {i} to bus.,Power")
    for i in range(30):
        rows.append(f"R{100 + i},The display shall render the altitude indicator frame {i}.,Display")
    path.write_text("\n".join(rows) + "\n", encoding="utf-8")
    return path


def test_mock_pipeline_and_hermetic_home(tmp_path, monkeypatch):
    from eval.pipeline import run_dataset_pipeline
    from eval.runner import hermetic_client

    csv_path = _mini_csv(tmp_path / "mini.csv")
    home = tmp_path / "home"
    monkeypatch.chdir(tmp_path)
    before = {p for p in tmp_path.rglob("*")}
    ds = DatasetConfig(name="mini", path=csv_path)
    with hermetic_client(model_mode="mock", provider="mock", home=home) as client:
        res = run_dataset_pipeline(client, ds, provider="mock")
    m = res["metrics"]
    assert res["status"] == "success"
    assert m["n"] == 60
    assert m["has_ground_truth"] is True
    assert "ari" in m and "accuracy_pct" in m and "noise_pct" in m
    # runner writes only under its home
    new = {p for p in tmp_path.rglob("*")} - before
    assert all(p == home or home in p.parents for p in new)


def test_extract_clustering_metrics_percent_names():
    m = extract_clustering_metrics({
        "n": 10, "n_clusters": 2, "noise_count": 1, "noise_rate": 0.1,
        "hungarian_accuracy": 0.8, "accuracy_noise_inclusive": 0.7, "purity": 0.9,
        "ari": 0.5, "has_ground_truth": True,
    })
    assert m["noise_pct"] == pytest.approx(10.0)
    # legacy accuracy_pct (and the committed baselines) is purity
    assert m["accuracy_pct"] == m["purity_pct"] == pytest.approx(90.0)
    assert m["accuracy_noise_inclusive_pct"] == pytest.approx(70.0)
    assert m["hungarian_accuracy_pct"] == pytest.approx(80.0)
