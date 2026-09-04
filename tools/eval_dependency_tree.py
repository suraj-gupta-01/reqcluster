#!/usr/bin/env python
"""Evaluation harness for the DP5 dependency tree engine.

Runs ``build_dependency_tree`` over a requirements CSV and scores the result
against a hand-labeled gold edge file, reporting precision / recall / F1 broken
out by relation type and split across three axes (detection, typed, undirected —
see ``core.dependency_eval``).

Mirrors ``tools/stress_test_reqcluster.py``: a standalone script that writes a
timestamped report, here into ``eval-results/``.

Usage
-----
Score one configuration::

    python tools/eval_dependency_tree.py data/aerospace_requirements.csv

Compare the regex baseline against spaCy, with and without artifact near-match
(this is what Phase 0's baseline and Phase 2's exit criterion are read from)::

    python tools/eval_dependency_tree.py data/aerospace_requirements.csv --compare

The gold file defaults to ``<dataset>_gold_edges.csv`` alongside the dataset.
Both CSV (``source_id,target_id,relation``) and JSON list-of-objects are read.
"""

from __future__ import annotations

import argparse
import csv
import json
import logging
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
for _path in (str(ROOT), str(ROOT / "backend")):
    if _path not in sys.path:
        sys.path.insert(0, _path)

from core.artifact_extractor import reset_nlp_cache, spacy_available  # noqa: E402
from core.dependency_eval import score_dependency_tree, summarize_errors  # noqa: E402
from core.dependency_tree import build_dependency_tree  # noqa: E402
from core.embeddings import generate_embeddings  # noqa: E402

logger = logging.getLogger("eval_dependency_tree")


# ---------------------------------------------------------------------------
# Loading
# ---------------------------------------------------------------------------

def load_requirements(csv_path: Path) -> Tuple[List[str], List[str], List[str]]:
    """Load ``(texts, req_ids, modules)`` from a requirements CSV."""
    with open(csv_path, "r", encoding="utf-8-sig", newline="") as handle:
        rows = list(csv.DictReader(handle))
    if not rows:
        raise ValueError(f"{csv_path} contains no rows.")

    fields = list(rows[0].keys())

    def pick(*candidates: str) -> Optional[str]:
        lowered = {f.lower().strip(): f for f in fields}
        for candidate in candidates:
            if candidate in lowered:
                return lowered[candidate]
        return None

    id_field = pick("id", "req_id", "requirement_id") or fields[0]
    text_field = pick("text", "requirement", "description") or fields[1]
    module_field = pick("module", "subsystem", "component")

    texts = [str(r.get(text_field) or "").strip() for r in rows]
    req_ids = [str(r.get(id_field) or "").strip() for r in rows]
    modules = [
        str(r.get(module_field) or "").strip() if module_field else "" for r in rows
    ]
    return texts, req_ids, modules


def load_gold_edges(path: Optional[Path]) -> List[Dict[str, Any]]:
    """Load gold edges from CSV (``source_id,target_id,relation``) or JSON."""
    if path is None or not path.exists():
        return []
    if path.suffix.lower() == ".json":
        with open(path, "r", encoding="utf-8") as handle:
            return json.load(handle)
    with open(path, "r", encoding="utf-8-sig", newline="") as handle:
        return [dict(row) for row in csv.DictReader(handle)]


def default_gold_path(csv_path: Path) -> Optional[Path]:
    for suffix in (".csv", ".json"):
        candidate = csv_path.with_name(f"{csv_path.stem}_gold_edges{suffix}")
        if candidate.exists():
            return candidate
    return None


# ---------------------------------------------------------------------------
# Running
# ---------------------------------------------------------------------------

def compute_labels(
    texts: Sequence[str],
    req_ids: Sequence[str],
    modules: Sequence[str],
    label_source: str,
) -> np.ndarray:
    """Cluster labels for the hierarchical pass.

    ``module`` uses the dataset's own subsystem column — deterministic, and the
    right choice when measuring the dependency engine rather than the clusterer.
    ``pipeline`` runs the real UMAP+HDBSCAN path for an end-to-end number.
    """
    if label_source == "module" and any(modules):
        seen: Dict[str, int] = {}
        out: List[int] = []
        for module in modules:
            if module not in seen:
                seen[module] = len(seen)
            out.append(seen[module])
        return np.asarray(out, dtype=np.int32)

    if label_source == "pipeline":
        from core.pipeline import run_pipeline  # noqa: PLC0415 — heavy import

        results = run_pipeline(texts=list(texts), req_ids=list(req_ids))
        return np.asarray(results["labels"])

    return np.full(len(texts), -1, dtype=np.int32)


def run_variant(
    texts: Sequence[str],
    req_ids: Sequence[str],
    embeddings: np.ndarray,
    labels: np.ndarray,
    gold_edges: Sequence[Dict[str, Any]],
    *,
    backend: str,
    near_match: bool,
    near_match_threshold: float,
    top_k: int,
    sim_threshold: float,
) -> Dict[str, Any]:
    """Build and score one engine configuration."""
    import core.artifact_extractor as extractor

    reset_nlp_cache()
    if backend == "regex":
        # Force the fallback path without touching the environment globally.
        extractor._nlp = None
        extractor._nlp_loaded = True
    elif backend == "spacy" and not spacy_available():
        raise RuntimeError("spaCy backend requested but unavailable.")

    tree = build_dependency_tree(
        embeddings=embeddings,
        texts=list(texts),
        req_ids=list(req_ids),
        labels=labels,
        top_k=top_k,
        sim_threshold=sim_threshold,
        enable_artifact_near_match=near_match,
        artifact_near_match_threshold=near_match_threshold,
    )
    reset_nlp_cache()

    id_to_index = {
        node["node_id"]: node["id"] for node in tree["nodes"]
    }
    metrics = score_dependency_tree(tree["edges"], list(gold_edges), id_to_index)
    errors = summarize_errors(tree["edges"], list(gold_edges), id_to_index)

    return {
        "backend": backend,
        "near_match": near_match,
        "near_match_threshold": near_match_threshold if near_match else None,
        "n_edges": len(tree["edges"]),
        "stats": tree["stats"],
        "metrics": metrics,
        "errors": errors,
    }


# ---------------------------------------------------------------------------
# Reporting
# ---------------------------------------------------------------------------

def _variant_name(variant: Dict[str, Any]) -> str:
    name = variant["backend"]
    if variant["near_match"]:
        name += f" + near-match@{variant['near_match_threshold']:.2f}"
    return name


def _metric_row(label: str, block: Dict[str, Any]) -> str:
    return (
        f"| {label} | {block['precision']:.3f} | {block['recall']:.3f} | "
        f"{block['f1']:.3f} | {block['true_positives']} | "
        f"{block['false_positives']} | {block['false_negatives']} | "
        f"{block['support']} |"
    )


def build_markdown_report(
    dataset: Path,
    gold_path: Optional[Path],
    variants: List[Dict[str, Any]],
    label_source: str,
    n_requirements: int,
    n_gold: int,
) -> str:
    lines: List[str] = [
        "# Dependency Engine Evaluation",
        "",
        f"- **Dataset:** `{dataset}` ({n_requirements} requirements)",
        f"- **Gold edges:** `{gold_path}` ({n_gold} labeled)" if gold_path
        else "- **Gold edges:** none (unscored run)",
        f"- **Cluster labels:** {label_source}",
        f"- **Generated:** {datetime.now(timezone.utc).isoformat(timespec='seconds')}",
        f"- **spaCy available:** {spacy_available()}",
        "",
        "## Summary",
        "",
        "| Configuration | Edges | Detection F1 | Typed F1 | Undirected F1 |",
        "|---|---|---|---|---|",
    ]
    for variant in variants:
        metrics = variant["metrics"]
        lines.append(
            f"| {_variant_name(variant)} | {variant['n_edges']} | "
            f"{metrics['detection']['f1']:.3f} | {metrics['typed']['f1']:.3f} | "
            f"{metrics['undirected']['f1']:.3f} |"
        )

    for variant in variants:
        metrics = variant["metrics"]
        lines += [
            "",
            f"## {_variant_name(variant)}",
            "",
            "| Axis | P | R | F1 | TP | FP | FN | Support |",
            "|---|---|---|---|---|---|---|---|",
            _metric_row("detection (any edge)", metrics["detection"]),
            _metric_row("typed (correct label)", metrics["typed"]),
            _metric_row("undirected (ignore arrow)", metrics["undirected"]),
        ]
        if metrics["per_relation"]:
            lines += ["", "### By relation type", "",
                      "| Relation | P | R | F1 | TP | FP | FN | Support |",
                      "|---|---|---|---|---|---|---|---|"]
            for relation, block in metrics["per_relation"].items():
                lines.append(_metric_row(relation, block))

        errors = variant["errors"]
        totals = errors["totals"]
        lines += [
            "",
            f"### Error breakdown "
            f"(missed {totals['missed']}, spurious {totals['spurious']}, "
            f"mislabeled {totals['mislabeled']})",
            "",
        ]
        if errors["missed"]:
            lines += ["**Missed gold edges**", ""]
            lines += [
                f"- `{e['source_id']} -> {e['target_id']}` ({e['gold_relation']})"
                for e in errors["missed"]
            ]
            lines.append("")
        if errors["mislabeled"]:
            lines += ["**Wrong relation label**", ""]
            lines += [
                f"- `{e['source_id']} -> {e['target_id']}`: predicted "
                f"*{e['predicted_relation']}*, gold *{e['gold_relation']}*"
                for e in errors["mislabeled"]
            ]
            lines.append("")

        near = variant["stats"].get("artifact_near_match")
        if near and near.get("merges"):
            lines += ["**Artifact synonym merges**", ""]
            lines += [
                f"- `{m['from']}` -> `{m['to']}` (similarity {m['similarity']:.3f})"
                for m in near["merges"]
            ]
            lines.append("")

    return "\n".join(lines) + "\n"


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(
        description="Evaluate the dependency tree engine against gold edges."
    )
    parser.add_argument("csv_path", type=Path, help="Requirements CSV.")
    parser.add_argument(
        "--gold-edges", type=Path, default=None,
        help="Gold edge CSV/JSON (default: <dataset>_gold_edges.csv).",
    )
    parser.add_argument(
        "--output-dir", type=Path, default=ROOT / "eval-results",
        help="Where to write the report (default: eval-results/).",
    )
    parser.add_argument(
        "--backend", choices=("auto", "spacy", "regex"), default="auto",
        help="Artifact-extraction backend for a single run.",
    )
    parser.add_argument(
        "--near-match", action="store_true",
        help="Enable embedding near-match on artifact names.",
    )
    parser.add_argument(
        "--near-match-threshold", type=float, default=0.80,
        help="Cosine threshold for artifact near-match (default: 0.80).",
    )
    parser.add_argument(
        "--compare", action="store_true",
        help="Run regex / spaCy / spaCy+near-match and tabulate all three.",
    )
    parser.add_argument(
        "--labels", choices=("module", "pipeline", "none"), default="module",
        help="Source of cluster labels for the hierarchical pass.",
    )
    parser.add_argument("--top-k", type=int, default=8)
    parser.add_argument("--sim-threshold", type=float, default=0.45)
    parser.add_argument(
        "--baseline-out", type=Path, default=None,
        help="Also write the first variant's metrics to this JSON path "
             "(used to capture Phase 0's baseline_metrics.json).",
    )
    parser.add_argument("--quiet", action="store_true")
    args = parser.parse_args(argv)

    logging.basicConfig(
        level=logging.WARNING if args.quiet else logging.INFO,
        format="%(levelname)s %(name)s: %(message)s",
    )

    texts, req_ids, modules = load_requirements(args.csv_path)
    gold_path = args.gold_edges or default_gold_path(args.csv_path)
    gold_edges = load_gold_edges(gold_path)
    if not gold_edges:
        logger.warning(
            "No gold edges found (looked for %s) — reporting edge counts only.",
            gold_path,
        )

    logger.info("Loaded %d requirements from %s", len(texts), args.csv_path)
    embeddings = generate_embeddings(texts)
    labels = compute_labels(texts, req_ids, modules, args.labels)

    if args.compare:
        configurations = [
            ("regex", False),
            ("spacy", False),
            ("spacy", True),
        ]
        if not spacy_available():
            logger.warning("spaCy unavailable — comparing the regex path only.")
            configurations = [("regex", False)]
    else:
        configurations = [(args.backend, args.near_match)]

    variants: List[Dict[str, Any]] = []
    for backend, near_match in configurations:
        logger.info("Running variant: backend=%s near_match=%s", backend, near_match)
        variants.append(
            run_variant(
                texts, req_ids, embeddings, labels, gold_edges,
                backend=backend,
                near_match=near_match,
                near_match_threshold=args.near_match_threshold,
                top_k=args.top_k,
                sim_threshold=args.sim_threshold,
            )
        )

    args.output_dir.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    stem = f"dependency_{args.csv_path.stem}_{stamp}"

    report_md = build_markdown_report(
        args.csv_path, gold_path, variants, args.labels, len(texts), len(gold_edges)
    )
    md_path = args.output_dir / f"{stem}.md"
    md_path.write_text(report_md, encoding="utf-8")

    json_path = args.output_dir / f"{stem}.json"
    json_path.write_text(
        json.dumps(
            {
                "dataset": str(args.csv_path),
                "gold_edges_path": str(gold_path) if gold_path else None,
                "generated_at": datetime.now(timezone.utc).isoformat(),
                "n_requirements": len(texts),
                "n_gold_edges": len(gold_edges),
                "label_source": args.labels,
                "spacy_available": spacy_available(),
                "variants": variants,
            },
            indent=2,
        ),
        encoding="utf-8",
    )

    if args.baseline_out and variants:
        args.baseline_out.parent.mkdir(parents=True, exist_ok=True)
        args.baseline_out.write_text(
            json.dumps(
                {
                    "dataset": str(args.csv_path),
                    "gold_edges_path": str(gold_path) if gold_path else None,
                    "generated_at": datetime.now(timezone.utc).isoformat(),
                    "configuration": _variant_name(variants[0]),
                    "n_requirements": len(texts),
                    "n_gold_edges": len(gold_edges),
                    "label_source": args.labels,
                    "metrics": variants[0]["metrics"],
                    "stats": variants[0]["stats"],
                },
                indent=2,
            ),
            encoding="utf-8",
        )
        print(f"Baseline written to {args.baseline_out}")

    print()
    print(report_md.split("## Summary", 1)[-1].split("\n##", 1)[0].strip())
    print()
    print(f"Full report: {md_path}")
    print(f"Raw JSON:    {json_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
