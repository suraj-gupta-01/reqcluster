"""Dependency-tree evaluation scoring (DP5 hardening, Phase 1).

Scores the output of :func:`core.dependency_tree.build_dependency_tree` against a
hand-labeled gold edge set.

Three axes are reported separately, because collapsing them hides *why* the
engine is wrong:

``detection``
    Did we find *an* edge between this ordered pair at all, regardless of the
    relation label?  Low detection recall means "found nothing".

``typed``
    Did we find the *correct relation label* for that pair?  High detection but
    low typed means "found something, labelled it wrong" — a different bug.

``undirected``
    Did we find the pair at all, ignoring arrow direction?  Undirected recall
    materially above detection recall means the engine is inverting edges.

This module is pure: no I/O, no DB, no LLM — consistent with the rest of
``core/``.
"""

from __future__ import annotations

import logging
from typing import Any, Dict, Iterable, List, Sequence, Set, Tuple

logger = logging.getLogger(__name__)

Triple = Tuple[str, str, str]
Pair = Tuple[str, str]


def _prf(tp: int, fp: int, fn: int) -> Dict[str, Any]:
    """Precision / recall / F1 from a confusion triple.

    A zero denominator scores 0.0 rather than raising; the accompanying counts
    make an all-zero row unambiguous to the reader.
    """
    precision = tp / (tp + fp) if (tp + fp) else 0.0
    recall = tp / (tp + fn) if (tp + fn) else 0.0
    f1 = (
        2 * precision * recall / (precision + recall)
        if (precision + recall)
        else 0.0
    )
    return {
        "precision": round(precision, 4),
        "recall": round(recall, 4),
        "f1": round(f1, 4),
        "true_positives": tp,
        "false_positives": fp,
        "false_negatives": fn,
        "support": tp + fn,
    }


def _score_sets(predicted: Set, gold: Set) -> Dict[str, Any]:
    return _prf(
        tp=len(predicted & gold),
        fp=len(predicted - gold),
        fn=len(gold - predicted),
    )


def _undirected(pairs: Iterable[Pair]) -> Set[Pair]:
    """Canonicalise ordered pairs to unordered ones (sorted tuple)."""
    return {(a, b) if a <= b else (b, a) for a, b in pairs}


def score_dependency_tree(
    predicted_edges: Sequence[Dict[str, Any]],
    gold_edges: Sequence[Dict[str, Any]],
    id_to_index: Dict[str, int],
) -> Dict[str, Any]:
    """Score predicted dependency edges against a gold edge set.

    Args:
        predicted_edges: ``build_dependency_tree(...)["edges"]`` — dicts carrying
            integer ``source``/``target`` node indices and a ``relation`` label.
        gold_edges: Hand-labeled truth — dicts with ``source_id``, ``target_id``
            and ``relation`` (string requirement IDs, not indices).
        id_to_index: ``{requirement_id: node_index}`` for the same run that
            produced *predicted_edges*.  Used both to translate predictions back
            into requirement IDs and to discard gold edges that reference
            requirements absent from the dataset.

    Returns:
        ``{"per_relation": {...}, "detection": {...}, "typed": {...},
        "undirected": {...}, "counts": {...}}``.  Every metric block carries
        precision/recall/f1 plus raw tp/fp/fn counts and gold ``support``.
    """
    index_to_id: Dict[int, str] = {}
    for req_id, idx in id_to_index.items():
        # Duplicate indices mean an ambiguous mapping; keep the first and warn
        # rather than silently scoring against an arbitrary winner.
        if idx in index_to_id:
            logger.warning(
                "score_dependency_tree: node index %s maps to both %r and %r; "
                "keeping %r.",
                idx, index_to_id[idx], req_id, index_to_id[idx],
            )
            continue
        index_to_id[idx] = req_id

    # ---- predictions -> (source_id, target_id[, relation]) -----------------
    predicted_typed: Set[Triple] = set()
    predicted_detection: Set[Pair] = set()
    n_pred_unmappable = 0

    for edge in predicted_edges:
        src_idx = edge.get("source")
        tgt_idx = edge.get("target")
        if src_idx not in index_to_id or tgt_idx not in index_to_id:
            n_pred_unmappable += 1
            continue
        src_id = index_to_id[src_idx]
        tgt_id = index_to_id[tgt_idx]
        relation = str(edge.get("relation") or "").strip().lower()
        predicted_typed.add((src_id, tgt_id, relation))
        predicted_detection.add((src_id, tgt_id))

    # ---- gold --------------------------------------------------------------
    gold_typed: Set[Triple] = set()
    gold_detection: Set[Pair] = set()
    n_gold_dropped = 0

    for edge in gold_edges:
        src_id = str(edge.get("source_id") or "").strip()
        tgt_id = str(edge.get("target_id") or "").strip()
        relation = str(edge.get("relation") or "").strip().lower()
        if not src_id or not tgt_id or src_id == tgt_id:
            n_gold_dropped += 1
            continue
        # A gold edge naming a requirement absent from this dataset cannot be
        # predicted by construction; counting it as a false negative would
        # understate recall for reasons unrelated to the engine.
        if src_id not in id_to_index or tgt_id not in id_to_index:
            logger.warning(
                "score_dependency_tree: dropping gold edge %s -> %s "
                "(endpoint not present in dataset).", src_id, tgt_id,
            )
            n_gold_dropped += 1
            continue
        gold_typed.add((src_id, tgt_id, relation))
        gold_detection.add((src_id, tgt_id))

    # ---- per-relation-type -------------------------------------------------
    relations = {r for _, _, r in gold_typed} | {r for _, _, r in predicted_typed}
    per_relation: Dict[str, Any] = {}
    for relation in sorted(relations):
        per_relation[relation] = _score_sets(
            predicted={t for t in predicted_typed if t[2] == relation},
            gold={t for t in gold_typed if t[2] == relation},
        )

    return {
        "per_relation": per_relation,
        "detection": _score_sets(predicted_detection, gold_detection),
        "typed": _score_sets(predicted_typed, gold_typed),
        "undirected": _score_sets(
            _undirected(predicted_detection), _undirected(gold_detection)
        ),
        "counts": {
            "n_predicted_edges": len(predicted_edges),
            "n_predicted_scored": len(predicted_detection),
            "n_predicted_unmappable": n_pred_unmappable,
            "n_gold_edges": len(gold_edges),
            "n_gold_scored": len(gold_detection),
            "n_gold_dropped": n_gold_dropped,
        },
    }


def summarize_errors(
    predicted_edges: Sequence[Dict[str, Any]],
    gold_edges: Sequence[Dict[str, Any]],
    id_to_index: Dict[str, int],
    limit: int = 25,
) -> Dict[str, Any]:
    """Return concrete missed / spurious / mislabeled edges for error analysis.

    The aggregate scores say *how much* is wrong; this says *what*, which is what
    you actually need when tuning extraction rules or a similarity threshold.
    """
    index_to_id = {idx: rid for rid, idx in id_to_index.items()}

    pred: Dict[Pair, Dict[str, Any]] = {}
    for edge in predicted_edges:
        src_idx, tgt_idx = edge.get("source"), edge.get("target")
        if src_idx not in index_to_id or tgt_idx not in index_to_id:
            continue
        pred[(index_to_id[src_idx], index_to_id[tgt_idx])] = edge

    gold: Dict[Pair, str] = {}
    for edge in gold_edges:
        src_id = str(edge.get("source_id") or "").strip()
        tgt_id = str(edge.get("target_id") or "").strip()
        if src_id in id_to_index and tgt_id in id_to_index and src_id != tgt_id:
            gold[(src_id, tgt_id)] = str(edge.get("relation") or "").strip().lower()

    missed: List[Dict[str, Any]] = [
        {"source_id": s, "target_id": t, "gold_relation": r}
        for (s, t), r in sorted(gold.items())
        if (s, t) not in pred
    ]
    spurious: List[Dict[str, Any]] = [
        {
            "source_id": s,
            "target_id": t,
            "predicted_relation": e.get("relation"),
            "weight": e.get("weight"),
            "rationale": e.get("rationale"),
        }
        for (s, t), e in sorted(pred.items())
        if (s, t) not in gold
    ]
    mislabeled: List[Dict[str, Any]] = [
        {
            "source_id": s,
            "target_id": t,
            "gold_relation": gold[(s, t)],
            "predicted_relation": e.get("relation"),
            "rationale": e.get("rationale"),
        }
        for (s, t), e in sorted(pred.items())
        if (s, t) in gold
        and str(e.get("relation") or "").strip().lower() != gold[(s, t)]
    ]

    return {
        "missed": missed[:limit],
        "spurious": spurious[:limit],
        "mislabeled": mislabeled[:limit],
        "totals": {
            "missed": len(missed),
            "spurious": len(spurious),
            "mislabeled": len(mislabeled),
        },
    }
