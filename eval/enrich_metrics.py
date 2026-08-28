"""Enrichment quality aggregates from GET /api/enrich/results."""

from __future__ import annotations

from typing import Any, Dict, List


def evaluate_enrichment_metrics(
    enrichment_results: List[Dict[str, Any]],
) -> Dict[str, Any]:
    """Aggregate enrichment quality report metrics across all requirements."""
    total = len(enrichment_results)
    if total == 0:
        return {
            "total_enrichment_rows": 0,
            "parse_success_rate": 0.0,
        }

    succeeded_rows = [
        r for r in enrichment_results
        if r.get("status") in {"succeeded", "success"}
    ]
    succeeded_count = len(succeeded_rows)
    parse_success_rate = round(succeeded_count / total, 4)

    hallucination_scores: List[float] = []
    confidence_scores: List[float] = []
    lexical_overlaps: List[float] = []
    domain_coverages: List[float] = []
    warning_count = 0

    for r in succeeded_rows:
        qr = r.get("quality_report") or {}
        
        h_score = qr.get("hallucination_risk_score")
        if h_score is not None:
            hallucination_scores.append(float(h_score))

        c_score = qr.get("adjusted_confidence_score")
        if c_score is None:
            c_score = r.get("confidence")
        if c_score is not None:
            confidence_scores.append(float(c_score))

        lex_score = qr.get("lexical_overlap")
        if lex_score is not None:
            lexical_overlaps.append(float(lex_score))

        cov_score = qr.get("domain_term_coverage")
        if cov_score is not None:
            domain_coverages.append(float(cov_score))

        warnings = r.get("warnings") or qr.get("warnings") or []
        if warnings:
            warning_count += 1

    def _mean(values: List[float]) -> float:
        return round(sum(values) / len(values), 4) if values else 0.0

    return {
        "total_enrichment_rows": total,
        "succeeded_enrichment_rows": succeeded_count,
        "parse_success_rate": parse_success_rate,
        "mean_hallucination_risk_score": _mean(hallucination_scores),
        "mean_adjusted_confidence_score": _mean(confidence_scores),
        "mean_lexical_overlap": _mean(lexical_overlaps),
        "mean_domain_term_coverage": _mean(domain_coverages),
        "warning_rate": round(warning_count / succeeded_count, 4) if succeeded_count else 0.0,
    }
