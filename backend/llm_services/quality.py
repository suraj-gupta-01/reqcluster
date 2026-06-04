from __future__ import annotations

import re
import unicodedata
from typing import Any, Mapping, Sequence

from .prompts import normalize_domain_terms, normalize_plain_text
from .vocabulary import REQUIREMENT_STOPWORDS


_TOKEN_RE = re.compile(r"[a-z0-9]+(?:[-.][a-z0-9]+)*")
_NUMBER_RE = re.compile(r"(?<![a-z0-9])\d+(?:\.\d+)?(?:%|ms|s|sec|seconds|m|kg|v)?(?![a-z0-9])")
_SPACE_RE = re.compile(r"\s+")


def _round(value: float | None) -> float | None:
    if value is None:
        return None
    return round(float(value), 6)


def _clamp(value: float, low: float = 0.0, high: float = 1.0) -> float:
    return min(max(float(value), low), high)


def _tokens(text: object) -> list[str]:
    normalized = unicodedata.normalize("NFKC", str(text or "")).lower()
    normalized = _SPACE_RE.sub(" ", normalized)
    return _TOKEN_RE.findall(normalized)


def _meaning_tokens(text: object) -> list[str]:
    return [
        token
        for token in _tokens(text)
        if token not in REQUIREMENT_STOPWORDS and (len(token) >= 3 or token.isdigit())
    ]


def _numbers(text: object) -> set[str]:
    return {match.group(0).lower() for match in _NUMBER_RE.finditer(str(text or "").lower())}


def _ordered_missing(original_tokens: Sequence[str], enriched_tokens: set[str]) -> list[str]:
    missing: list[str] = []
    seen: set[str] = set()
    for token in original_tokens:
        if token in enriched_tokens or token in seen:
            continue
        seen.add(token)
        missing.append(token)
        if len(missing) >= 25:
            break
    return missing


def evaluate_expansion_quality(
    original_text: str,
    enriched_text: str,
    domain_terms: Sequence[str] | None = None,
    provider_confidence: float | None = None,
) -> dict:
    """Return conservative semantic augmentation quality metrics and warnings."""
    original = normalize_plain_text(original_text)
    enriched = normalize_plain_text(enriched_text)
    warnings: list[str] = []

    original_tokens = _tokens(original)
    enriched_tokens = _tokens(enriched)
    original_meaning = _meaning_tokens(original)
    enriched_meaning = _meaning_tokens(enriched)
    enriched_meaning_set = set(enriched_meaning)

    if not enriched:
        warnings.append("Expanded text is empty.")

    length_ratio = None
    if original_tokens:
        length_ratio = len(enriched_tokens) / max(len(original_tokens), 1)
        if length_ratio < 0.75:
            warnings.append("Expanded text is too short compared with the original requirement.")
        if length_ratio > 4.0:
            warnings.append("Expanded text is too long compared with the original requirement.")

    original_set = set(original_meaning)
    enriched_set = set(enriched_meaning)
    lexical_overlap = 1.0
    if original_set:
        lexical_overlap = len(original_set & enriched_set) / len(original_set)
        if lexical_overlap < 0.45:
            warnings.append("Lexical overlap with the original requirement is very low.")

    terms = normalize_domain_terms(domain_terms)
    covered_terms = []
    enriched_lower = enriched.lower()
    for term in terms:
        if term and term in enriched_lower:
            covered_terms.append(term)
    domain_term_coverage = None if not terms else len(covered_terms) / len(terms)

    missing_critical = _ordered_missing(original_meaning, enriched_meaning_set)
    if original_meaning and len(missing_critical) / max(len(set(original_meaning)), 1) > 0.4:
        warnings.append("Expanded text omits many critical tokens from the original requirement.")

    original_numbers = _numbers(original)
    enriched_numbers = _numbers(enriched)
    invented_numbers = sorted(enriched_numbers - original_numbers)
    if invented_numbers:
        warnings.append("Expanded text introduces numeric values not present in the original requirement.")

    original_obligations = set(_tokens(original)) & {"may", "should", "shall", "must"}
    enriched_obligations = set(_tokens(enriched)) & {"may", "should", "shall", "must"}
    changed_obligation = False
    if "may" in original_obligations and ("shall" in enriched_obligations or "must" in enriched_obligations):
        changed_obligation = True
    if "should" in original_obligations and ("shall" in enriched_obligations or "must" in enriched_obligations):
        changed_obligation = True
    if changed_obligation:
        warnings.append("Expanded text strengthens obligation language from may/should to shall/must.")

    risk = 0.0
    if invented_numbers:
        risk += 0.25
    if changed_obligation:
        risk += 0.25
    if lexical_overlap < 0.45:
        risk += 0.20
    if length_ratio is not None and (length_ratio < 0.75 or length_ratio > 4.0):
        risk += 0.15
    if original_meaning and len(missing_critical) / max(len(set(original_meaning)), 1) > 0.4:
        risk += 0.15
    risk = _clamp(risk)

    base_confidence = 0.5 if provider_confidence is None else _clamp(float(provider_confidence))
    adjusted_confidence = _clamp(base_confidence * (1.0 - risk))

    return {
        "length_ratio": _round(length_ratio),
        "lexical_overlap": _round(lexical_overlap),
        "domain_term_coverage": _round(domain_term_coverage),
        "missing_original_critical_tokens": missing_critical,
        "invented_numeric_values": invented_numbers,
        "changed_obligation_strength": bool(changed_obligation),
        "hallucination_risk_score": _round(risk),
        "adjusted_confidence_score": _round(adjusted_confidence),
        "warnings": warnings,
    }


def _value(obj: Any, key: str, default: Any = None) -> Any:
    if isinstance(obj, Mapping):
        return obj.get(key, default)
    return getattr(obj, key, default)


def summarize_quality_report(per_requirement: Sequence[dict]) -> dict:
    successful = [row for row in per_requirement if row.get("status") == "succeeded"]
    risks = [
        float(row["hallucination_risk_score"])
        for row in successful
        if row.get("hallucination_risk_score") is not None
    ]
    adjusted = [
        float(row["adjusted_confidence_score"])
        for row in successful
        if row.get("adjusted_confidence_score") is not None
    ]

    warnings: list[str] = []
    seen: set[str] = set()
    for row in per_requirement:
        for warning in row.get("warnings", []):
            if warning not in seen:
                seen.add(warning)
                warnings.append(warning)

    return {
        "total": int(len(per_requirement)),
        "succeeded": int(len(successful)),
        "failed": int(len(per_requirement) - len(successful)),
        "aggregate": {
            "mean_hallucination_risk_score": _round(sum(risks) / len(risks)) if risks else None,
            "mean_adjusted_confidence_score": _round(sum(adjusted) / len(adjusted)) if adjusted else None,
        },
        "per_requirement": list(per_requirement),
        "warnings": warnings,
    }


def build_quality_report(
    original_texts: Sequence[str],
    results_by_index: Sequence[Any | None],
    domain_vocabulary: Sequence[str] | None = None,
) -> dict:
    per_requirement: list[dict] = []
    for idx, original in enumerate(original_texts):
        result = results_by_index[idx] if idx < len(results_by_index) else None
        if result is None:
            per_requirement.append(
                {
                    "index": int(idx),
                    "requirement_id": None,
                    "status": "failed",
                    "warnings": ["Requirement enrichment failed."],
                }
            )
            continue

        result_terms = _value(result, "domain_terms", None) or domain_vocabulary
        metrics = evaluate_expansion_quality(
            original,
            _value(result, "expanded_text", ""),
            domain_terms=result_terms,
            provider_confidence=_value(result, "confidence", None),
        )
        per_requirement.append(
            {
                "index": int(idx),
                "requirement_id": _value(result, "requirement_id", None),
                "status": "succeeded",
                **metrics,
            }
        )

    return summarize_quality_report(per_requirement)
