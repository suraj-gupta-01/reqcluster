from __future__ import annotations

import re
import unicodedata
from typing import Iterable, Sequence


PROMPT_VERSION = "reqcluster-requirement-expansion-v1"
EXPLAIN_EDGE_PROMPT_VERSION = "reqcluster-edge-explanation-v1"
MAX_REQUIREMENT_CHARS = 12_000
MAX_VOCAB_TERMS = 50
MAX_TERM_CHARS = 80

_CONTROL_CHARS_RE = re.compile(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]")
_WHITESPACE_RE = re.compile(r"\s+")


def normalize_plain_text(value: object, max_chars: int = MAX_REQUIREMENT_CHARS) -> str:
    """Normalize inert text for prompts, cache keys, and model response fields."""
    if value is None:
        return ""
    text = unicodedata.normalize("NFC", str(value))
    text = _CONTROL_CHARS_RE.sub(" ", text)
    text = _WHITESPACE_RE.sub(" ", text).strip()
    if max_chars > 0:
        return text[:max_chars]
    return text


def normalize_domain_terms(
    terms: Iterable[object] | None,
    max_terms: int = MAX_VOCAB_TERMS,
) -> list[str]:
    normalized: list[str] = []
    seen: set[str] = set()
    if terms is None:
        return normalized

    for term in terms:
        cleaned = normalize_plain_text(term, max_chars=MAX_TERM_CHARS).lower()
        cleaned = cleaned.strip(" ,.;:()[]{}")
        if not cleaned:
            continue
        key = cleaned.casefold()
        if key in seen:
            continue
        seen.add(key)
        normalized.append(cleaned)
        if len(normalized) >= max_terms:
            break
    return normalized


def build_requirement_expansion_prompt(
    requirement_text: str,
    domain_vocabulary: Sequence[str] | None = None,
) -> str:
    """
    Build the strict JSON prompt used by non-mock providers.

    The prompt is intentionally explicit about meaning preservation and the
    allowed response schema. It asks for domain context, not new requirements.
    """
    original = normalize_plain_text(requirement_text)
    vocabulary = normalize_domain_terms(domain_vocabulary)
    vocabulary_text = ", ".join(vocabulary) if vocabulary else "(none provided)"

    return (
        f"Prompt version: {PROMPT_VERSION}\n"
        "Task: Expand one cleaned software or systems requirement for semantic embedding.\n"
        "Return strict JSON only. Do not return Markdown, comments, prose outside JSON, or code fences.\n"
        "Preserve the original requirement meaning exactly.\n"
        "Do not add new obligations, constraints, shall-statements, must-statements, or should-statements.\n"
        "Do not invent numeric thresholds, quantities, timings, standards, subsystems, interfaces, "
        "signals, APIs, verification criteria, test methods, or acceptance criteria.\n"
        "Use only information explicitly present in the requirement, plus generic wording that clarifies "
        "semantic context without changing meaning.\n"
        "Identify the functional intent, explicitly mentioned components, and useful domain vocabulary.\n"
        "If uncertain, keep the expansion conservative, lower confidence, and add a warning.\n"
        "The JSON object must have exactly these keys with these value types:\n"
        "{\n"
        '  "expanded_text": "string",\n'
        '  "domain_terms": ["string"],\n'
        '  "functional_intent": "string",\n'
        '  "mentioned_components": ["string"],\n'
        '  "assumptions": ["string"],\n'
        '  "confidence": 0.0,\n'
        '  "warnings": ["string"]\n'
        "}\n"
        "Domain vocabulary candidates from this batch:\n"
        f"{vocabulary_text}\n"
        "Original requirement:\n"
        f"{original}\n"
    )


# ---------------------------------------------------------------------------
# Dependency-edge explanation (DP5 Phase 3)
# ---------------------------------------------------------------------------

MAX_EVIDENCE_ITEMS = 12
MAX_EVIDENCE_VALUE_CHARS = 200


def format_edge_evidence(evidence: object) -> str:
    """Render an edge's structured evidence as flat ``key: value`` lines.

    Only inert scalars and short lists are emitted, each normalized through
    :func:`normalize_plain_text`. Evidence comes from the deterministic engine
    rather than from user input, but it still reaches a prompt, so it is bounded
    and stripped here instead of being interpolated raw.
    """
    if not isinstance(evidence, dict) or not evidence:
        return "(no structured evidence recorded)"

    lines: list[str] = []
    for key in sorted(evidence):
        if len(lines) >= MAX_EVIDENCE_ITEMS:
            break
        value = evidence[key]
        if isinstance(value, bool):
            rendered = "true" if value else "false"
        elif isinstance(value, (int, float)):
            rendered = f"{value}"
        elif isinstance(value, (list, tuple)):
            parts = [
                normalize_plain_text(v, max_chars=MAX_TERM_CHARS)
                for v in list(value)[:MAX_VOCAB_TERMS]
            ]
            rendered = ", ".join(p for p in parts if p)
        elif value is None:
            continue
        else:
            rendered = normalize_plain_text(value, max_chars=MAX_EVIDENCE_VALUE_CHARS)
        if not rendered:
            continue
        clean_key = normalize_plain_text(key, max_chars=60)
        lines.append(f"- {clean_key}: {rendered}")

    return "\n".join(lines) if lines else "(no structured evidence recorded)"


def build_edge_explanation_prompt(
    source_text: str,
    target_text: str,
    relation: str,
    evidence: dict | None = None,
    source_id: str = "",
    target_id: str = "",
    confidence_tier: str = "",
) -> str:
    """Build the prompt asking for a plain-language justification of one edge.

    The model is grounded strictly in the structured evidence the engine already
    computed rather than asked to re-derive the relationship from the two texts.
    That keeps the call cheap and makes hallucination a much smaller risk:
    everything it is permitted to assert is already in the prompt.
    """
    source = normalize_plain_text(source_text)
    target = normalize_plain_text(target_text)
    relation_clean = normalize_plain_text(relation, max_chars=40) or "unspecified"
    tier = normalize_plain_text(confidence_tier, max_chars=20) or "unspecified"
    src_id = normalize_plain_text(source_id, max_chars=80) or "the source requirement"
    tgt_id = normalize_plain_text(target_id, max_chars=80) or "the target requirement"

    return (
        f"Prompt version: {EXPLAIN_EDGE_PROMPT_VERSION}\n"
        "Task: Explain, in one short plain-language paragraph, why a dependency "
        "was inferred between two engineering requirements.\n"
        "Reply with prose only. Do not use Markdown, bullet points, headings, or "
        "code fences.\n"
        "Ground every claim in the structured evidence provided below.\n"
        "Do not invent signals, interfaces, thresholds, standards, timings, or "
        "subsystems that are not present in the evidence or the requirement texts.\n"
        "Do not re-analyse the requirements to propose a different relationship.\n"
        "If the evidence is weak, say so plainly instead of overstating the link.\n"
        "Write at most 4 sentences.\n"
        f"Direction: {src_id} is a prerequisite of {tgt_id}.\n"
        f"Inferred relation type: {relation_clean}\n"
        f"Confidence tier: {tier}\n"
        "Structured evidence produced by the inference engine:\n"
        f"{format_edge_evidence(evidence)}\n"
        f"Source requirement ({src_id}):\n{source}\n"
        f"Target requirement ({tgt_id}):\n{target}\n"
    )
