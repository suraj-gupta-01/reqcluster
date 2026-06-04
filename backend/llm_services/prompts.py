from __future__ import annotations

import re
import unicodedata
from typing import Iterable, Sequence


PROMPT_VERSION = "reqcluster-requirement-expansion-v1"
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
