"""Plain-language justification for a single dependency edge (DP5 Phase 3.2).

Called only when a user clicks a specific edge, never across the pipeline. That
bound is what makes an LLM affordable here at all: total call volume tracks how
many edges a human actually inspects, not the size of the dataset.

The ``mock`` provider templates the structured evidence directly, so the feature
works fully offline by default with zero hallucination risk — the same posture
as enrichment. Only the prose version needs a configured model.
"""

from __future__ import annotations

import logging
from typing import Any, Dict, Optional

from .prompts import (
    EXPLAIN_EDGE_PROMPT_VERSION,
    build_edge_explanation_prompt,
    normalize_plain_text,
)
from .providers import (
    LLMProviderError,
    ProviderResponseError,
    generate_completion,
)

logger = logging.getLogger(__name__)

MAX_EXPLANATION_CHARS = 1200

# Relation-specific phrasing for the deterministic templates.
_RELATION_PHRASES = {
    "data": "passes data to",
    "sequential": "must happen before",
    "hierarchical": "is a broader requirement than",
    "reference": "is explicitly cited by",
}

_MOCK_PROVIDERS = {"mock", "offline", "test", ""}


def _clean(value: object, max_chars: int = 200) -> str:
    return normalize_plain_text(value, max_chars=max_chars)


def parse_explanation_response(raw_response: object) -> str:
    """Coerce a model's reply into a single clean prose paragraph.

    Small local models return odd shapes here for the same reasons they do in
    enrichment — a bare string, a JSON-ish object, a fenced block, or the answer
    wrapped in a chat envelope. This applies the same leniency-then-bound
    approach used by ``parse_expansion_response`` rather than rejecting outright,
    because the payload is inert prose destined for display, not a schema the
    rest of the pipeline depends on.
    """
    if raw_response is None:
        raise ProviderResponseError("Edge explanation response was empty.")

    text: object = raw_response

    # Unwrap the common envelope shapes before touching the text.
    for _ in range(3):
        if isinstance(text, dict):
            for key in ("explanation", "text", "content", "answer", "output", "result"):
                if key in text:
                    text = text[key]
                    break
            else:
                break
        elif isinstance(text, (list, tuple)):
            if not text:
                raise ProviderResponseError("Edge explanation response was empty.")
            text = text[0]
        else:
            break

    if not isinstance(text, str):
        text = str(text)

    # Strip Markdown fences rather than rejecting them: a fenced paragraph is
    # still a usable paragraph, and refusing it would fail a click for a purely
    # cosmetic reason.
    stripped = text.strip()
    if stripped.startswith("```"):
        lines = [
            line for line in stripped.splitlines()
            if not line.strip().startswith("```")
        ]
        stripped = "\n".join(lines)

    cleaned = _clean(stripped, max_chars=MAX_EXPLANATION_CHARS)
    if not cleaned:
        raise ProviderResponseError("Edge explanation response was empty.")
    return cleaned


def build_mock_explanation(
    source_id: str,
    target_id: str,
    relation: str,
    evidence: Optional[Dict[str, Any]],
    confidence_tier: str = "",
    rationale: str = "",
) -> str:
    """Deterministic explanation templated straight from the evidence.

    Same text for the same edge every time, and it states only what the engine
    actually computed.
    """
    evidence = evidence if isinstance(evidence, dict) else {}
    src = _clean(source_id, 80) or "the source requirement"
    tgt = _clean(target_id, 80) or "the target requirement"
    relation_clean = _clean(relation, 40).lower()
    phrase = _RELATION_PHRASES.get(relation_clean, "is related to")

    parts = [f"{src} {phrase} {tgt}."]
    which_pass = _clean(evidence.get("pass"), 60)

    if which_pass == "pass_0_artifact_match":
        producer = _clean(evidence.get("producer_artifact_name")) or _clean(
            evidence.get("artifact")
        )
        consumer = _clean(evidence.get("consumer_artifact_name")) or producer
        if evidence.get("synonym_match") and producer and consumer:
            similarity = evidence.get("artifact_similarity")
            suffix = (
                f" (similarity {float(similarity):.2f})"
                if isinstance(similarity, (int, float))
                else ""
            )
            parts.append(
                f"{src} produces '{producer}', which was matched as a synonym of "
                f"'{consumer}' consumed by {tgt}{suffix}."
            )
        elif producer:
            parts.append(
                f"Both requirements name the artifact '{producer}': {src} "
                f"produces it and {tgt} consumes it."
            )
    elif which_pass == "pass_1_explicit_reference":
        matched = _clean(evidence.get("matched_text"), 60)
        if matched:
            parts.append(
                f"The text of {tgt} explicitly cites '{matched}', so {src} is a "
                f"stated prerequisite rather than an inferred one."
            )
        cue = _clean(evidence.get("hierarchical_cue"), 60)
        if cue:
            parts.append(
                f"The citation uses the phrase '{cue}', which indicates a "
                f"hierarchical relationship."
            )
    elif which_pass == "pass_2_sequential":
        cue = _clean(evidence.get("cue"), 60)
        similarity = evidence.get("cosine_similarity")
        token = _clean(evidence.get("primary_shared_token"), 60)
        detail = f"{tgt} is gated by the cue '{cue}'" if cue else f"{tgt} is time-gated"
        if token:
            detail += f", and both requirements mention '{token}'"
        if isinstance(similarity, (int, float)):
            detail += f" (similarity {float(similarity):.2f})"
        parts.append(detail + ".")
    elif which_pass == "pass_2_hierarchical":
        similarity = evidence.get("cosine_similarity")
        detail = (
            f"Both sit in the same cluster and {src} is stated more generally "
            f"than {tgt}"
        )
        if isinstance(similarity, (int, float)):
            detail += f" (similarity {float(similarity):.2f})"
        parts.append(detail + ".")
    elif rationale:
        parts.append(_clean(rationale, 400))

    tier = _clean(confidence_tier, 20)
    if tier:
        parts.append(f"Confidence for this link is {tier.lower()}.")

    return _clean(" ".join(parts), max_chars=MAX_EXPLANATION_CHARS)


def generate_edge_explanation(
    source_text: str,
    target_text: str,
    relation: str,
    evidence: Optional[Dict[str, Any]] = None,
    *,
    source_id: str = "",
    target_id: str = "",
    confidence_tier: str = "",
    rationale: str = "",
    provider_name: str = "mock",
    timeout_seconds: float = 30.0,
) -> Dict[str, Any]:
    """Explain one edge, falling back to the deterministic template on any error.

    Returns ``{"explanation_text", "provider", "prompt_version", "fallback",
    "warning"}``. A provider failure degrades to the mock template rather than
    surfacing an error: the user clicked an edge and is owed an answer, and the
    template is always available.
    """
    selected = _clean(provider_name, 80).lower()
    mock_text = build_mock_explanation(
        source_id, target_id, relation, evidence, confidence_tier, rationale
    )

    if selected in _MOCK_PROVIDERS:
        return {
            "explanation_text": mock_text,
            "provider": "mock",
            "prompt_version": EXPLAIN_EDGE_PROMPT_VERSION,
            "fallback": False,
            "warning": None,
        }

    prompt = build_edge_explanation_prompt(
        source_text=source_text,
        target_text=target_text,
        relation=relation,
        evidence=evidence,
        source_id=source_id,
        target_id=target_id,
        confidence_tier=confidence_tier,
    )

    try:
        raw = generate_completion(
            prompt,
            provider_name=selected,
            timeout_seconds=timeout_seconds,
            max_chars=MAX_EXPLANATION_CHARS,
        )
        return {
            "explanation_text": parse_explanation_response(raw),
            "provider": selected,
            "prompt_version": EXPLAIN_EDGE_PROMPT_VERSION,
            "fallback": False,
            "warning": None,
        }
    except (LLMProviderError, ValueError, TypeError) as exc:
        logger.warning(
            "Edge explanation via provider %r failed (%s); using the "
            "deterministic template.", selected, exc,
        )
        return {
            "explanation_text": mock_text,
            "provider": "mock",
            "prompt_version": EXPLAIN_EDGE_PROMPT_VERSION,
            "fallback": True,
            "warning": f"{selected} provider unavailable: {exc}",
        }
