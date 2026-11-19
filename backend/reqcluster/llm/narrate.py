"""Cluster narratives and dependency-edge explanations (LLM with template fallback)."""
from dataclasses import replace
from typing import Any, Literal

from ..domain import ProviderName
from .gateway import LLMGateway, LLMResult
from .prompts import Prompt, normalize_plain_text

NARRATIVE_VERSION = "reqcluster-cluster-narrative-v2"
EDGE_PROMPT_VERSION = "reqcluster-edge-explanation-v2"
MAX_NARRATIVE_CHARS = 700
MAX_EXPLANATION_CHARS = 1200

_RELATION_PHRASES = {
    "data": "passes data to",
    "sequential": "must happen before",
    "hierarchical": "is a broader requirement than",
    "reference": "is explicitly cited by",
}

_NARRATIVE_TASKS = {
    "merge": "Two requirement clusters may be merged. Explain in 2-3 sentences whether merging "
             "is advisable, citing the metrics.",
    "split": "A requirement cluster may be split into two. Explain in 2-3 sentences whether "
             "splitting is advisable, citing the metrics.",
    "summary": "Write a 2-3 sentence summary of this engineering requirement cluster for a "
               "review document.",
}


# -------------------------------------------------------------------- templates

def merge_rationale(cluster_a_label, cluster_b_label, centroid_similarity, silhouette_delta,
                    coherence_a, coherence_b) -> str:
    parts = [f'Clusters "{cluster_a_label}" and "{cluster_b_label}" share high centroid '
             f"similarity ({centroid_similarity:.3f})."]
    if silhouette_delta > 0:
        parts.append(f"Merging improves the overall silhouette score by {silhouette_delta:.4f}, "
                     "indicating better-defined cluster boundaries.")
    elif silhouette_delta == 0:
        parts.append("Merging has a neutral effect on the silhouette score.")
    else:
        parts.append(f"Merging decreases the silhouette score by {abs(silhouette_delta):.4f}. "
                     "Review the clusters carefully before merging.")
    if (coherence_a + coherence_b) / 2 >= 0.7:
        parts.append("Both clusters have high internal coherence, suggesting they cover "
                     "closely related functional domains.")
    else:
        parts.append("One or both clusters have moderate coherence. Merging may improve "
                     "or reduce overall coherence depending on the requirements.")
    return " ".join(parts)


def split_rationale(cluster_label, bimodality_score, bic_improvement, spread_score,
                    sub_cluster_sizes) -> str:
    parts = [f'Cluster "{cluster_label}" shows a bimodal distribution '
             f"(bimodality score: {bimodality_score:.3f}, BIC improvement: {bic_improvement:.1f})."]
    if len(sub_cluster_sizes) == 2:
        parts.append(f"The two identified sub-groups contain {sub_cluster_sizes[0]} and "
                     f"{sub_cluster_sizes[1]} requirements respectively.")
    if spread_score >= 0.3:
        parts.append(f"The cluster has high internal spread ({spread_score:.3f}), "
                     "supporting the case for splitting.")
    parts.append("Splitting would create more focused clusters with tighter semantic coherence.")
    return " ".join(parts)


def cluster_summary(cluster_label, keywords, representative_texts) -> str:
    covers = (f"covers requirements related to {', '.join(keywords[:5])}"
              if keywords else "contains related requirements")
    parts = [f'Cluster "{cluster_label}" {covers}.']
    if representative_texts:
        parts.append("Representative requirements include: "
                     + "; ".join(f'"{t[:120]}"' for t in representative_texts[:3]) + ".")
    return " ".join(parts)


_TEMPLATES = {"merge": merge_rationale, "split": split_rationale, "summary": cluster_summary}


def cluster_narrative(gateway: LLMGateway, kind: Literal["merge", "split", "summary"],
                      provider: ProviderName | None = None, **params: Any) -> LLMResult:
    """Merge/split rationale or cluster summary. `params` are the template's arguments."""
    prompt = Prompt("narrative_" + kind, NARRATIVE_VERSION, _NARRATIVE_TASKS[kind] + " Reply with prose only.",
                    dict(params), max_output_chars=MAX_NARRATIVE_CHARS)
    return gateway.complete(prompt, lambda: _TEMPLATES[kind](**params), provider)


# ------------------------------------------------------------------------ edges

def _clean(value: object, max_chars: int = 200) -> str:
    return normalize_plain_text(value, max_chars)


def parse_explanation_response(raw: object) -> str:
    """Coerce a model reply (string, envelope dict/list, fenced block) to one prose paragraph."""
    if raw is None:
        raise ValueError("Edge explanation response was empty.")
    text: object = raw
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
                raise ValueError("Edge explanation response was empty.")
            text = text[0]
        else:
            break
    stripped = str(text).strip()
    if stripped.startswith("```"):
        stripped = "\n".join(ln for ln in stripped.splitlines() if not ln.strip().startswith("```"))
    cleaned = _clean(stripped, MAX_EXPLANATION_CHARS)
    if not cleaned:
        raise ValueError("Edge explanation response was empty.")
    return cleaned


def mock_explanation(source_id: str, target_id: str, relation: str, evidence: dict | None,
                     confidence_tier: str = "", rationale: str = "") -> str:
    """Deterministic explanation templated from the engine's evidence."""
    evidence = evidence if isinstance(evidence, dict) else {}
    src = _clean(source_id, 80) or "the source requirement"
    tgt = _clean(target_id, 80) or "the target requirement"
    phrase = _RELATION_PHRASES.get(_clean(relation, 40).lower(), "is related to")
    parts = [f"{src} {phrase} {tgt}."]
    which = _clean(evidence.get("pass"), 60)
    sim = evidence.get("artifact_similarity")

    if which == "pass_0_artifact_match":
        producer = _clean(evidence.get("producer_artifact_name")) or _clean(evidence.get("artifact"))
        consumer = _clean(evidence.get("consumer_artifact_name")) or producer
        if evidence.get("synonym_match") and producer and consumer:
            suffix = f" (similarity {float(sim):.2f})" if isinstance(sim, (int, float)) else ""
            parts.append(f"{src} produces '{producer}', which was matched as a synonym of "
                         f"'{consumer}' consumed by {tgt}{suffix}.")
        elif producer:
            parts.append(f"Both requirements name the artifact '{producer}': {src} "
                         f"produces it and {tgt} consumes it.")
    elif which == "pass_1_explicit_reference":
        matched = _clean(evidence.get("matched_text"), 60)
        if matched:
            parts.append(f"The text of {tgt} explicitly cites '{matched}', so {src} is a "
                         "stated prerequisite rather than an inferred one.")
        cue = _clean(evidence.get("hierarchical_cue"), 60)
        if cue:
            parts.append(f"The citation uses the phrase '{cue}', which indicates a "
                         "hierarchical relationship.")
    elif which == "pass_2_sequential":
        cue = _clean(evidence.get("cue"), 60)
        cos = evidence.get("cosine_similarity")
        token = _clean(evidence.get("primary_shared_token"), 60)
        detail = f"{tgt} is gated by the cue '{cue}'" if cue else f"{tgt} is time-gated"
        if token:
            detail += f", and both requirements mention '{token}'"
        if isinstance(cos, (int, float)):
            detail += f" (similarity {float(cos):.2f})"
        parts.append(detail + ".")
    elif which == "pass_2_hierarchical":
        cos = evidence.get("cosine_similarity")
        detail = f"Both sit in the same cluster and {src} is stated more generally than {tgt}"
        if isinstance(cos, (int, float)):
            detail += f" (similarity {float(cos):.2f})"
        parts.append(detail + ".")
    elif rationale:
        parts.append(_clean(rationale, 400))

    tier = _clean(confidence_tier, 20)
    if tier:
        parts.append(f"Confidence for this link is {tier.lower()}.")
    return _clean(" ".join(parts), MAX_EXPLANATION_CHARS)


_EDGE_SYSTEM = (
    "Explain, in one short plain-language paragraph, why a dependency was inferred between two "
    "engineering requirements. Reply with prose only: no Markdown, bullets, headings or code "
    "fences. Ground every claim in the structured evidence provided. Do not invent signals, "
    "interfaces, thresholds, standards, timings, or subsystems that are not present in the "
    "evidence or the requirement texts. Do not propose a different relationship. If the evidence "
    "is weak, say so plainly. Write at most 4 sentences. The source is a prerequisite of the target."
)


def explain_edge(gateway: LLMGateway, source_text: str, target_text: str, relation: str,
                 evidence: dict | None = None, *, source_id: str = "", target_id: str = "",
                 confidence_tier: str = "", rationale: str = "",
                 provider: ProviderName | None = None) -> LLMResult:
    prompt = Prompt(
        "explain_edge", EDGE_PROMPT_VERSION, _EDGE_SYSTEM,
        {"source_id": source_id, "target_id": target_id, "relation": relation,
         "confidence_tier": confidence_tier, "evidence": evidence or {},
         "source_text": source_text, "target_text": target_text},
        max_output_chars=MAX_EXPLANATION_CHARS)
    res = gateway.complete(
        prompt, lambda: mock_explanation(source_id, target_id, relation, evidence,
                                         confidence_tier, rationale), provider)
    if res.source == "fallback":
        return res
    try:
        return replace(res, output=parse_explanation_response(res.output))
    except ValueError:
        return LLMResult(mock_explanation(source_id, target_id, relation, evidence,
                                          confidence_tier, rationale),
                         "fallback", res.provider, "empty explanation")
