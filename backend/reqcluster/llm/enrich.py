"""Requirement enrichment: LLM expansion with a deterministic fallback, plus
the parse caps, quality checks and vocabulary extraction it relies on."""
import math
import re
import unicodedata
from collections import Counter
from collections.abc import Iterable, Sequence
from dataclasses import dataclass, field
from typing import Literal

from pydantic import BaseModel, Field

from ..domain import ProviderName
from .gateway import LLMGateway
from .prompts import Prompt, normalize_plain_text

PROMPT_VERSION = "reqcluster-requirement-expansion-v2"
MAX_EXPANDED_TEXT_CHARS = 12_000
MAX_SHORT_STRING_CHARS = 1_000
MAX_LIST_ITEM_CHARS = 160
MAX_DOMAIN_TERMS = 50
MAX_COMPONENTS = 50
MAX_ASSUMPTIONS = 20
MAX_WARNINGS = 30
MAX_TERM_CHARS = 80

SYSTEM = (
    "Expand one cleaned software or systems requirement for semantic embedding. "
    "Return strict JSON only, matching the given schema. "
    "Preserve the original requirement meaning exactly. "
    "Do not add new obligations, constraints, shall/must/should statements. "
    "Do not invent numeric thresholds, quantities, timings, standards, subsystems, interfaces, "
    "signals, APIs, verification criteria, test methods, or acceptance criteria. "
    "Use only information explicitly present in the requirement, plus generic wording that "
    "clarifies semantic context without changing meaning. "
    "Identify the functional intent, explicitly mentioned components, and useful domain vocabulary. "
    "If uncertain, keep the expansion conservative, lower confidence, and add a warning."
)


class ExpansionOut(BaseModel):
    expanded_text: str = Field(min_length=1)
    domain_terms: list[str]
    functional_intent: str
    mentioned_components: list[str]
    assumptions: list[str]
    confidence: float
    warnings: list[str]


@dataclass
class Expansion:
    requirement_id: int
    expanded_text: str = ""
    domain_terms: list[str] = field(default_factory=list)
    functional_intent: str = ""
    mentioned_components: list[str] = field(default_factory=list)
    assumptions: list[str] = field(default_factory=list)
    confidence: float = 0.0
    warnings: list[str] = field(default_factory=list)
    status: Literal["succeeded", "failed"] = "succeeded"
    source: str = "fallback"
    error: str | None = None


# ------------------------------------------------------------------ parse caps

def normalize_domain_terms(terms: Iterable[object] | None, max_terms: int = 50) -> list[str]:
    out: list[str] = []
    seen: set[str] = set()
    for term in terms or ():
        cleaned = normalize_plain_text(term, MAX_TERM_CHARS).lower().strip(" ,.;:()[]{}")
        if cleaned and cleaned not in seen:
            seen.add(cleaned)
            out.append(cleaned)
            if len(out) >= max_terms:
                break
    return out


def _dedupe(values: Sequence[str], max_items: int | None = None) -> list[str]:
    out: list[str] = []
    seen: set[str] = set()
    for v in values:
        cleaned = normalize_plain_text(v, MAX_SHORT_STRING_CHARS)
        key = cleaned.casefold()
        if cleaned and key not in seen:
            seen.add(key)
            out.append(cleaned)
            if max_items is not None and len(out) >= max_items:
                break
    return out


def _str_field(data: dict, name: str, cap: int, warns: list[str]) -> str:
    value = data.get(name, "")
    if not isinstance(value, str):
        raise ValueError(f"Field '{name}' must be a string.")
    text = normalize_plain_text(value, 0)
    if len(text) > cap:
        warns.append(f"Field '{name}' was truncated to the maximum allowed length.")
    return text[:cap]


def _list_field(data: dict, name: str, max_items: int, cap: int, warns: list[str]) -> list[str]:
    value = data.get(name, [])
    # Small local models return list fields as strings/objects; coerce, don't reject.
    if isinstance(value, str):
        value = [value] if value.strip() else []
    elif isinstance(value, dict):
        value = [str(v) for v in value.values()]
    elif not isinstance(value, list):
        warns.append(f"Field '{name}' was not a list; treated as empty.")
        value = []
    out: list[str] = []
    seen: set[str] = set()
    for item in value:
        if not isinstance(item, str):
            warns.append(f"Non-string item ignored in '{name}'.")
            continue
        text = normalize_plain_text(item, 0)
        if len(text) > cap:
            warns.append(f"An item in '{name}' was truncated.")
        text = text[:cap]
        if not text or text.casefold() in seen:
            continue
        seen.add(text.casefold())
        out.append(text)
        if len(out) >= max_items:
            if len(value) > max_items:
                warns.append(f"Field '{name}' was truncated to {max_items} items.")
            break
    return out


_REQUIRED = ("expanded_text", "domain_terms", "functional_intent", "mentioned_components",
             "assumptions", "confidence", "warnings")


def parse_expansion(data: dict) -> ExpansionOut:
    """Bound and sanitize an already-decoded expansion object. Raises ValueError."""
    if not isinstance(data, dict):
        raise ValueError("Expansion must be a JSON object.")
    missing = [k for k in _REQUIRED if k not in data]
    if missing:
        raise ValueError(f"Expansion missing required fields: {', '.join(missing)}.")
    pw: list[str] = []
    expanded = _str_field(data, "expanded_text", MAX_EXPANDED_TEXT_CHARS, pw)
    if not expanded:
        raise ValueError("Field 'expanded_text' must not be empty.")
    intent = _str_field(data, "functional_intent", MAX_SHORT_STRING_CHARS, pw)
    terms = normalize_domain_terms(
        _list_field(data, "domain_terms", MAX_DOMAIN_TERMS, MAX_LIST_ITEM_CHARS, pw),
        MAX_DOMAIN_TERMS)
    comps = _list_field(data, "mentioned_components", MAX_COMPONENTS, MAX_LIST_ITEM_CHARS, pw)
    assumptions = _list_field(data, "assumptions", MAX_ASSUMPTIONS, MAX_SHORT_STRING_CHARS, pw)
    resp_warns = _list_field(data, "warnings", MAX_WARNINGS, MAX_SHORT_STRING_CHARS, pw)

    raw = data.get("confidence", 0.0)
    if isinstance(raw, bool) or not isinstance(raw, (int, float)):
        conf = 0.0
        pw.append("Invalid confidence value; set to 0.0.")
    elif not math.isfinite(float(raw)):
        conf = 0.0
        pw.append("Non-finite confidence value; set to 0.0.")
    else:
        conf = min(max(float(raw), 0.0), 1.0)
        if conf != float(raw):
            pw.append("Confidence value was clamped to the 0..1 range.")
    return ExpansionOut(
        expanded_text=expanded, domain_terms=terms, functional_intent=intent,
        mentioned_components=comps, assumptions=assumptions, confidence=round(conf, 6),
        warnings=_dedupe(resp_warns + pw, MAX_WARNINGS))


# ------------------------------------------------------------------ vocabulary

REQUIREMENT_STOPWORDS = frozenset(
    "a an and any are as at be been being by can could do does during each for from had has have "  # noqa: SIM905
    "how in include includes including into is it its may must of on or provide provides "
    "requirement requirements required shall should support supports such system that the these "
    "this to will with".split())
MAX_TOP_N = 500
_VTOKEN_RE = re.compile(r"[a-z][a-z0-9-]{1,}")
_PUNCT_RE = re.compile(r"[^a-z0-9\-\s]+")
_SPACE_RE = re.compile(r"\s+")


def _vtokens(text: object) -> list[str]:
    norm = unicodedata.normalize("NFKC", str(text or "")).lower()
    norm = _SPACE_RE.sub(" ", _PUNCT_RE.sub(" ", norm)).strip()
    return [t for t in _VTOKEN_RE.findall(norm) if t not in REQUIREMENT_STOPWORDS]


def extract_domain_vocabulary_details(texts: Sequence[str], top_n: int = 50) -> list[dict]:
    """Deterministic uni/bi/trigram terms scored (1+log tf)*(1+log((N+1)/(df+1)))."""
    docs = [d for d in (_vtokens(t) for t in texts or ()) if d]
    top_n = min(max(int(top_n), 0), MAX_TOP_N)
    if not docs or not top_n:
        return []
    tf: Counter[str] = Counter()
    df: Counter[str] = Counter()
    size: dict[str, int] = {}
    for tokens in docs:
        seen: set[str] = set()
        for n in (1, 2, 3):
            for i in range(max(0, len(tokens) - n + 1)):
                parts = tokens[i:i + n]
                if all(p not in REQUIREMENT_STOPWORDS for p in parts):
                    term = " ".join(parts)
                    tf[term] += 1
                    seen.add(term)
                    size[term] = n
        df.update(seen)
    rows = [{
        "term": t, "term_frequency": int(c), "document_frequency": int(df[t]),
        "ngram_size": int(size[t]),
        "score": round((1 + math.log(c)) * (1 + math.log((len(docs) + 1) / (df[t] + 1))), 6),
    } for t, c in tf.items()]
    rows.sort(key=lambda r: (-r["score"], -r["document_frequency"], -r["term_frequency"], r["term"]))
    return rows[:top_n]


def extract_domain_vocabulary(texts: Sequence[str], top_n: int = 50) -> list[str]:
    return [r["term"] for r in extract_domain_vocabulary_details(texts, top_n)]


# --------------------------------------------------------------------- quality

_TOKEN_RE = re.compile(r"[a-z0-9]+(?:[-.][a-z0-9]+)*")
_NUMBER_RE = re.compile(r"(?<![a-z0-9])\d+(?:\.\d+)?(?:%|ms|s|sec|seconds|m|kg|v)?(?![a-z0-9])")


def _round(v: float | None) -> float | None:
    return None if v is None else round(float(v), 6)


def _clamp(v: float) -> float:
    return min(max(float(v), 0.0), 1.0)


def _tokens(text: object) -> list[str]:
    norm = unicodedata.normalize("NFKC", str(text or "")).lower()
    return _TOKEN_RE.findall(_SPACE_RE.sub(" ", norm))


def _meaning(text: object) -> list[str]:
    return [t for t in _tokens(text)
            if t not in REQUIREMENT_STOPWORDS and (len(t) >= 3 or t.isdigit())]


def _numbers(text: object) -> set[str]:
    return {m.group(0) for m in _NUMBER_RE.finditer(str(text or "").lower())}


def evaluate_expansion_quality(original_text: str, enriched_text: str,
                               domain_terms: Sequence[str] | None = None,
                               provider_confidence: float | None = None) -> dict:
    """Conservative augmentation quality metrics and warnings."""
    original = normalize_plain_text(original_text)
    enriched = normalize_plain_text(enriched_text)
    warns: list[str] = []
    o_tok, e_tok = _tokens(original), _tokens(enriched)
    o_mean, e_mean = _meaning(original), _meaning(enriched)
    if not enriched:
        warns.append("Expanded text is empty.")

    ratio = None
    if o_tok:
        ratio = len(e_tok) / len(o_tok)
        if ratio < 0.75:
            warns.append("Expanded text is too short compared with the original requirement.")
        if ratio > 4.0:
            warns.append("Expanded text is too long compared with the original requirement.")

    overlap = 1.0
    if o_mean:
        overlap = len(set(o_mean) & set(e_mean)) / len(set(o_mean))
        if overlap < 0.45:
            warns.append("Lexical overlap with the original requirement is very low.")

    terms = normalize_domain_terms(domain_terms)
    low = enriched.lower()
    coverage = None if not terms else sum(1 for t in terms if t in low) / len(terms)

    missing: list[str] = []
    e_set = set(e_mean)
    for t in o_mean:
        if t not in e_set and t not in missing:
            missing.append(t)
            if len(missing) >= 25:
                break
    many_missing = bool(o_mean) and len(missing) / len(set(o_mean)) > 0.4
    if many_missing:
        warns.append("Expanded text omits many critical tokens from the original requirement.")

    invented = sorted(_numbers(enriched) - _numbers(original))
    if invented:
        warns.append("Expanded text introduces numeric values not present in the original requirement.")

    o_obl = set(o_tok) & {"may", "should", "shall", "must"}
    e_obl = set(e_tok) & {"may", "should", "shall", "must"}
    changed = bool(o_obl & {"may", "should"}) and bool(e_obl & {"shall", "must"})
    if changed:
        warns.append("Expanded text strengthens obligation language from may/should to shall/must.")

    risk = _clamp(
        0.25 * bool(invented) + 0.25 * changed + 0.20 * (overlap < 0.45)
        + 0.15 * (ratio is not None and (ratio < 0.75 or ratio > 4.0)) + 0.15 * many_missing)
    base = 0.5 if provider_confidence is None else _clamp(provider_confidence)
    return {
        "length_ratio": _round(ratio), "lexical_overlap": _round(overlap),
        "domain_term_coverage": _round(coverage),
        "missing_original_critical_tokens": missing, "invented_numeric_values": invented,
        "changed_obligation_strength": changed, "hallucination_risk_score": _round(risk),
        "adjusted_confidence_score": _round(_clamp(base * (1.0 - risk))), "warnings": warns,
    }


def build_quality_report(reqs: Sequence[tuple[int, str]], expansions: Sequence[Expansion],
                         domain_vocabulary: Sequence[str] | None = None) -> dict:
    rows: list[dict] = []
    for idx, ((rid, text), exp) in enumerate(zip(reqs, expansions, strict=True)):
        if exp.status != "succeeded":
            rows.append({"index": idx, "requirement_id": rid, "status": "failed",
                         "warnings": ["Requirement enrichment failed."]})
            continue
        metrics = evaluate_expansion_quality(
            text, exp.expanded_text, exp.domain_terms or domain_vocabulary, exp.confidence)
        rows.append({"index": idx, "requirement_id": rid, "status": "succeeded", **metrics})
    ok = [r for r in rows if r["status"] == "succeeded"]
    risks = [r["hallucination_risk_score"] for r in ok]
    adj = [r["adjusted_confidence_score"] for r in ok]
    warns = list(dict.fromkeys(w for r in rows for w in r.get("warnings", [])))
    return {
        "total": len(rows), "succeeded": len(ok), "failed": len(rows) - len(ok),
        "aggregate": {
            "mean_hallucination_risk_score": _round(sum(risks) / len(risks)) if risks else None,
            "mean_adjusted_confidence_score": _round(sum(adj) / len(adj)) if adj else None,
        },
        "per_requirement": rows, "warnings": warns,
    }


# ------------------------------------------------------- deterministic fallback

_COMPONENT_RE = re.compile(r"\b(?:[A-Z][A-Za-z0-9-]{2,}|[A-Z]{2,}(?:-[A-Z0-9]+)*)\b")
_INTENT_PREFIXES = tuple(f"{a}{m} " for a in ("the system ", "system ")
                         for m in ("shall", "must", "should", "may"))


def _derive_intent(text: str) -> str:
    cleaned = normalize_plain_text(text, MAX_SHORT_STRING_CHARS)
    low = cleaned.lower()
    for p in _INTENT_PREFIXES:
        if low.startswith(p):
            return cleaned[len(p):].strip(" .") or cleaned
    return cleaned


def mock_expansion(text: str, domain_vocabulary: Sequence[str] | None = None) -> ExpansionOut:
    original = normalize_plain_text(text)
    if not original:
        raise ValueError("Requirement text is empty.")
    local = extract_domain_vocabulary([original], top_n=12)
    low = original.lower()
    supplied = [t for t in normalize_domain_terms(domain_vocabulary) if t in low]
    terms = _dedupe([*supplied, *local], MAX_DOMAIN_TERMS)
    comps = [m.group(0) for m in _COMPONENT_RE.finditer(original)]
    comps += [t for t in terms if any(p in t for p in ("module", "service", "api", "sensor", "controller"))]
    comps = _dedupe(comps, MAX_COMPONENTS)
    intent = _derive_intent(original)
    parts = [f"Original requirement: {original}"]
    if intent and intent != original:
        parts.append(f"Functional intent: {intent}.")
    if comps:
        parts.append(f"Explicitly mentioned components: {', '.join(comps)}.")
    if terms:
        parts.append(f"Domain vocabulary from the requirement: {', '.join(terms[:12])}.")
    return parse_expansion({
        "expanded_text": " ".join(parts), "domain_terms": terms, "functional_intent": intent,
        "mentioned_components": comps, "assumptions": [], "confidence": 0.85, "warnings": [],
    })


# ------------------------------------------------------------------- entrypoint

def expand_all(reqs: list[tuple[int, str]], gateway: LLMGateway,
               provider: ProviderName | None = None, on_chunk=None,
               chunk: int = 32) -> list[Expansion]:
    """`on_chunk(done, total, new)` runs after each chunk; it may raise to abort."""
    vocab = extract_domain_vocabulary([t for _, t in reqs], top_n=50)
    out: list[Expansion | None] = [None] * len(reqs)
    todo: list[int] = []
    for i, (rid, text) in enumerate(reqs):
        if normalize_plain_text(text):
            todo.append(i)
        else:
            out[i] = Expansion(rid, status="failed", source="fallback",
                               error="Requirement text is empty.")
    if on_chunk and len(todo) < len(reqs):  # report the empty-text failures up front
        on_chunk(len(reqs) - len(todo), len(reqs), [x for x in out if x is not None])
    step = max(1, chunk)
    done = len(reqs) - len(todo)
    for start in range(0, len(todo), step):
        idx = todo[start:start + step]
        prompts = [
            Prompt("enrich", PROMPT_VERSION, SYSTEM,
                   {"requirement": normalize_plain_text(reqs[i][1]), "domain_vocabulary": vocab},
                   output_model=ExpansionOut)
            for i in idx
        ]
        results = gateway.map(
            prompts, lambda p: mock_expansion(p.data["requirement"], p.data["domain_vocabulary"]),
            provider)
        for i, res in zip(idx, results, strict=True):
            rid, text = reqs[i]
            try:
                parsed = parse_expansion(res.output.model_dump())
            except ValueError as e:
                out[i] = Expansion(rid, status="failed", source=res.source, error=str(e))
                continue
            quality = evaluate_expansion_quality(text, parsed.expanded_text, parsed.domain_terms,
                                                 parsed.confidence)
            out[i] = Expansion(
                rid, parsed.expanded_text, parsed.domain_terms, parsed.functional_intent,
                parsed.mentioned_components, parsed.assumptions, parsed.confidence,
                _dedupe([*parsed.warnings, *quality["warnings"]]), "succeeded", res.source,
                res.error)
        done += len(idx)
        if on_chunk:
            on_chunk(done, len(reqs), [out[i] for i in idx])
    return out  # type: ignore[return-value]
