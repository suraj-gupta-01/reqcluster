"""Artifact extraction for producer-consumer dependency inference (DP5).

For each requirement, identifies the named data items / signals / states it
**produces** (generates, outputs, stores …) and **consumes** (reads, receives,
depends on …). This drives Pass 0 in ``dependency_tree.build_dependency_tree``.

Design principles
-----------------
* Deterministic — no LLM or external calls.
* Regex-only NP extraction anchored on output/input verbs found in the text.
* Artifact names are normalized (lower-case, articles stripped, whitespace
  collapsed) so that "an Altitude Report" and "altitude report" match.
* A near-match step (optional) can be layered on top using embedding cosine
  similarity on artifact name strings to catch synonyms like "temperature" /
  "temp reading".
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Dict, List, Set, Tuple

# ---------------------------------------------------------------------------
# Verb banks  (must be kept in sync with dependency_tree._OUTPUT_VERBS /
# _INPUT_VERBS so that Pass 0 and Pass 2 are consistent)
# ---------------------------------------------------------------------------

_OUTPUT_VERBS: Tuple[str, ...] = (
    "generate", "produce", "output", "compute", "calculate",
    "emit", "return", "report", "transmit", "send", "create", "store",
    "record", "log", "measure", "detect", "raise", "issue", "provide",
    "publish", "broadcast", "write", "save", "derive", "forward",
)

_INPUT_VERBS: Tuple[str, ...] = (
    "receive", "require", "use", "read", "accept", "consume", "depend",
    "retrieve", "obtain", "process", "monitor", "respond", "trigger",
    "subscribe", "listen", "load", "fetch", "import", "request", "poll",
)

# ---------------------------------------------------------------------------
# Regex helpers
# ---------------------------------------------------------------------------

# Matches a verb word (whole-word, case-insensitive).
def _verb_pattern(verbs: Tuple[str, ...]) -> re.Pattern[str]:
    alt = "|".join(re.escape(v) for v in sorted(verbs, key=len, reverse=True))
    return re.compile(rf"\b(?:{alt})s?\b", re.IGNORECASE)


_OUT_VERB_RE = _verb_pattern(_OUTPUT_VERBS)
_IN_VERB_RE  = _verb_pattern(_INPUT_VERBS)

# Noun phrase: optional article/determiner + adjectives + noun(s) with
# possible hyphens/slashes (e.g. "raw sensor data", "fault-detection flag").
# Stops at prepositions, conjunctions, punctuation, or end of input.
_NP_RE = re.compile(
    r"""
    (?:
        (?:a|an|the|this|that|each|every|its|their|all)\s+   # optional det.
    )?
    (?:[A-Za-z][A-Za-z0-9\-]{1,}[ ])*   # leading adjective/modifier words
    [A-Za-z][A-Za-z0-9\-]{2,}           # head noun (at least 3 chars)
    (?:\s+(?:data|value|values|status|flag|report|signal|
             message|packet|reading|readings|output|result|
             results|estimate|request|response|event|state|
             list|table|record|records|log|buffer|stream|
             rate|count|index|identifier|id|code|error|
             alert|notification|update|feed|set|map|matrix
             ))?                          # optional semantic head completion
    """,
    re.VERBOSE | re.IGNORECASE,
)

# Articles and determiners to strip from the front of an extracted NP.
_ARTICLE_RE = re.compile(
    r"^(?:a|an|the|this|that|each|every|its|their|all)\s+",
    re.IGNORECASE,
)

# Very generic words that are not meaningful artifact names.
_GENERIC = {
    "data", "information", "value", "values", "result", "results",
    "output", "outputs", "input", "inputs", "status", "state", "signal",
    "message", "packet", "event", "response", "request", "report",
    "update", "feed", "stream", "buffer", "record", "records",
    "system", "unit", "module", "function", "interface", "parameter",
    "configuration", "setting", "mode", "level", "error", "alert",
    "notification", "log", "list", "table", "set", "map", "matrix",
    "index", "identifier", "code", "count", "rate", "flag", "bit",
    "byte", "field", "item", "object", "entity", "type", "class",
    "process", "thread", "task", "service", "component", "element",
}


def _normalize(text: str) -> str:
    """Lower-case, strip leading article, collapse whitespace."""
    t = text.strip().lower()
    t = _ARTICLE_RE.sub("", t)
    return re.sub(r"\s+", " ", t).strip()


def _extract_np_after_verb(
    sentence: str, verb_re: re.Pattern[str]
) -> List[str]:
    """Find all verb occurrences in *sentence* and extract the NP that follows."""
    results: List[str] = []
    for m in verb_re.finditer(sentence):
        tail = sentence[m.end():].lstrip()
        np_m = _NP_RE.match(tail)
        if np_m:
            raw = np_m.group(0).strip()
            norm = _normalize(raw)
            # Filter: must be ≥ 3 chars, not a pure generic word
            if len(norm) >= 3 and norm not in _GENERIC:
                results.append(norm)
    return results


# Sentence splitter (simple — splits on ". ", "; ", or newline).
_SENT_SPLIT_RE = re.compile(r"(?<=[.;])\s+|\n")


def _sentences(text: str) -> List[str]:
    return [s.strip() for s in _SENT_SPLIT_RE.split(text) if s.strip()]


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

@dataclass
class ReqArtifacts:
    """Artifacts produced and consumed by a single requirement."""
    req_index: int
    req_id: str
    produced: Set[str] = field(default_factory=set)
    consumed: Set[str] = field(default_factory=set)

    def to_dict(self) -> Dict:
        return {
            "req_index": self.req_index,
            "req_id": self.req_id,
            "produced": sorted(self.produced),
            "consumed": sorted(self.consumed),
        }


def extract_artifacts(
    texts: List[str],
    req_ids: List[str],
) -> List[ReqArtifacts]:
    """Extract produced and consumed artifact names from requirement texts.

    Args:
        texts:   Requirement text strings, length N.
        req_ids: Requirement identifiers, length N.

    Returns:
        List of :class:`ReqArtifacts` (one per requirement, same order).

    Example::

        texts = [
            "The system shall generate an altitude report every second.",
            "The display shall receive the altitude report from the nav unit.",
        ]
        arts = extract_artifacts(texts, ["REQ-001", "REQ-002"])
        # arts[0].produced == {"altitude report"}
        # arts[1].consumed == {"altitude report"}
    """
    result: List[ReqArtifacts] = []
    for idx, (text, rid) in enumerate(zip(texts, req_ids)):
        ra = ReqArtifacts(req_index=idx, req_id=rid)
        for sent in _sentences(text):
            ra.produced.update(_extract_np_after_verb(sent, _OUT_VERB_RE))
            ra.consumed.update(_extract_np_after_verb(sent, _IN_VERB_RE))
        result.append(ra)
    return result


def build_artifact_index(
    artifacts: List[ReqArtifacts],
) -> Tuple[Dict[str, List[int]], Dict[str, List[int]]]:
    """Build lookup tables: artifact_name → list of producer/consumer indices.

    Args:
        artifacts: Output of :func:`extract_artifacts`.

    Returns:
        ``(producers, consumers)`` where each is
        ``Dict[artifact_name, List[req_index]]``.
    """
    producers: Dict[str, List[int]] = {}
    consumers: Dict[str, List[int]] = {}

    for ra in artifacts:
        for art in ra.produced:
            producers.setdefault(art, []).append(ra.req_index)
        for art in ra.consumed:
            consumers.setdefault(art, []).append(ra.req_index)

    return producers, consumers
