"""Pure model-input text builder (no config/db)."""
import math
import re
import unicodedata
from collections.abc import Sequence

MAX_TEXT_CHARS = 12_000


def _normalize_text(text: object) -> str:
    if text is None:
        return ""
    return unicodedata.normalize("NFC", str(text)).replace("\r\n", "\n").replace("\r", "\n")


def _is_missing_enrichment(text: object) -> bool:
    return _normalize_text(text).strip() == ""


def strip_boilerplate(texts: Sequence[str]) -> list[str]:
    """Drop sentences templated across many rows; they swamp embeddings (evidence: uav_fms_stress_600)."""
    split = [[p for p in re.split(r"(?<=[.!?;)])\s+", t) if p] for t in texts]

    def key(s: str) -> str:
        return re.sub(r"\d+", "#", s.lower()).strip(" \t\n.,;:!?()[]-")

    rows: dict[str, int] = {}
    for parts in split:
        for k in {key(p) for p in parts}:
            rows[k] = rows.get(k, 0) + 1
    limit = max(5, math.ceil(0.05 * len(texts)))
    out = []
    for t, parts in zip(texts, split, strict=True):
        kept = [p for p in parts if rows[key(p)] < limit]
        out.append(t if len(kept) == len(parts) or not kept else " ".join(kept))
    return out


def build_texts(
    texts: Sequence[str], mode: str, enriched: Sequence[str | None] | None = None
) -> tuple[list[str], list[str]]:
    """Model input per mode; missing enrichment falls back to base with a warning."""
    originals = list(texts)
    warnings: list[str] = []
    if mode == "base":
        enriched = None
    elif enriched is None:
        warnings.append(f"{mode} mode without enriched texts; using base texts")
        enriched = [None] * len(originals)
    elif len(enriched) != len(originals):
        raise ValueError("texts and enriched must have the same length.")

    out: list[str] = []
    missing = 0
    bases = strip_boilerplate([_normalize_text(o) for o in originals])
    for i, base in enumerate(bases):
        value = enriched[i] if enriched is not None else None
        if mode == "base":
            payload = base
        elif _is_missing_enrichment(value):
            payload = base
            missing += 1
        elif mode == "enriched":
            payload = _normalize_text(value)
        else:
            payload = f"Original Requirement:\n{base}\n\nDomain-Aware Context:\n{_normalize_text(value)}"
        out.append(payload[:MAX_TEXT_CHARS])
    if missing and enriched is not None:
        warnings.append(f"{missing} requirement(s) missing enriched text; used base text")
    return out, warnings
