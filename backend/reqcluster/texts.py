"""Pure model-input text builder (no config/db)."""
import unicodedata
from collections.abc import Sequence

MAX_TEXT_CHARS = 12_000


def _normalize_text(text: object) -> str:
    if text is None:
        return ""
    return unicodedata.normalize("NFC", str(text)).replace("\r\n", "\n").replace("\r", "\n")


def _is_missing_enrichment(text: object) -> bool:
    return _normalize_text(text).strip() == ""


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
    for i, original in enumerate(originals):
        base = _normalize_text(original)
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
