from __future__ import annotations

import math
import re
import unicodedata
from collections import Counter
from typing import Iterable, Sequence


MAX_TOP_N = 500

REQUIREMENT_STOPWORDS = {
    "a",
    "an",
    "and",
    "any",
    "are",
    "as",
    "at",
    "be",
    "been",
    "being",
    "by",
    "can",
    "could",
    "do",
    "does",
    "during",
    "each",
    "for",
    "from",
    "had",
    "has",
    "have",
    "how",
    "in",
    "include",
    "includes",
    "including",
    "into",
    "is",
    "it",
    "its",
    "may",
    "must",
    "of",
    "on",
    "or",
    "provide",
    "provides",
    "requirement",
    "requirements",
    "required",
    "shall",
    "should",
    "support",
    "supports",
    "such",
    "system",
    "that",
    "the",
    "these",
    "this",
    "to",
    "will",
    "with",
}

_TOKEN_RE = re.compile(r"[a-z][a-z0-9-]{1,}")
_PUNCT_RE = re.compile(r"[^a-z0-9\-\s]+")
_SPACE_RE = re.compile(r"\s+")


def _normalize_text(text: object) -> str:
    if text is None:
        return ""
    normalized = unicodedata.normalize("NFKC", str(text)).lower()
    normalized = _PUNCT_RE.sub(" ", normalized)
    return _SPACE_RE.sub(" ", normalized).strip()


def _tokens(text: object) -> list[str]:
    normalized = _normalize_text(text)
    tokens = _TOKEN_RE.findall(normalized)
    return [tok for tok in tokens if tok not in REQUIREMENT_STOPWORDS]


def _ngrams(tokens: Sequence[str], n: int) -> Iterable[str]:
    for idx in range(0, max(0, len(tokens) - n + 1)):
        parts = tokens[idx : idx + n]
        if len(parts) == n and all(part not in REQUIREMENT_STOPWORDS for part in parts):
            yield " ".join(parts)


def extract_domain_vocabulary_details(
    texts: Sequence[str],
    top_n: int = 50,
) -> list[dict]:
    """
    Extract deterministic unigram, bigram, and trigram domain vocabulary.

    Scores use a small TF-IDF-like formula:
    ``(1 + log(tf)) * (1 + log((N + 1) / (df + 1)))``.
    """
    if texts is None:
        return []

    docs = [_tokens(text) for text in texts]
    docs = [doc for doc in docs if doc]
    if not docs:
        return []

    bounded_top_n = min(max(int(top_n), 0), MAX_TOP_N)
    if bounded_top_n == 0:
        return []

    tf: Counter[str] = Counter()
    df: Counter[str] = Counter()
    ngram_size: dict[str, int] = {}

    for tokens in docs:
        doc_terms: set[str] = set()
        for size in (1, 2, 3):
            for term in _ngrams(tokens, size):
                tf[term] += 1
                doc_terms.add(term)
                ngram_size[term] = size
        for term in doc_terms:
            df[term] += 1

    n_docs = len(docs)
    rows = []
    for term, term_tf in tf.items():
        term_df = df[term]
        score = (1.0 + math.log(term_tf)) * (1.0 + math.log((n_docs + 1.0) / (term_df + 1.0)))
        rows.append(
            {
                "term": term,
                "term_frequency": int(term_tf),
                "document_frequency": int(term_df),
                "ngram_size": int(ngram_size[term]),
                "score": round(float(score), 6),
            }
        )

    rows.sort(
        key=lambda item: (
            -float(item["score"]),
            -int(item["document_frequency"]),
            -int(item["term_frequency"]),
            str(item["term"]),
        )
    )
    return rows[:bounded_top_n]


def extract_domain_vocabulary(
    texts: Sequence[str],
    top_n: int = 50,
    return_scores: bool = False,
) -> list[str] | list[dict]:
    details = extract_domain_vocabulary_details(texts, top_n=top_n)
    if return_scores:
        return details
    return [str(row["term"]) for row in details]
