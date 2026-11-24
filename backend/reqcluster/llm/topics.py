"""LLM topic discovery + classification: an LLM proposes topics, then assigns every requirement."""
import random

import numpy as np
from pydantic import BaseModel

from .prompts import Prompt

TAXONOMY_SYSTEM = (
    "You receive numbered system requirements. Identify 4 to 20 non-overlapping subsystems or "
    "functional areas that partition these requirements, so that every requirement fits exactly "
    'one. Reply with JSON only: {"topics":[{"name": "...", "description": "one sentence"}]}')
CLASSIFY_SYSTEM = (
    "You receive numbered topics and numbered requirements. Assign each requirement to exactly "
    'one topic. Reply with JSON only: {"assignments":[{"id": <requirement number>, '
    '"topic": <topic number>}]}')


class Topic(BaseModel):
    name: str
    description: str = ""


class Taxonomy(BaseModel):
    topics: list[Topic]


class Assignment(BaseModel):
    id: int
    topic: int


class Assignments(BaseModel):
    assignments: list[Assignment]


def discover(gateway, texts, provider=None, sample=200, seed=0) -> list[Topic] | None:
    idx = list(range(len(texts)))
    if len(idx) > sample:
        idx = sorted(random.Random(seed).sample(idx, sample))
    prompt = Prompt("topics_taxonomy", "1", TAXONOMY_SYSTEM,
                    {"requirements": [f"{j + 1}. {texts[i]}" for j, i in enumerate(idx)]},
                    output_model=Taxonomy)
    res = gateway.complete(prompt, lambda: Taxonomy(topics=[]), provider)
    if res.source == "fallback":
        return None
    seen, topics = set(), []
    for t in res.output.topics:
        key = t.name.strip().lower()
        if key and key not in seen:
            seen.add(key)
            topics.append(t)
    return topics if len(topics) >= 2 else None


def classify(gateway, texts, topics, provider=None, batch=20, progress=None) -> np.ndarray:
    tl = [f"{j + 1}. {t.name}: {t.description}" for j, t in enumerate(topics)]
    starts = list(range(0, len(texts), batch))
    prompts = [Prompt("topics_classify", "1", CLASSIFY_SYSTEM,
                      {"topics": tl, "requirements": [f"{k + 1}. {t}" for k, t in
                                                      enumerate(texts[b:b + batch])]},
                      output_model=Assignments) for b in starts]
    results = gateway.map(prompts, lambda p: Assignments(assignments=[]), provider)
    lab = np.full(len(texts), -1, dtype=np.int64)
    for b, res in zip(starts, results, strict=True):
        size = len(texts[b:b + batch])
        for a in getattr(res.output, "assignments", []):
            if 1 <= a.id <= size and 1 <= a.topic <= len(topics):
                lab[b + a.id - 1] = a.topic - 1
        if progress:
            progress(min(b + batch, len(texts)), len(texts))
    return lab
