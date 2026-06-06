"""Dependency tree inference for engineering requirements (DP5).

Infers hierarchical and sequential relationships between requirements based on
the inputs, pre-conditions, and outputs implied by each requirement's text,
combined with semantic similarity. Produces a directed acyclic graph with
typed, weighted, explainable edges plus per-node tree levels.

Edge direction convention: ``source -> target`` means *source is a prerequisite
of target* (target depends on / follows / specialises source).

This module is deterministic and performs no LLM calls or DB access. Rationale
prose can be layered on top by the service/LLM provider.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any, Dict, List, Optional

import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

from .labeling import STOPWORDS

# Relation types
HIERARCHICAL = "hierarchical"
SEQUENTIAL = "sequential"
DATA = "data"
REFERENCE = "reference"

# Matches identifiers such as REQ-001, SR_12, FR 4, SYS-1007.
_REQ_ID_RE = re.compile(r"\b(?:REQ|SR|FR|SYS|R)[-_ ]?\d{1,5}\b", re.IGNORECASE)

_SEQUENTIAL_CUES = (
    "after", "once", "upon", "following", "subsequent to",
    "prior to", "before", "until", "as soon as", "in response to",
    "when ", "whenever",
)
_HIERARCHICAL_CUES = (
    "in accordance with", "as defined in", "as specified in", "subject to",
    "comply with", "conform to", "according to", "consistent with",
    "as required by", "in compliance with", "as described in",
)
_OUTPUT_VERBS = (
    "provide", "generate", "produce", "output", "compute", "calculate",
    "emit", "return", "report", "transmit", "send", "create", "store",
    "record", "log", "measure", "detect", "raise", "issue",
)
_INPUT_VERBS = (
    "receive", "require", "use", "read", "accept", "consume", "depend",
    "retrieve", "obtain", "process", "monitor", "respond to", "trigger on",
    "based on", "derived from",
)

_WORD_RE = re.compile(r"[a-zA-Z][a-zA-Z0-9\-]{2,}")


@dataclass
class DependencyEdge:
    source: int
    target: int
    relation: str
    weight: float
    rationale: str = ""

    def to_dict(self) -> Dict[str, Any]:
        return {
            "source": int(self.source),
            "target": int(self.target),
            "relation": self.relation,
            "weight": round(float(self.weight), 4),
            "rationale": self.rationale,
        }


@dataclass
class DependencyNode:
    id: int
    node_id: str
    requirement_text: str
    cluster_id: int
    level: int = 0

    def to_dict(self) -> Dict[str, Any]:
        return {
            "id": int(self.id),
            "node_id": self.node_id,
            "requirement_text": self.requirement_text,
            "cluster_id": int(self.cluster_id),
            "level": int(self.level),
        }


def _normalize_id(value: Optional[str]) -> str:
    if not value:
        return ""
    return re.sub(r"[-_ ]", "", str(value)).upper()


def _salient_tokens(text: str) -> set[str]:
    """Content words from a requirement, excluding requirements-domain stopwords."""
    tokens = set()
    for match in _WORD_RE.finditer(text.lower()):
        word = match.group(0)
        if word not in STOPWORDS:
            tokens.add(word)
    return tokens


def _first_cue(text_lower: str, cues: tuple[str, ...]) -> Optional[str]:
    for cue in cues:
        if cue in text_lower:
            return cue.strip()
    return None


def _verb_present(text_lower: str, verbs: tuple[str, ...]) -> bool:
    return any(re.search(rf"\b{re.escape(v)}", text_lower) for v in verbs)


def _add_edge(
    edges: Dict[tuple[int, int], DependencyEdge],
    source: int,
    target: int,
    relation: str,
    weight: float,
    rationale: str,
) -> None:
    if source == target:
        return
    key = (source, target)
    existing = edges.get(key)
    if existing is None or weight > existing.weight:
        edges[key] = DependencyEdge(source, target, relation, weight, rationale)


def _break_cycles(
    n: int, edges: List[DependencyEdge]
) -> List[DependencyEdge]:
    """Greedily keep the highest-weight edges that preserve acyclicity.

    Produces a DAG so that tree levels are well defined. Edges are added in
    descending weight order; any edge that would introduce a cycle is dropped.
    """
    kept: List[DependencyEdge] = []
    adjacency: Dict[int, set[int]] = {i: set() for i in range(n)}

    def reachable(start: int, goal: int) -> bool:
        # Is `goal` reachable from `start` following current edges?
        stack = [start]
        seen = set()
        while stack:
            node = stack.pop()
            if node == goal:
                return True
            if node in seen:
                continue
            seen.add(node)
            stack.extend(adjacency[node])
        return False

    for edge in sorted(edges, key=lambda e: e.weight, reverse=True):
        # Adding source->target creates a cycle iff target can already reach source.
        if reachable(edge.target, edge.source):
            continue
        adjacency[edge.source].add(edge.target)
        kept.append(edge)
    return kept


def _assign_levels(n: int, edges: List[DependencyEdge]) -> List[int]:
    """Longest-path level for each node on the DAG (roots at level 0)."""
    children: Dict[int, List[int]] = {i: [] for i in range(n)}
    indegree = [0] * n
    for edge in edges:
        children[edge.source].append(edge.target)
        indegree[edge.target] += 1

    levels = [0] * n
    # Kahn topological order, propagating max level.
    queue = [i for i in range(n) if indegree[i] == 0]
    remaining = indegree[:]
    while queue:
        node = queue.pop()
        for child in children[node]:
            levels[child] = max(levels[child], levels[node] + 1)
            remaining[child] -= 1
            if remaining[child] == 0:
                queue.append(child)
    return levels


def build_dependency_tree(
    embeddings: np.ndarray,
    texts: List[str],
    req_ids: List[str],
    labels: np.ndarray,
    top_k: int = 8,
    sim_threshold: float = 0.45,
    max_edges: int = 2000,
) -> Dict[str, Any]:
    """Infer a dependency DAG over requirements.

    Args:
        embeddings: (N, D) original embeddings (used for candidate generation).
        texts: requirement texts, length N.
        req_ids: requirement identifiers, length N.
        labels: cluster labels, length N (-1 = noise).
        top_k: max semantically-similar candidates examined per requirement.
        sim_threshold: minimum cosine similarity for a candidate pair.
        max_edges: cap on returned edges (highest weight retained).

    Returns:
        ``{"nodes": [...], "edges": [...], "stats": {...}}`` (JSON-serialisable).
    """
    n = len(texts)
    labels = np.asarray(labels)
    lowers = [t.lower() for t in texts]
    id_norm = [_normalize_id(r) for r in req_ids]
    id_to_index = {id_norm[i]: i for i in range(n) if id_norm[i]}
    token_sets = [_salient_tokens(t) for t in texts]

    edges: Dict[tuple[int, int], DependencyEdge] = {}

    # 1. Explicit cross-references: "as defined in REQ-002" => REQ-002 -> this.
    for i in range(n):
        for match in _REQ_ID_RE.finditer(texts[i]):
            key = _normalize_id(match.group(0))
            j = id_to_index.get(key)
            if j is None or j == i:
                continue
            relation = (
                HIERARCHICAL
                if _first_cue(lowers[i], _HIERARCHICAL_CUES)
                else REFERENCE
            )
            _add_edge(
                edges, j, i, relation, 0.95,
                f"{req_ids[i]} explicitly references {req_ids[j]}.",
            )

    # 2. Semantic candidate pairs scored by lexical signals.
    if n >= 2:
        sim = cosine_similarity(embeddings)
        for i in range(n):
            order = np.argsort(sim[i])[::-1]
            examined = 0
            for j in order:
                j = int(j)
                if j == i:
                    continue
                s = float(sim[i, j])
                if s < sim_threshold:
                    break
                examined += 1
                if examined > top_k:
                    break

                shared = token_sets[i] & token_sets[j]
                if not shared:
                    continue
                shared_term = sorted(shared, key=len, reverse=True)[0]

                # 2a. Data dependency: i produces what j consumes => i -> j.
                if _verb_present(lowers[i], _OUTPUT_VERBS) and _verb_present(
                    lowers[j], _INPUT_VERBS
                ):
                    _add_edge(
                        edges, i, j, DATA, 0.5 + 0.4 * s,
                        f"{req_ids[i]} produces and {req_ids[j]} consumes "
                        f"'{shared_term}'.",
                    )

                # 2b. Sequential: j is gated by a pre-condition cue => i -> j.
                cue = _first_cue(lowers[j], _SEQUENTIAL_CUES)
                if cue and cue not in ("before", "prior to"):
                    _add_edge(
                        edges, i, j, SEQUENTIAL, 0.45 + 0.4 * s,
                        f"{req_ids[j]} occurs '{cue}' a precondition shared "
                        f"with {req_ids[i]} ('{shared_term}').",
                    )

                # 2c. Hierarchical: same cluster, i more general than j.
                if (
                    labels[i] != -1
                    and labels[i] == labels[j]
                    and len(token_sets[i]) + 1 < len(token_sets[j])
                    and s >= max(sim_threshold, 0.6)
                ):
                    _add_edge(
                        edges, i, j, HIERARCHICAL, 0.4 + 0.3 * s,
                        f"{req_ids[i]} is broader than {req_ids[j]} within the "
                        f"same cluster.",
                    )

    edge_list = list(edges.values())
    edge_list = _break_cycles(n, edge_list)
    if len(edge_list) > max_edges:
        edge_list = sorted(edge_list, key=lambda e: e.weight, reverse=True)[:max_edges]

    levels = _assign_levels(n, edge_list)

    nodes = [
        DependencyNode(
            id=i,
            node_id=req_ids[i] if i < len(req_ids) and req_ids[i] else f"REQ-{i + 1:03d}",
            requirement_text=texts[i],
            cluster_id=int(labels[i]) if i < len(labels) else -1,
            level=levels[i],
        ).to_dict()
        for i in range(n)
    ]
    edges_out = [e.to_dict() for e in edge_list]

    relation_counts: Dict[str, int] = {}
    for e in edge_list:
        relation_counts[e.relation] = relation_counts.get(e.relation, 0) + 1

    roots = [i for i in range(n) if levels[i] == 0 and any(
        e.source == i for e in edge_list
    )]

    return {
        "nodes": nodes,
        "edges": edges_out,
        "stats": {
            "n_nodes": n,
            "n_edges": len(edges_out),
            "max_depth": max(levels) if levels else 0,
            "relation_counts": relation_counts,
            "root_count": len(roots),
        },
    }
