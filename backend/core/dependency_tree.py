"""Dependency tree inference for engineering requirements (DP5).

Infers hierarchical and sequential relationships between requirements based on:
  - **Pass 0** – Producer-consumer artifact matching: requirements that produce
    a named data item are linked to requirements that consume that same item.
  - **Pass 1** – Explicit cross-references (REQ-ID mentions in text).
  - **Pass 2** – Semantic candidate pairs: sequential cues and hierarchical
    (same-cluster, more-general-to-specific) edges.

Edge direction convention: ``source -> target`` means *source is a prerequisite
of target* (target depends on / follows / specialises source).

Every edge carries a structured ``evidence`` dict recording which pass fired and
the raw feature values behind its weight, plus a coarse ``confidence_tier``.
That is what makes an edge explainable without any LLM call — see
``services.dependency_service`` for the optional prose layer on top.

This module is deterministic and performs no LLM calls or DB access.
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np

from .artifact_extractor import (
    DEFAULT_NEAR_MATCH_THRESHOLD,
    build_artifact_index,
    extract_artifacts,
    resolve_near_matches,
)
from .labeling import STOPWORDS

logger = logging.getLogger(__name__)

# Relation types
HIERARCHICAL = "hierarchical"
SEQUENTIAL = "sequential"
DATA = "data"
REFERENCE = "reference"

# Pass identifiers recorded in each edge's evidence payload.
PASS_ARTIFACT = "pass_0_artifact_match"
PASS_REFERENCE = "pass_1_explicit_reference"
PASS_SEQUENTIAL = "pass_2_sequential"
PASS_HIERARCHICAL = "pass_2_hierarchical"

# Confidence tiers for display. Pass 1 explicit references and Pass 0 artifact
# matches are structural facts about the text, so they land in "High"; Pass 2
# edges are similarity-driven and fall into Medium/Low by weight.
TIER_HIGH = "High"
TIER_MEDIUM = "Medium"
TIER_LOW = "Low"
_TIER_HIGH_MIN = 0.85
_TIER_MEDIUM_MIN = 0.60

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

# Cues that mark the *target* as the precondition rather than the consequence,
# so an edge derived from them must not be drawn in the forward direction.
_REVERSED_SEQUENTIAL_CUES = frozenset({"before", "prior to"})

_WORD_RE = re.compile(r"[a-zA-Z][a-zA-Z0-9\-]{2,}")

# A single generic artifact name shared by many requirements would otherwise
# expand into |producers| x |consumers| edges. Cap the fan-out per artifact so a
# noisy extraction cannot quadratically blow up the edge set.
MAX_PAIRS_PER_ARTIFACT = 400

# Row-block width for the Pass 2 similarity scan. The full N x N matrix is never
# materialised: at N = 35 000 that alone would be ~4.9 GB.
_SIM_BLOCK = 512


@dataclass
class DependencyEdge:
    source: int
    target: int
    relation: str
    weight: float
    rationale: str = ""
    evidence: Dict[str, Any] = field(default_factory=dict)

    def confidence_tier(self) -> str:
        """Coarse display bucket for the edge weight.

        Buckets exist so the UI can say "High confidence" instead of inviting
        scrutiny of why a weight is exactly 0.73.
        """
        if self.weight >= _TIER_HIGH_MIN:
            return TIER_HIGH
        if self.weight >= _TIER_MEDIUM_MIN:
            return TIER_MEDIUM
        return TIER_LOW

    def to_dict(self) -> Dict[str, Any]:
        return {
            "source": int(self.source),
            "target": int(self.target),
            "relation": self.relation,
            "weight": round(float(self.weight), 4),
            "rationale": self.rationale,
            "confidence_tier": self.confidence_tier(),
            "evidence": self.evidence,
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


def _add_edge(
    edges: Dict[tuple[int, int], DependencyEdge],
    source: int,
    target: int,
    relation: str,
    weight: float,
    rationale: str,
    evidence: Optional[Dict[str, Any]] = None,
) -> None:
    if source == target:
        return
    key = (source, target)
    existing = edges.get(key)
    if existing is None or weight > existing.weight:
        edges[key] = DependencyEdge(
            source, target, relation, weight, rationale, evidence or {}
        )


def _break_cycles(n: int, edges: List[DependencyEdge]) -> List[DependencyEdge]:
    """Greedily keep the highest-weight edges that preserve acyclicity.

    Produces a DAG so that tree levels are well defined. Edges are added in
    descending weight order; any edge that would introduce a cycle is dropped.
    """
    kept: List[DependencyEdge] = []
    adjacency: Dict[int, set[int]] = {}

    def reachable(start: int, goal: int) -> bool:
        # Is `goal` reachable from `start` following current edges?
        stack = [start]
        seen: set[int] = set()
        while stack:
            node = stack.pop()
            if node == goal:
                return True
            if node in seen:
                continue
            seen.add(node)
            stack.extend(adjacency.get(node, ()))
        return False

    # Deterministic ordering: weight desc, then (source, target) so equal-weight
    # edges resolve the same way across runs.
    for edge in sorted(
        edges, key=lambda e: (-e.weight, e.source, e.target)
    ):
        # Adding source->target creates a cycle iff target can already reach source.
        if reachable(edge.target, edge.source):
            continue
        adjacency.setdefault(edge.source, set()).add(edge.target)
        kept.append(edge)
    return kept


def _assign_levels(n: int, edges: List[DependencyEdge]) -> List[int]:
    """Longest-path level for each node on the DAG (roots at level 0)."""
    children: Dict[int, List[int]] = {}
    indegree = [0] * n
    for edge in edges:
        children.setdefault(edge.source, []).append(edge.target)
        indegree[edge.target] += 1

    levels = [0] * n
    # Kahn topological order, propagating max level.
    queue = [i for i in range(n) if indegree[i] == 0]
    remaining = indegree[:]
    while queue:
        node = queue.pop()
        for child in children.get(node, ()):
            levels[child] = max(levels[child], levels[node] + 1)
            remaining[child] -= 1
            if remaining[child] == 0:
                queue.append(child)
    return levels


def _normalize_rows(matrix: np.ndarray) -> np.ndarray:
    """Unit-normalise rows so a dot product equals cosine similarity."""
    matrix = np.asarray(matrix, dtype=np.float32)
    if matrix.ndim != 2:
        raise ValueError(f"embeddings must be 2-D, got shape {matrix.shape}.")
    norms = np.linalg.norm(matrix, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    return matrix / norms


def _top_k_neighbours(
    normalized: np.ndarray, row_start: int, row_end: int, top_k: int
) -> Tuple[np.ndarray, np.ndarray]:
    """Top-k most similar rows for a block of rows.

    Returns ``(indices, scores)`` each shaped ``(block, k)``, sorted by
    descending score. Computed block-wise with ``argpartition`` so neither the
    full similarity matrix nor a full per-row sort is ever materialised.
    """
    block = normalized[row_start:row_end]
    sims = block @ normalized.T  # (block, n)
    n = sims.shape[1]
    # +1 because a row's nearest neighbour is always itself.
    k = min(top_k + 1, n)
    part = np.argpartition(-sims, kth=k - 1, axis=1)[:, :k]
    part_scores = np.take_along_axis(sims, part, axis=1)
    order = np.argsort(-part_scores, axis=1)
    return np.take_along_axis(part, order, axis=1), np.take_along_axis(
        part_scores, order, axis=1
    )


def build_dependency_tree(
    embeddings: np.ndarray,
    texts: Sequence[str],
    req_ids: Sequence[str],
    labels: np.ndarray,
    top_k: int = 8,
    sim_threshold: float = 0.45,
    max_edges: int = 2000,
    enable_artifact_near_match: bool = True,
    artifact_near_match_threshold: float = DEFAULT_NEAR_MATCH_THRESHOLD,
) -> Dict[str, Any]:
    """Infer a dependency DAG over requirements.

    Args:
        embeddings: (N, D) original embeddings (used for candidate generation).
        texts: requirement texts, length N.
        req_ids: requirement identifiers. Shorter lists are tolerated; missing
            entries fall back to a positional ``REQ-<i>``.
        labels: cluster labels, length N (-1 = noise). Shorter lists are padded
            with -1.
        top_k: max semantically-similar candidates examined per requirement.
        sim_threshold: minimum cosine similarity for a candidate pair.
        max_edges: cap on returned edges (highest weight retained).
        enable_artifact_near_match: merge synonymous artifact names via embedding
            similarity before Pass 0. Enabled once the harness showed it a net
            win (adversarial detection F1 0.821 -> 0.913 with the aerospace
            set's precision unchanged); pass ``False`` to fall back to strict
            exact-name matching.
        artifact_near_match_threshold: cosine threshold for that merge. See
            :data:`~core.artifact_extractor.DEFAULT_NEAR_MATCH_THRESHOLD`.

    Returns:
        ``{"nodes": [...], "edges": [...], "stats": {...}}`` (JSON-serialisable).
    """
    texts = [t if isinstance(t, str) else "" for t in texts]
    n = len(texts)

    if n == 0:
        return {
            "nodes": [],
            "edges": [],
            "stats": {
                "n_nodes": 0, "n_edges": 0, "max_depth": 0,
                "relation_counts": {}, "root_count": 0,
            },
        }

    # Pad/trim the parallel arrays rather than letting a short list raise deep
    # inside a pass — callers assemble these from several sources.
    safe_req_ids: List[str] = [
        str(req_ids[i]) if i < len(req_ids) and req_ids[i] else f"REQ-{i + 1:03d}"
        for i in range(n)
    ]
    labels_arr = np.asarray(labels).ravel()
    if labels_arr.shape[0] < n:
        labels_arr = np.concatenate(
            [labels_arr, np.full(n - labels_arr.shape[0], -1, dtype=labels_arr.dtype)]
        )

    lowers = [t.lower() for t in texts]
    id_norm = [_normalize_id(r) for r in safe_req_ids]
    id_to_index: Dict[str, int] = {}
    for i, key in enumerate(id_norm):
        if not key:
            continue
        if key in id_to_index:
            logger.warning(
                "build_dependency_tree: duplicate requirement id %r at indices "
                "%d and %d; keeping the first.", safe_req_ids[i], id_to_index[key], i,
            )
            continue
        id_to_index[key] = i
    token_sets = [_salient_tokens(t) for t in texts]

    edges: Dict[tuple[int, int], DependencyEdge] = {}

    # ---- Pass 1: explicit cross-references --------------------------------
    # "as defined in REQ-002" => REQ-002 -> this.
    for i in range(n):
        for match in _REQ_ID_RE.finditer(texts[i]):
            key = _normalize_id(match.group(0))
            j = id_to_index.get(key)
            if j is None or j == i:
                continue
            cue = _first_cue(lowers[i], _HIERARCHICAL_CUES)
            relation = HIERARCHICAL if cue else REFERENCE
            _add_edge(
                edges, j, i, relation, 0.95,
                f"{safe_req_ids[i]} explicitly references {safe_req_ids[j]}.",
                {
                    "pass": PASS_REFERENCE,
                    "matched_text": match.group(0),
                    "text_span": [match.start(), match.end()],
                    "span_in": "target",
                    "hierarchical_cue": cue,
                },
            )

    # ---- Pass 0: producer-consumer artifact matching ----------------------
    # Highest-confidence DATA edges. No embedding-similarity gate: the shared
    # artifact name is itself the match signal.
    all_artifacts = extract_artifacts(texts, safe_req_ids)
    producers, consumers = build_artifact_index(all_artifacts)

    near_match_report: Dict[str, Any] = {}
    rename: Dict[str, Tuple[str, float]] = {}
    if enable_artifact_near_match:
        producers, consumers, near_match_report = resolve_near_matches(
            producers, consumers, similarity_threshold=artifact_near_match_threshold
        )
        rename = {
            m["from"]: (m["to"], m["similarity"])
            for m in near_match_report.get("merges", [])
        }

    # Recover the pre-merge name a given requirement actually used, so the
    # rationale can name the synonym pair rather than only the canonical form.
    original_names: Dict[Tuple[int, str], str] = {}
    for ra in all_artifacts:
        for name in list(ra.produced) + list(ra.consumed):
            canonical = rename.get(name, (name, 0.0))[0]
            original_names.setdefault((ra.req_index, canonical), name)

    n_artifact_pairs_capped = 0
    for artifact_name, producer_indices in producers.items():
        consumer_indices = consumers.get(artifact_name, [])
        if not consumer_indices:
            continue
        if len(producer_indices) * len(consumer_indices) > MAX_PAIRS_PER_ARTIFACT:
            # A name this widely shared carries almost no signal; emitting the
            # full cross-product would swamp the graph.
            n_artifact_pairs_capped += 1
            logger.debug(
                "Pass 0: artifact %r skipped (%d producers x %d consumers "
                "exceeds the %d-pair cap).",
                artifact_name, len(producer_indices), len(consumer_indices),
                MAX_PAIRS_PER_ARTIFACT,
            )
            continue

        for p in producer_indices:
            for c in consumer_indices:
                if p == c:
                    continue
                producer_name = original_names.get((p, artifact_name), artifact_name)
                consumer_name = original_names.get((c, artifact_name), artifact_name)
                evidence: Dict[str, Any] = {
                    "pass": PASS_ARTIFACT,
                    "artifact": artifact_name,
                    "producer_artifact_name": producer_name,
                    "consumer_artifact_name": consumer_name,
                    "synonym_match": producer_name != consumer_name,
                }
                if producer_name != consumer_name:
                    similarity = max(
                        rename.get(producer_name, ("", 0.0))[1],
                        rename.get(consumer_name, ("", 0.0))[1],
                    )
                    evidence["artifact_similarity"] = round(float(similarity), 4)
                    rationale = (
                        f"{safe_req_ids[p]} produces '{producer_name}', matched to "
                        f"'{consumer_name}' consumed by {safe_req_ids[c]} "
                        f"(synonym match, similarity {similarity:.2f})."
                    )
                else:
                    rationale = (
                        f"{safe_req_ids[p]} produces '{artifact_name}' "
                        f"consumed by {safe_req_ids[c]}."
                    )
                _add_edge(edges, p, c, DATA, 0.90, rationale, evidence)

    # ---- Pass 2: semantic candidate pairs scored by lexical signals -------
    if n >= 2:
        normalized = _normalize_rows(embeddings)
        if normalized.shape[0] != n:
            raise ValueError(
                f"embeddings has {normalized.shape[0]} rows but {n} texts were given."
            )

        for row_start in range(0, n, _SIM_BLOCK):
            row_end = min(row_start + _SIM_BLOCK, n)
            neighbour_idx, neighbour_sim = _top_k_neighbours(
                normalized, row_start, row_end, top_k
            )

            for offset in range(row_end - row_start):
                i = row_start + offset
                examined = 0
                for slot in range(neighbour_idx.shape[1]):
                    j = int(neighbour_idx[offset, slot])
                    if j == i:
                        continue
                    s = float(neighbour_sim[offset, slot])
                    if s < sim_threshold:
                        break
                    examined += 1
                    if examined > top_k:
                        break

                    shared = token_sets[i] & token_sets[j]
                    if not shared:
                        continue
                    # Longest shared token, ties broken alphabetically so the
                    # rationale text is stable across runs.
                    shared_term = sorted(shared, key=lambda w: (-len(w), w))[0]

                    # 2b. Sequential: j is gated by a pre-condition cue => i -> j.
                    cue = _first_cue(lowers[j], _SEQUENTIAL_CUES)
                    if cue and cue not in _REVERSED_SEQUENTIAL_CUES:
                        _add_edge(
                            edges, i, j, SEQUENTIAL, 0.45 + 0.4 * s,
                            f"{safe_req_ids[j]} occurs '{cue}' a precondition shared "
                            f"with {safe_req_ids[i]} ('{shared_term}').",
                            {
                                "pass": PASS_SEQUENTIAL,
                                "cue": cue,
                                "cosine_similarity": round(s, 4),
                                "shared_tokens": sorted(shared)[:10],
                                "primary_shared_token": shared_term,
                                "span_in": "target",
                            },
                        )

                    # 2c. Hierarchical: same cluster, i more general than j.
                    if (
                        labels_arr[i] != -1
                        and labels_arr[i] == labels_arr[j]
                        and len(token_sets[i]) + 1 < len(token_sets[j])
                        and s >= max(sim_threshold, 0.6)
                    ):
                        _add_edge(
                            edges, i, j, HIERARCHICAL, 0.4 + 0.3 * s,
                            f"{safe_req_ids[i]} is broader than {safe_req_ids[j]} "
                            f"within the same cluster.",
                            {
                                "pass": PASS_HIERARCHICAL,
                                "cosine_similarity": round(s, 4),
                                "same_cluster": True,
                                "cluster_id": int(labels_arr[i]),
                                "source_token_count": len(token_sets[i]),
                                "target_token_count": len(token_sets[j]),
                                "shared_tokens": sorted(shared)[:10],
                            },
                        )

    edge_list = list(edges.values())

    # Cap *before* cycle-breaking: the reachability check is the expensive step,
    # and truncating afterwards would pay for edges that are then discarded.
    if len(edge_list) > max_edges:
        edge_list = sorted(
            edge_list, key=lambda e: (-e.weight, e.source, e.target)
        )[:max_edges]
    edge_list = _break_cycles(n, edge_list)

    levels = _assign_levels(n, edge_list)

    nodes = [
        DependencyNode(
            id=i,
            node_id=safe_req_ids[i],
            requirement_text=texts[i],
            cluster_id=int(labels_arr[i]),
            level=levels[i],
        ).to_dict()
        for i in range(n)
    ]
    edges_out = [e.to_dict() for e in edge_list]

    relation_counts: Dict[str, int] = {}
    tier_counts: Dict[str, int] = {}
    source_nodes = set()
    for e in edge_list:
        relation_counts[e.relation] = relation_counts.get(e.relation, 0) + 1
        tier = e.confidence_tier()
        tier_counts[tier] = tier_counts.get(tier, 0) + 1
        source_nodes.add(e.source)

    roots = [i for i in range(n) if levels[i] == 0 and i in source_nodes]

    stats: Dict[str, Any] = {
        "n_nodes": n,
        "n_edges": len(edges_out),
        "max_depth": max(levels) if levels else 0,
        "relation_counts": relation_counts,
        "confidence_tier_counts": tier_counts,
        "root_count": len(roots),
        "artifact_near_match_enabled": bool(enable_artifact_near_match),
    }
    if n_artifact_pairs_capped:
        stats["artifacts_skipped_fanout_cap"] = n_artifact_pairs_capped
    if near_match_report:
        stats["artifact_near_match"] = {
            "n_merges": len(near_match_report.get("merges", [])),
            "skipped": near_match_report.get("skipped"),
            "merges": near_match_report.get("merges", [])[:50],
        }

    return {"nodes": nodes, "edges": edges_out, "stats": stats}
