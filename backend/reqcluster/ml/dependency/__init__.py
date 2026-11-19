"""Requirement dependency inference: artifact extraction + tree builder."""
from .extract import ReqArtifacts, build_artifact_index, extract_artifacts, resolve_near_matches, spacy_available
from .tree import DependencyEdge, DependencyNode, DependencyTree, build

__all__ = [
    "DependencyEdge", "DependencyNode", "DependencyTree", "ReqArtifacts", "build",
    "build_artifact_index", "extract_artifacts", "resolve_near_matches", "spacy_available",
]
