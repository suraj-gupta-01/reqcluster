"""Domain types shared across layers."""
from dataclasses import dataclass
from datetime import datetime
from typing import Annotated, Literal

import numpy as np
from pydantic import BaseModel, ConfigDict, Field

ProviderName = Literal["mock", "openai", "local"]
EmbeddingMode = Literal["base", "enriched", "hybrid"]
JobKind = Literal["cluster", "recluster", "enrich", "suggest", "dependencies"]
JobStatus = Literal["queued", "running", "succeeded", "failed", "cancelled"]
SessionStatus = Literal["uploaded", "queued", "running", "done", "error"]

NOISE: int = -1

# SQLite integers are signed 64-bit; reject anything larger at the HTTP boundary (422, not 500).
DbId = Annotated[int, Field(ge=1, le=2**63 - 1)]
ClusterRef = Annotated[int, Field(ge=-1, le=2**63 - 1)]  # -1 = noise


class RequirementIn(BaseModel):
    req_id: str
    text: str
    ordinal: int
    attributes: dict[str, str] = {}


class ClusteringParams(BaseModel):
    model_config = ConfigDict(extra="forbid")

    min_cluster_size: int | None = Field(default=None, ge=2)  # None = auto
    min_samples: int = Field(default=3, ge=1)
    sim_threshold: float = Field(default=0.5, ge=0.0, le=1.0)
    top_k: int = Field(default=10, ge=1)
    embedding_mode: EmbeddingMode = "base"
    seed: int | None = None
    enable_embedding_comparison: bool = False
    run_ablation: bool = False


class ClusterInfo(BaseModel):
    cluster_id: int
    size: int
    label: str
    keywords: list[str]


@dataclass
class ClusteringResult:
    labels: np.ndarray  # int64
    probabilities: np.ndarray  # float32
    x10: np.ndarray  # float32
    xy: np.ndarray  # float32 (N, 2)
    clusters: list[ClusterInfo]
    edges: list[tuple[int, int, float]]
    warnings: list[str]
    durations: dict[str, float]


class Job(BaseModel):
    id: str
    session_id: int
    kind: JobKind
    status: JobStatus
    progress: float
    step: str
    message: str | None
    result: dict | None
    error: str | None
    created_at: datetime
    started_at: datetime | None
    finished_at: datetime | None
