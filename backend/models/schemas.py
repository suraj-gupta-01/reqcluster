from pydantic import BaseModel, Field, ConfigDict
from typing import Optional, List, Any, Literal, Dict
from datetime import datetime
import os

def get_default_provider() -> str:
    prov = os.getenv("REQCLUSTER_LLM_PROVIDER", "mock").strip().lower()
    if prov in {"openai", "openai-compatible"}:
        return "openai_compatible"
    return prov



class SessionCreate(BaseModel):
    name: str


class SessionOut(BaseModel):
    id: int
    name: str
    filename: str
    created_at: datetime
    status: str
    total_requirements: int
    total_clusters: int
    noise_count: int

    model_config = ConfigDict(from_attributes=True)


class RequirementOut(BaseModel):
    id: int
    session_id: int
    req_id: Optional[str]
    text: str
    module: Optional[str]
    section: Optional[str]
    cluster_id: Optional[int]
    membership_prob: Optional[float]
    umap_x: Optional[float]
    umap_y: Optional[float]
    is_noise: bool

    model_config = ConfigDict(from_attributes=True)


class ClusterOut(BaseModel):
    id: int
    session_id: int
    cluster_id: int
    label: str
    keywords: Optional[List[str]]
    size: int

    model_config = ConfigDict(from_attributes=True)


class ClusterDetail(BaseModel):
    cluster: ClusterOut
    requirements: List[RequirementOut]


class GraphOut(BaseModel):
    nodes: List[Any]
    edges: List[Any]


class UploadResponse(BaseModel):
    session_id: int
    filename: str
    total_requirements: int
    duplicates_removed: int
    empty_removed: int
    status: str


class ClusterRequest(BaseModel):
    session_id: int = Field(..., ge=1)
    min_cluster_size: Optional[int] = Field(default=None, ge=2, le=10000)
    min_samples: Optional[int] = Field(default=3, ge=1, le=1000)
    similarity_threshold: Optional[float] = Field(default=0.65, ge=0.0, le=1.0)
    embedding_mode: Literal["base", "enriched", "hybrid"] = "base"
    enable_embedding_comparison: bool = False
    run_ablation: bool = False


class EnrichmentRequest(BaseModel):
    session_id: int = Field(..., ge=1)
    provider_name: Literal["mock", "openai_compatible", "local"] = Field(
        default_factory=get_default_provider
    )
    embedding_mode: Literal["enriched", "hybrid"] = "hybrid"
    batch_size: int = Field(default=8, ge=1, le=64)
    max_concurrency: int = Field(default=4, ge=1, le=16)
    timeout_seconds: float = Field(default=30, ge=1, le=120)
    force_refresh: bool = False
    fail_fast: bool = False
    use_cache: bool = True


class EnrichmentResponse(BaseModel):
    session_id: int
    status: str
    total: int
    succeeded: int
    failed: int
    provider: Optional[str]
    model: Optional[str]
    prompt_version: Optional[str]
    domain_vocabulary: List[str]
    quality_report: Dict[str, Any]
    warnings: List[str]
    duration_ms: float


class EnrichmentStatusResponse(BaseModel):
    session_id: int
    status: str
    total: int
    succeeded: int
    failed: int
    pending: int
    latest_run_created_at: Optional[datetime]
    provider: Optional[str]
    model: Optional[str]
    warnings: List[str]


class EnrichmentResultOut(BaseModel):
    requirement_id: Optional[str]
    expanded_text: Optional[str]
    domain_terms: List[str]
    functional_intent: Optional[str]
    mentioned_components: List[str]
    assumptions: List[str]
    confidence: Optional[float]
    warnings: List[str]
    quality_report: Optional[Dict[str, Any]]
    status: str


class ClusterResponse(BaseModel):
    session_id: int
    total_clusters: Optional[int] = None
    noise_count: Optional[int] = None
    clusters: Optional[List[ClusterOut]] = None
    status: str
    embedding_mode: Optional[Literal["base", "enriched", "hybrid"]] = None
    warnings: Optional[List[str]] = None
    embedding_comparison: Optional[Dict[str, Any]] = None
    ablation_report: Optional[Dict[str, Any]] = None



class ProgressUpdate(BaseModel):
    step: str
    progress: int
    message: str


# --- DP5: Dependency tree + rationale schemas ---


class DependencyGenerateRequest(BaseModel):
    session_id: int = Field(..., ge=1)
    provider_name: str = Field(default_factory=get_default_provider)
    sim_threshold: float = Field(default=0.45, ge=0.0, le=1.0)
    top_k: int = Field(default=8, ge=1, le=50)


class DependencyResponse(BaseModel):
    session_id: int
    nodes: List[Any]
    edges: List[Any]
    stats: Dict[str, Any]
    rationale: Dict[str, Any]


# --- Phase 5: Active learning schemas ---


class ConstrainedClusterRequest(BaseModel):
    session_id: int = Field(..., ge=1)


class ConstrainedClusterResponse(BaseModel):
    session_id: int
    iteration: int
    constraints: Dict[str, Any]
    quality: Dict[str, Any]


class UncertaintyQueueResponse(BaseModel):
    session_id: int
    queue: List[Dict[str, Any]]
    count: int


class QualityHistoryResponse(BaseModel):
    session_id: int
    history: List[Dict[str, Any]]
    count: int


# --- Phase 3: Refinement schemas ---


class RefinementSuggestRequest(BaseModel):
    session_id: int = Field(..., ge=1)
    provider_name: str = Field(default_factory=get_default_provider)
    top_n_merges: int = Field(default=5, ge=1, le=20)
    top_n_splits: int = Field(default=5, ge=1, le=20)
    sim_threshold: float = Field(default=0.75, ge=0.0, le=1.0)
    spread_threshold: float = Field(default=0.3, ge=0.0, le=1.0)


class RepresentativeOut(BaseModel):
    req_id: Optional[str]
    text: str
    similarity_to_centroid: float
    rank: int


class RefinementSuggestionOut(BaseModel):
    id: int
    session_id: int
    suggestion_type: str
    status: str
    cluster_a_id: Optional[int] = None
    cluster_b_id: Optional[int] = None
    cluster_id: Optional[int] = None
    cluster_a_label: Optional[str] = None
    cluster_b_label: Optional[str] = None
    cluster_label: Optional[str] = None
    similarity_score: Optional[float] = None
    silhouette_delta: Optional[float] = None
    coherence_score: Optional[float] = None
    spread_score: Optional[float] = None
    bimodality_score: Optional[float] = None
    rationale: Optional[str] = None
    summary: Optional[str] = None
    representative_req_ids: List[str] = []
    sub_cluster_sizes: List[int] = []
    created_at: Optional[datetime] = None

    model_config = ConfigDict(from_attributes=True)


class CoherenceResultOut(BaseModel):
    cluster_id: int
    coherence_score: float
    top_keywords: List[str]
    size: int
    assessment: str


class ClusterSummaryOut(BaseModel):
    cluster_id: int
    summary: str
    representative_count: int


class RefinementSuggestResponse(BaseModel):
    session_id: int
    merge_suggestions: List[RefinementSuggestionOut]
    split_suggestions: List[RefinementSuggestionOut]
    coherence_scores: List[CoherenceResultOut]
    cluster_summaries: List[ClusterSummaryOut]
    warnings: List[str]


class ApplySuggestionRequest(BaseModel):
    session_id: int = Field(..., ge=1)
    suggestion_id: int = Field(..., ge=1)
    action: Literal["accept", "reject"]


class ApplySuggestionResponse(BaseModel):
    suggestion_id: int
    action: str
    status: str
    affected_clusters: List[int]
    message: str


class AuditLogEntry(BaseModel):
    id: int
    session_id: int
    suggestion_id: int
    action: str
    before_state: Optional[Dict[str, Any]] = None
    after_state: Optional[Dict[str, Any]] = None
    applied_by: Optional[str] = None
    created_at: Optional[datetime] = None


class FeedbackSubmitRequest(BaseModel):
    session_id: int = Field(..., ge=1)
    requirement_id: int = Field(..., ge=1)
    new_cluster_id: Optional[int] = None
    confidence_score: float = Field(1.0, ge=0.0, le=1.0)
    comments: Optional[str] = None
    applied_by: Optional[str] = None


class FeedbackCorrectionOut(BaseModel):
    id: int
    session_id: int
    requirement_id: int
    previous_cluster_id: Optional[int] = None
    new_cluster_id: Optional[int] = None
    confidence_score: float
    comments: Optional[str] = None
    applied_by: Optional[str] = None
    status: str
    created_at: datetime
    updated_at: datetime

    model_config = ConfigDict(from_attributes=True)


class FeedbackReviewRequest(BaseModel):
    session_id: int = Field(..., ge=1)
    feedback_id: int = Field(..., ge=1)
    status: Literal["approved", "rejected"]


class ConstraintPairOut(BaseModel):
    id: int
    session_id: int
    requirement_a_id: int
    requirement_b_id: int
    constraint_type: str
    feedback_id: int
    created_at: datetime

    model_config = ConfigDict(from_attributes=True)


