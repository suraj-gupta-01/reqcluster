from pydantic import BaseModel
from typing import Optional, List, Any
from datetime import datetime


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

    class Config:
        from_attributes = True


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

    class Config:
        from_attributes = True


class ClusterOut(BaseModel):
    id: int
    session_id: int
    cluster_id: int
    label: str
    keywords: Optional[List[str]]
    size: int

    class Config:
        from_attributes = True


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
    session_id: int
    min_cluster_size: Optional[int] = None
    min_samples: Optional[int] = 3
    similarity_threshold: Optional[float] = 0.65


class ClusterResponse(BaseModel):
    session_id: int
    total_clusters: int
    noise_count: int
    clusters: List[ClusterOut]
    status: str


class ProgressUpdate(BaseModel):
    step: str
    progress: int
    message: str
