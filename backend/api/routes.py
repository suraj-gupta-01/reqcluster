from fastapi import APIRouter, UploadFile, File, HTTPException, Depends
from fastapi.concurrency import run_in_threadpool
from fastapi.responses import JSONResponse
from sqlalchemy.orm import Session as DBSession
from typing import List, Optional
import asyncio
import json
import logging

from models.database import get_db, Session, Requirement, Cluster, Graph
from models.schemas import (
    UploadResponse, ClusterRequest, ClusterResponse,
    SessionOut, RequirementOut, ClusterOut, ClusterDetail, GraphOut
)
from core.preprocessing import preprocess_requirements
from core.pipeline import run_pipeline

logger = logging.getLogger(__name__)
router = APIRouter()

# In-memory progress store
pipeline_progress: dict = {}

# Reject uploads larger than this to avoid loading huge files into memory.
MAX_UPLOAD_BYTES = 25 * 1024 * 1024  # 25 MB


@router.post("/upload", response_model=UploadResponse)
async def upload_requirements(
    file: UploadFile = File(...),
    db: DBSession = Depends(get_db),
):
    """Upload and preprocess a CSV or XLSX requirements file."""
    if not file.filename or not file.filename.endswith((".csv", ".xlsx", ".xls")):
        raise HTTPException(400, "Only CSV and XLSX files are supported.")

    content = await file.read()

    if len(content) > MAX_UPLOAD_BYTES:
        raise HTTPException(
            413, f"File too large. Maximum size is {MAX_UPLOAD_BYTES // (1024 * 1024)} MB."
        )

    try:
        df, stats = preprocess_requirements(content, file.filename)
    except ValueError as e:
        raise HTTPException(400, str(e))
    except Exception as e:
        logger.exception("Preprocessing error")
        raise HTTPException(500, f"Failed to process file: {str(e)}")

    if stats["final"] == 0:
        raise HTTPException(400, "No valid requirements found in file.")

    # Create session
    session = Session(
        name=file.filename.rsplit(".", 1)[0],
        filename=file.filename,
        status="uploaded",
        total_requirements=stats["final"],
    )
    db.add(session)
    db.commit()
    db.refresh(session)

    # Store requirements
    for _, row in df.iterrows():
        req = Requirement(
            session_id=session.id,
            req_id=str(row.get("id", "")),
            text=str(row["text"]),
            module=str(row.get("module", "")),
            section=str(row.get("section", "")),
        )
        db.add(req)
    db.commit()

    return UploadResponse(
        session_id=session.id,
        filename=file.filename,
        total_requirements=stats["final"],
        duplicates_removed=stats["duplicate_removed"],
        empty_removed=stats["empty_removed"],
        status="uploaded",
    )


@router.post("/cluster", response_model=ClusterResponse)
async def cluster_requirements_endpoint(
    request: ClusterRequest,
    db: DBSession = Depends(get_db),
):
    """Run the clustering pipeline on an uploaded session."""
    session = db.query(Session).filter(Session.id == request.session_id).first()
    if not session:
        raise HTTPException(404, f"Session {request.session_id} not found.")

    if session.status == "processing":
        raise HTTPException(409, "Clustering already in progress for this session.")

    # Load requirements
    reqs = db.query(Requirement).filter(
        Requirement.session_id == request.session_id
    ).all()

    if not reqs:
        raise HTTPException(400, "No requirements found for this session.")

    texts = [r.text for r in reqs]
    req_ids = [r.req_id for r in reqs]

    session.status = "processing"
    db.commit()

    # Progress tracker
    session_id = request.session_id
    pipeline_progress[session_id] = {"step": "starting", "progress": 0, "message": "Initializing..."}

    def progress_callback(step: str, pct: int, msg: str):
        pipeline_progress[session_id] = {"step": step, "progress": pct, "message": msg}

    try:
        # run_pipeline is CPU-bound (SBERT/UMAP/HDBSCAN). Run it in a worker
        # thread so the event loop stays free to serve progress polling.
        results = await run_in_threadpool(
            run_pipeline,
            texts=texts,
            req_ids=req_ids,
            min_cluster_size=request.min_cluster_size,
            min_samples=request.min_samples or 3,
            similarity_threshold=request.similarity_threshold or 0.65,
            progress_callback=progress_callback,
        )

        labels = results["labels"]
        probabilities = results["probabilities"]
        embeddings_2d = results["embeddings_2d"]
        cluster_info = results["cluster_info"]
        graph_data = results["graph_data"]

        # Clear old cluster data for this session
        db.query(Cluster).filter(Cluster.session_id == session_id).delete()
        db.query(Graph).filter(Graph.session_id == session_id).delete()

        # Update requirements with cluster assignments. `reqs` is already
        # loaded and ordered identically to texts/labels, so mutate in place
        # instead of re-querying each row.
        for i, req in enumerate(reqs):
            req.cluster_id = int(labels[i])
            req.membership_prob = float(probabilities[i])
            req.umap_x = float(embeddings_2d[i, 0])
            req.umap_y = float(embeddings_2d[i, 1])
            req.is_noise = bool(labels[i] == -1)

        # Store clusters
        cluster_db_objs = []
        for cluster_id, info in cluster_info.items():
            c = Cluster(
                session_id=session_id,
                cluster_id=cluster_id,
                label=info["label"],
                keywords=info["keywords"],
                size=info["size"],
            )
            db.add(c)
            cluster_db_objs.append(c)

        # Store graph
        g = Graph(
            session_id=session_id,
            nodes=graph_data["nodes"],
            edges=graph_data["edges"],
        )
        db.add(g)

        # Update session
        session.status = "done"
        session.total_clusters = results["n_clusters"]
        session.noise_count = results["noise_count"]
        db.commit()

        db.refresh(session)
        clusters_out = db.query(Cluster).filter(Cluster.session_id == session_id).all()

        pipeline_progress[session_id] = {"step": "done", "progress": 100, "message": "Complete!"}

        return ClusterResponse(
            session_id=session_id,
            total_clusters=results["n_clusters"],
            noise_count=results["noise_count"],
            clusters=[
                ClusterOut(
                    id=c.id,
                    session_id=c.session_id,
                    cluster_id=c.cluster_id,
                    label=c.label,
                    keywords=c.keywords,
                    size=c.size,
                )
                for c in clusters_out
            ],
            status="done",
        )

    except Exception as e:
        logger.exception("Pipeline error")
        session.status = "error"
        db.commit()
        pipeline_progress[session_id] = {"step": "error", "progress": 0, "message": str(e)}
        raise HTTPException(500, f"Pipeline failed: {str(e)}")


@router.get("/progress/{session_id}")
async def get_progress(session_id: int):
    """Get pipeline progress for a session."""
    progress = pipeline_progress.get(session_id, {"step": "idle", "progress": 0, "message": "Not started"})
    return progress


@router.get("/sessions", response_model=List[SessionOut])
def list_sessions(db: DBSession = Depends(get_db)):
    sessions = db.query(Session).order_by(Session.created_at.desc()).all()
    return sessions


@router.get("/sessions/{session_id}", response_model=SessionOut)
def get_session(session_id: int, db: DBSession = Depends(get_db)):
    session = db.query(Session).filter(Session.id == session_id).first()
    if not session:
        raise HTTPException(404, "Session not found")
    return session


@router.get("/clusters", response_model=List[ClusterOut])
def get_clusters(session_id: int, db: DBSession = Depends(get_db)):
    """Get all clusters for a session."""
    clusters = db.query(Cluster).filter(
        Cluster.session_id == session_id
    ).order_by(Cluster.size.desc()).all()
    return clusters


@router.get("/cluster/{cluster_id}", response_model=ClusterDetail)
def get_cluster_detail(
    cluster_id: int,
    session_id: int,
    db: DBSession = Depends(get_db),
):
    """Get cluster details with requirements."""
    cluster = db.query(Cluster).filter(
        Cluster.session_id == session_id,
        Cluster.cluster_id == cluster_id,
    ).first()
    if not cluster:
        raise HTTPException(404, f"Cluster {cluster_id} not found")

    reqs = db.query(Requirement).filter(
        Requirement.session_id == session_id,
        Requirement.cluster_id == cluster_id,
    ).all()

    return ClusterDetail(
        cluster=ClusterOut(
            id=cluster.id,
            session_id=cluster.session_id,
            cluster_id=cluster.cluster_id,
            label=cluster.label,
            keywords=cluster.keywords,
            size=cluster.size,
        ),
        requirements=[
            RequirementOut(
                id=r.id,
                session_id=r.session_id,
                req_id=r.req_id,
                text=r.text,
                module=r.module,
                section=r.section,
                cluster_id=r.cluster_id,
                membership_prob=r.membership_prob,
                umap_x=r.umap_x,
                umap_y=r.umap_y,
                is_noise=r.is_noise,
            )
            for r in reqs
        ],
    )


@router.get("/graph", response_model=GraphOut)
def get_graph(session_id: int, db: DBSession = Depends(get_db)):
    """Get similarity graph for a session."""
    graph = db.query(Graph).filter(Graph.session_id == session_id).first()
    if not graph:
        raise HTTPException(404, "Graph not found. Run clustering first.")
    return GraphOut(nodes=graph.nodes, edges=graph.edges)


@router.get("/requirements", response_model=List[RequirementOut])
def get_requirements(
    session_id: int,
    cluster_id: Optional[int] = None,
    db: DBSession = Depends(get_db),
):
    """Get requirements for a session, optionally filtered by cluster."""
    query = db.query(Requirement).filter(Requirement.session_id == session_id)
    if cluster_id is not None:
        query = query.filter(Requirement.cluster_id == cluster_id)
    reqs = query.all()
    return reqs
