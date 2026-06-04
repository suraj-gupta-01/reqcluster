from fastapi import APIRouter, UploadFile, File, HTTPException, Depends, Path, Query, Response
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
    SessionOut, RequirementOut, ClusterOut, ClusterDetail, GraphOut,
    EnrichmentRequest, EnrichmentResponse, EnrichmentStatusResponse,
    EnrichmentResultOut,
    RefinementSuggestRequest, RefinementSuggestResponse,
    RefinementSuggestionOut, ApplySuggestionRequest, ApplySuggestionResponse,
    AuditLogEntry,
    FeedbackSubmitRequest, FeedbackCorrectionOut, FeedbackReviewRequest,
    ConstraintPairOut,
    DependencyGenerateRequest, DependencyResponse,
)
from core.preprocessing import preprocess_requirements
from core.pipeline import run_pipeline
from services.enrichment_service import (
    EnrichmentServiceError,
    build_enriched_texts_for_pipeline,
    get_enrichment_results,
    get_enrichment_status,
    run_and_persist_enrichment,
)
from services.refinement_service import (
    RefinementServiceError,
    generate_and_persist_suggestions,
    apply_suggestion,
    get_suggestions,
    get_audit_log,
)
from services.feedback_service import (
    submit_feedback,
    review_feedback,
    get_feedback_queue,
    export_feedback_csv,
    export_feedback_json,
)
from core.feedback_bridge import detect_constraints_conflicts
from services.dependency_service import (
    DependencyServiceError,
    generate_and_persist_dependencies,
    get_dependencies,
)

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


@router.post("/enrich", response_model=EnrichmentResponse)
async def enrich_requirements_endpoint(
    request: EnrichmentRequest,
    db: DBSession = Depends(get_db),
):
    """Run Phase 2 requirement enrichment and persist results for a session."""
    session_id = request.session_id
    pipeline_progress[session_id] = {
        "step": "loading_requirements",
        "progress": 0,
        "message": "Starting enrichment...",
    }

    def progress_callback(step: str, pct: int, msg: str):
        pipeline_progress[session_id] = {"step": step, "progress": pct, "message": msg}

    try:
        return await run_and_persist_enrichment(
            db,
            request,
            progress_callback=progress_callback,
        )
    except EnrichmentServiceError as exc:
        pipeline_progress[session_id] = {
            "step": "failed",
            "progress": 0,
            "message": exc.message,
        }
        raise HTTPException(exc.status_code, exc.message)
    except Exception:
        logger.exception("Enrichment error")
        pipeline_progress[session_id] = {
            "step": "failed",
            "progress": 0,
            "message": "Enrichment failed.",
        }
        raise HTTPException(500, "Enrichment failed.")


@router.get("/enrich/status/{session_id}", response_model=EnrichmentStatusResponse)
def get_enrichment_status_endpoint(
    session_id: int = Path(..., ge=1),
    db: DBSession = Depends(get_db),
):
    try:
        return get_enrichment_status(db, session_id)
    except EnrichmentServiceError as exc:
        raise HTTPException(exc.status_code, exc.message)


@router.get("/enrich/results", response_model=List[EnrichmentResultOut])
def get_enrichment_results_endpoint(
    session_id: int = Query(..., ge=1),
    db: DBSession = Depends(get_db),
):
    try:
        return get_enrichment_results(db, session_id)
    except EnrichmentServiceError as exc:
        raise HTTPException(exc.status_code, exc.message)


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
    ).order_by(Requirement.id.asc()).all()

    if not reqs:
        raise HTTPException(400, "No requirements found for this session.")

    texts = [r.text for r in reqs]
    req_ids = [r.req_id for r in reqs]
    enriched_texts = None
    if request.embedding_mode != "base":
        try:
            enriched_texts = build_enriched_texts_for_pipeline(
                db,
                request.session_id,
                request.embedding_mode,
            )
        except EnrichmentServiceError as exc:
            raise HTTPException(exc.status_code, exc.message)

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
            min_samples=request.min_samples if request.min_samples is not None else 3,
            similarity_threshold=(
                request.similarity_threshold
                if request.similarity_threshold is not None
                else 0.65
            ),
            progress_callback=progress_callback,
            embedding_mode=request.embedding_mode,
            enriched_texts=enriched_texts,
            enable_embedding_comparison=request.enable_embedding_comparison,
            run_ablation=request.run_ablation,
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
            embedding_mode=results.get("embedding_mode"),
            warnings=results.get("warnings"),
            embedding_comparison=results.get("embedding_comparison"),
            ablation_report=results.get("ablation_report"),
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


# --- Phase 3: Refinement endpoints ---


@router.post("/suggestions/generate", response_model=RefinementSuggestResponse)
async def generate_suggestions_endpoint(
    request: RefinementSuggestRequest,
    db: DBSession = Depends(get_db),
):
    """Analyze clusters and generate merge/split refinement suggestions."""
    try:
        return await run_in_threadpool(generate_and_persist_suggestions, db, request)
    except RefinementServiceError as exc:
        raise HTTPException(exc.status_code, exc.message)
    except Exception:
        logger.exception("Refinement suggestion generation error")
        raise HTTPException(500, "Failed to generate refinement suggestions.")


@router.get("/suggestions", response_model=List[RefinementSuggestionOut])
def list_suggestions_endpoint(
    session_id: int = Query(..., ge=1),
    status: Optional[str] = Query(default=None),
    db: DBSession = Depends(get_db),
):
    """List refinement suggestions for a session, optionally filtered by status."""
    try:
        return get_suggestions(db, session_id, status)
    except RefinementServiceError as exc:
        raise HTTPException(exc.status_code, exc.message)


@router.post("/suggestions/apply", response_model=ApplySuggestionResponse)
async def apply_suggestion_endpoint(
    request: ApplySuggestionRequest,
    db: DBSession = Depends(get_db),
):
    """Accept or reject a refinement suggestion."""
    try:
        return apply_suggestion(db, request)
    except RefinementServiceError as exc:
        raise HTTPException(exc.status_code, exc.message)
    except Exception:
        logger.exception("Suggestion apply error")
        raise HTTPException(500, "Failed to apply suggestion.")


@router.get("/suggestions/audit", response_model=List[AuditLogEntry])
def get_audit_log_endpoint(
    session_id: int = Query(..., ge=1),
    db: DBSession = Depends(get_db),
):
    """Get audit log of applied refinements for a session."""
    try:
        return get_audit_log(db, session_id)
    except RefinementServiceError as exc:
        raise HTTPException(exc.status_code, exc.message)


# --- Phase 4: Human-in-the-Loop endpoints ---


@router.post("/feedback/submit", response_model=FeedbackCorrectionOut)
def submit_feedback_endpoint(
    request: FeedbackSubmitRequest,
    db: DBSession = Depends(get_db),
):
    """Submit a manual cluster correction for a requirement."""
    try:
        return submit_feedback(db, request)
    except ValueError as exc:
        raise HTTPException(400, str(exc))
    except Exception as exc:
        logger.exception("Feedback submit error")
        raise HTTPException(500, f"Failed to submit feedback: {str(exc)}")


@router.get("/feedback/queue", response_model=List[FeedbackCorrectionOut])
def get_feedback_queue_endpoint(
    session_id: int = Query(..., ge=1),
    status: Optional[str] = Query(default=None),
    db: DBSession = Depends(get_db),
):
    """Retrieve the human feedback review queue for a session."""
    try:
        return get_feedback_queue(db, session_id, status)
    except Exception as exc:
        logger.exception("Get feedback queue error")
        raise HTTPException(500, "Failed to retrieve feedback queue.")


@router.post("/feedback/review", response_model=FeedbackCorrectionOut)
def review_feedback_endpoint(
    request: FeedbackReviewRequest,
    db: DBSession = Depends(get_db),
):
    """Approve or reject a pending cluster correction."""
    try:
        return review_feedback(db, request.session_id, request.feedback_id, request.status)
    except ValueError as exc:
        raise HTTPException(400, str(exc))
    except Exception as exc:
        logger.exception("Review feedback error")
        raise HTTPException(500, "Failed to review feedback.")


@router.get("/feedback/constraints")
def get_constraints_endpoint(
    session_id: int = Query(..., ge=1),
    db: DBSession = Depends(get_db),
):
    """Get active constraint pairs and detect conflicts in the constraint network."""
    try:
        from models.database import ConstraintPair, FeedbackCorrection
        # Fetch active constraints (where status != rejected)
        pairs = (
            db.query(ConstraintPair)
            .join(FeedbackCorrection, ConstraintPair.feedback_id == FeedbackCorrection.id)
            .filter(
                ConstraintPair.session_id == session_id,
                FeedbackCorrection.status != "rejected"
            )
            .all()
        )
        serialized_pairs = [
            {
                "id": p.id,
                "session_id": p.session_id,
                "requirement_a_id": p.requirement_a_id,
                "requirement_b_id": p.requirement_b_id,
                "constraint_type": p.constraint_type,
                "feedback_id": p.feedback_id,
                "created_at": p.created_at,
            }
            for p in pairs
        ]
        conflicts = detect_constraints_conflicts(db, session_id)
        return {
            "session_id": session_id,
            "constraint_pairs": serialized_pairs,
            "conflicts": conflicts,
            "has_conflicts": len(conflicts) > 0,
        }
    except Exception as exc:
        logger.exception("Get constraints error")
        raise HTTPException(500, "Failed to retrieve constraints and conflicts.")


@router.get("/feedback/export")
def export_feedback_endpoint(
    session_id: int = Query(..., ge=1),
    format: str = Query(default="csv"),
    db: DBSession = Depends(get_db),
):
    """Export the human feedback queue as CSV or JSON."""
    try:
        fmt = format.strip().lower()
        if fmt == "csv":
            csv_content = export_feedback_csv(db, session_id)
            return Response(
                content=csv_content,
                media_type="text/csv",
                headers={"Content-Disposition": f"attachment; filename=feedback_session_{session_id}.csv"}
            )
        elif fmt == "json":
            json_content = export_feedback_json(db, session_id)
            return Response(
                content=json_content,
                media_type="application/json",
                headers={"Content-Disposition": f"attachment; filename=feedback_session_{session_id}.json"}
            )
        else:
            raise HTTPException(400, "Unsupported export format. Use 'csv' or 'json'.")
    except HTTPException:
        raise
    except Exception as exc:
        logger.exception("Feedback export error")
        raise HTTPException(500, "Failed to export feedback data.")


# --- DP5: Dependency tree + rationale endpoints ---


@router.post("/dependencies/generate", response_model=DependencyResponse)
async def generate_dependencies_endpoint(
    request: DependencyGenerateRequest,
    db: DBSession = Depends(get_db),
):
    """Infer the requirement dependency tree and rationale document for a session."""
    try:
        return await run_in_threadpool(
            generate_and_persist_dependencies,
            db,
            request.session_id,
            request.provider_name,
            request.sim_threshold,
            request.top_k,
        )
    except DependencyServiceError as exc:
        raise HTTPException(exc.status_code, exc.message)
    except Exception:
        logger.exception("Dependency generation error")
        raise HTTPException(500, "Failed to generate dependency tree.")


@router.get("/dependencies", response_model=DependencyResponse)
def get_dependencies_endpoint(
    session_id: int = Query(..., ge=1),
    db: DBSession = Depends(get_db),
):
    """Get the persisted dependency tree + rationale document for a session."""
    try:
        return get_dependencies(db, session_id)
    except DependencyServiceError as exc:
        raise HTTPException(exc.status_code, exc.message)

