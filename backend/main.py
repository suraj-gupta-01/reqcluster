import sys
import os
from contextlib import asynccontextmanager

sys.path.insert(0, os.path.dirname(__file__))

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
import logging

from models.database import init_db, reset_stale_sessions
from api.routes import router

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info("Initializing database...")
    init_db()
    reset_stale_sessions()
    logger.info("ReqCluster API ready.")
    yield
    logger.info("Shutting down ReqCluster API...")


app = FastAPI(
    title="ReqCluster API",
    description="Requirement clustering with Phase 2 LLM enrichment and hybrid embedding support.",
    version="2.0.0",
    lifespan=lifespan,
)

# Restrict CORS to known frontends. Override via CORS_ORIGINS (comma-separated).
# Note: the "*" wildcard cannot be combined with credentials per the CORS spec.
_origins = os.getenv("CORS_ORIGINS", "http://localhost:5173,http://localhost:3000")
allow_origins = [o.strip() for o in _origins.split(",") if o.strip()]

app.add_middleware(
    CORSMiddleware,
    allow_origins=allow_origins,
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(router, prefix="/api")


@app.get("/")
def root():
    return {
        "name": "ReqCluster",
        "version": "2.0.0",
        "status": "running",
        "docs": "/docs",
    }


@app.get("/health")
def health():
    return {"status": "ok"}
