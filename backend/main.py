import sys
import os
from contextlib import asynccontextmanager

def load_dotenv():
    paths = [
        os.path.join(os.getcwd(), ".env"),
        os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), ".env"),
        os.path.join(os.path.dirname(os.path.abspath(__file__)), ".env"),
    ]
    for path in paths:
        if os.path.isfile(path):
            try:
                with open(path, "r", encoding="utf-8") as f:
                    for line in f:
                        line = line.strip()
                        if not line or line.startswith("#"):
                            continue
                        if "=" in line:
                            k, v = line.split("=", 1)
                            k = k.strip()
                            v = v.strip().strip("'\"")
                            if k and k not in os.environ:
                                os.environ[k] = v
                break
            except Exception:
                pass

load_dotenv()

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
    description="AI-assisted requirements clustering with LLM enrichment, ClusterLLM refinement, human-in-the-loop corrections, active learning, dependency trees, and MBSE export.",
    version="5.0.0",
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
        "version": "5.0.0",
        "status": "running",
        "docs": "/docs",
    }


@app.get("/health")
def health():
    return {"status": "ok"}
