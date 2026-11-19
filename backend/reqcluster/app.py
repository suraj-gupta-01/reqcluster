"""FastAPI application factory."""
import hmac
import logging
from contextlib import asynccontextmanager

import uvicorn
from fastapi import Depends, FastAPI, Header, Request
from fastapi.exceptions import RequestValidationError
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from starlette.exceptions import HTTPException as StarletteHTTPException

from reqcluster import db, embeddings, jobs, runtime
from reqcluster.api import dependencies, enrichment, export, feedback, refinement
from reqcluster.api import jobs as jobs_api
from reqcluster.api import sessions as sessions_api
from reqcluster.config import Settings, get_settings
from reqcluster.errors import AppError, Unauthorized

log = logging.getLogger(__name__)


def _envelope(status: int, code: str, message: str) -> JSONResponse:
    return JSONResponse({"error": code, "message": message}, status_code=status)


def create_app(settings: Settings | None = None) -> FastAPI:
    s = settings or get_settings()

    @asynccontextmanager
    async def lifespan(_app):
        try:
            runtime.configure_process(s)
            db.init_db()
            jobs.get_runner().sweep_on_startup()
            runtime.warm_up(s)
            if s.embedding_backend != "mock":  # tests: compile lazily, no competing thread
                runtime.warm_up_reducer(s.seed)
        except Exception:
            log.exception("startup failed: could not initialise database, runner or embedder")
            raise
        yield
        jobs.reset_runner(wait=False)

    def require_key(x_api_key: str | None = Header(default=None)) -> None:
        if s.api_key and not hmac.compare_digest(x_api_key or "", s.api_key):
            raise Unauthorized("missing or invalid API key")

    app = FastAPI(title="ReqCluster", lifespan=lifespan)
    app.add_middleware(
        CORSMiddleware, allow_origins=list(s.cors_origins), allow_methods=["*"],
        allow_headers=["*"], expose_headers=["X-Total-Count"],
    )

    @app.exception_handler(AppError)
    def _app_error(_r: Request, exc: AppError):
        return _envelope(exc.status, exc.code, exc.message)

    @app.exception_handler(RequestValidationError)
    def _validation(_r: Request, exc: RequestValidationError):
        msg = "; ".join(
            f"{'.'.join(str(p) for p in e['loc'])}: {e['msg']}" for e in exc.errors())
        return _envelope(422, "validation_error", msg)

    @app.exception_handler(StarletteHTTPException)
    def _http(_r: Request, exc: StarletteHTTPException):
        return _envelope(exc.status_code, "http_error", str(exc.detail))

    @app.exception_handler(Exception)
    def _unhandled(_r: Request, exc: Exception):
        log.error("unhandled error", exc_info=exc)
        return _envelope(500, "internal", "internal server error")

    @app.get("/health")
    def health() -> dict:
        e = embeddings.get_embedder()
        return {"status": "ok", "embedder": e.name, "is_mock": e.is_mock}

    deps = [Depends(require_key)]
    for m in (sessions_api, jobs_api, enrichment, dependencies, refinement, feedback, export):
        app.include_router(m.router, prefix="/api", dependencies=deps)
    return app


def main() -> None:
    s = get_settings()
    uvicorn.run(create_app(), host=s.host, port=s.port)
