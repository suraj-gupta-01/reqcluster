"""Typed settings. The only module that reads the environment."""
import os
import sys
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Any, Literal, get_args

_REPO = Path(__file__).resolve().parents[2]

Backend = Literal["torch", "mock"]
Device = Literal["auto", "cpu", "cuda"]
Provider = Literal["mock", "openai", "local"]


@dataclass(frozen=True, slots=True)
class Settings:
    home: Path
    database_url: str
    cors_origins: tuple[str, ...]
    host: str
    port: int
    api_key: str
    job_workers: int
    seed: int
    threads: int
    embedding_model: str
    embedding_revision: str
    embedding_backend: Backend
    embedding_device: Device
    max_upload_mb: int
    max_rows: int
    llm_provider: Provider
    llm_base_url: str
    llm_api_key: str
    llm_model: str
    local_llm_url: str
    local_llm_model: str
    local_llm_timeout_s: float
    local_llm_load_timeout_s: float
    local_llm_concurrency: int
    local_llm_keep_alive: str
    llm_timeout_s: float
    llm_max_retries: int
    llm_concurrency: int
    breaker_threshold: int
    breaker_cooldown_s: float
    spacy_model: str
    disable_spacy: bool
    jama_project_id: int
    jama_item_type_id: int
    jama_rel_type_id: int

    @property
    def cache_dir(self) -> Path:
        return self.home / "cache"

    @property
    def runs_dir(self) -> Path:
        return self.home / "runs"


def _read_dotenv(path: Path) -> dict[str, str]:
    out: dict[str, str] = {}
    if not path.is_file():
        return out
    try:
        text = path.read_text(encoding="utf-8")
    except OSError as e:
        print(f"warning: could not read env file {path}: {e}", file=sys.stderr)
        return out
    for line in text.splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        k, v = line.split("=", 1)
        k = k.strip()
        if k:
            out[k] = v.strip().strip("'\"")
    return out


def _bool(v: str) -> bool:
    s = v.strip().lower()
    if s in ("1", "true", "yes", "on"):
        return True
    if s in ("0", "false", "no", "off", ""):
        return False
    raise ValueError(v)


def load_settings(
    env: Mapping[str, str] | None = None, dotenv: Path | None = _REPO / ".env"
) -> Settings:
    merged: dict[str, str] = _read_dotenv(dotenv) if dotenv else {}
    merged.update(os.environ if env is None else env)

    def get(name: str, default: Any, cast: Callable[[str], Any] = str, lo=None, hi=None):
        raw = merged.get(name)
        if raw is None:
            return default
        try:
            val = cast(raw)
        except ValueError:
            raise ValueError(f"{name}: invalid value {raw!r}") from None
        if (lo is not None and val < lo) or (hi is not None and val > hi):
            raise ValueError(f"{name}: {raw!r} out of range [{lo}, {hi}]")
        return val

    def lit(name: str, default: str, typ) -> str:
        val = get(name, default)
        if val not in get_args(typ):
            raise ValueError(f"{name}: {val!r} not one of {get_args(typ)}")
        return val

    home = Path(get("REQCLUSTER_HOME", str(_REPO / "var")))
    origins = get("CORS_ORIGINS", "http://localhost:5173,http://localhost:3000")
    return Settings(
        home=home,
        database_url=get("DATABASE_URL", "sqlite:///" + (home / "reqcluster.db").as_posix()),
        cors_origins=tuple(o.strip() for o in origins.split(",") if o.strip()),
        host=get("REQCLUSTER_HOST", "127.0.0.1"),
        port=get("REQCLUSTER_PORT", 8000, int),
        api_key=get("REQCLUSTER_API_KEY", ""),
        job_workers=get("REQCLUSTER_JOB_WORKERS", 1, int, lo=0),
        seed=get("REQCLUSTER_SEED", 42, int),
        threads=get("REQCLUSTER_THREADS", os.cpu_count() or 1, int, lo=1),
        embedding_model=get("REQCLUSTER_EMBEDDING_MODEL", "auto"),  # auto: gte on GPU, MiniLM on CPU
        embedding_revision=get("REQCLUSTER_EMBEDDING_REVISION", "1"),
        embedding_backend=lit("REQCLUSTER_EMBEDDING_BACKEND", "torch", Backend),
        embedding_device=lit("REQCLUSTER_EMBEDDING_DEVICE", "auto", Device),
        max_upload_mb=get("REQCLUSTER_MAX_UPLOAD_MB", 25, int),
        max_rows=get("REQCLUSTER_MAX_ROWS", 50000, int),
        llm_provider=lit("REQCLUSTER_LLM_PROVIDER", "mock", Provider),
        llm_base_url=get("REQCLUSTER_LLM_BASE_URL", ""),
        llm_api_key=get("REQCLUSTER_LLM_API_KEY", ""),
        llm_model=get("REQCLUSTER_LLM_MODEL", ""),
        local_llm_url=get("REQCLUSTER_LOCAL_LLM_URL", "http://localhost:11434"),
        local_llm_model=get("REQCLUSTER_LOCAL_LLM_MODEL", "qwen3.5:4b"),
        # local models run on CPU / small GPUs: ~10 tok/s and a 100 s cold load were measured
        local_llm_timeout_s=get("REQCLUSTER_LOCAL_LLM_TIMEOUT_SECONDS", 120.0, float),
        local_llm_load_timeout_s=get("REQCLUSTER_LOCAL_LLM_LOAD_TIMEOUT_SECONDS", 600.0, float),
        local_llm_concurrency=get("REQCLUSTER_LOCAL_LLM_CONCURRENCY", 1, int, lo=1),
        local_llm_keep_alive=get("REQCLUSTER_LOCAL_LLM_KEEP_ALIVE", "30m"),
        llm_timeout_s=get("REQCLUSTER_LLM_TIMEOUT_SECONDS", 30.0, float),
        llm_max_retries=get("REQCLUSTER_LLM_MAX_RETRIES", 2, int, lo=0, hi=5),
        llm_concurrency=get("REQCLUSTER_LLM_CONCURRENCY", 4, int, lo=1),
        breaker_threshold=get("REQCLUSTER_BREAKER_THRESHOLD", 5, int),
        breaker_cooldown_s=get("REQCLUSTER_BREAKER_COOLDOWN_SECONDS", 60.0, float),
        spacy_model=get("REQCLUSTER_SPACY_MODEL", "en_core_web_sm"),
        disable_spacy=get("REQCLUSTER_DISABLE_SPACY", False, _bool),
        jama_project_id=get("REQCLUSTER_JAMA_PROJECT_ID", 0, int),
        jama_item_type_id=get("REQCLUSTER_JAMA_ITEM_TYPE_ID", 33, int),
        jama_rel_type_id=get("REQCLUSTER_JAMA_REL_TYPE_ID", 1, int),
    )


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    return load_settings()
