"""Single entry for every LLM call: retries, breaker, cache, fallback."""
import hashlib
import json
import random
import re
import sqlite3
import threading
import time
import urllib.error
import urllib.request
from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from typing import Literal

from pydantic import BaseModel, ValidationError

from ..config import Settings, get_settings
from ..domain import ProviderName
from .prompts import Prompt, render

MAX_BODY = 64 * 1024
_sleep = time.sleep  # tests monkeypatch this
_FENCE_RE = re.compile(r"^```[a-zA-Z]*\s*|\s*```$")


@dataclass
class LLMResult:
    output: BaseModel | str
    source: Literal["llm", "cache", "fallback"]
    provider: str
    error: str | None = None


class _Fail(Exception):
    def __init__(self, msg: str, retry: bool):
        super().__init__(msg)
        self.retry = retry


def _parse_model(text: str, model: type[BaseModel]) -> BaseModel:
    text = _FENCE_RE.sub("", text.strip()).strip()
    try:
        obj = json.loads(text, strict=False)
    except json.JSONDecodeError:
        m = re.search(r"\{.*\}", text, re.DOTALL)
        try:
            obj = json.loads(m.group(0), strict=False) if m else None
        except json.JSONDecodeError:
            obj = None
        if obj is None:
            raise _Fail("response was not valid JSON", True) from None
    try:
        return model.model_validate(obj)
    except ValidationError as e:
        raise _Fail(f"response failed validation: {e.error_count()} errors", True) from None


class LLMGateway:
    def __init__(self, settings: Settings):
        self.s = settings
        self._lock = threading.Lock()
        self._slots = threading.BoundedSemaphore(max(1, settings.llm_concurrency))
        self._breaker: dict[str, list[float]] = {}  # provider -> [fails, opened_at]

    # ---- public
    def complete(self, prompt: Prompt, fallback: Callable[[], BaseModel | str],
                 provider: ProviderName | None = None) -> LLMResult:
        provider = provider or self.s.llm_provider
        if provider == "mock":
            return LLMResult(fallback(), "fallback", "mock", None)
        if self._open(provider):
            return LLMResult(fallback(), "fallback", provider, "circuit breaker open")
        messages = render(prompt)
        key = self._cache_key(provider, prompt, messages)
        hit = self._cache_read(key, prompt)
        if hit is not None:
            return LLMResult(hit, "cache", provider, None)
        try:
            out = self._call_with_retries(provider, prompt, messages)
        except _Fail as e:
            self._record(provider, ok=False)
            return LLMResult(fallback(), "fallback", provider, str(e))
        self._record(provider, ok=True)
        self._cache_write(key, out)
        return LLMResult(out, "llm", provider, None)

    def map(self, prompts: list[Prompt], fallback_for: Callable[[Prompt], BaseModel | str],
            provider: ProviderName | None = None) -> list[LLMResult]:
        if not prompts:
            return []
        with ThreadPoolExecutor(max_workers=max(1, self.s.llm_concurrency)) as ex:
            return list(ex.map(
                lambda p: self.complete(p, lambda: fallback_for(p), provider), prompts))

    # ---- transport
    def _call_with_retries(self, provider: str, prompt: Prompt, messages: list[dict]):
        last: _Fail | None = None
        for n in range(self.s.llm_max_retries + 1):
            if n:
                _sleep(0.5 * 2 ** (n - 1) + random.uniform(0, 0.1))
            try:
                text = self._post(provider, prompt, messages)
                if prompt.output_model is not None:
                    return _parse_model(text, prompt.output_model)
                text = text.strip()
                if not text:
                    raise _Fail("empty response", True)
                return text[: prompt.max_output_chars]
            except _Fail as e:
                last = e
                if not e.retry:
                    break
        assert last is not None
        raise last

    def _post(self, provider: str, prompt: Prompt, messages: list[dict]) -> str:
        s = self.s
        schema = prompt.output_model.model_json_schema() if prompt.output_model else None
        headers = {"Content-Type": "application/json"}
        if provider == "openai":
            if not s.llm_base_url:
                raise _Fail("LLM base url not configured", False)
            url = s.llm_base_url.rstrip("/") + "/chat/completions"
            headers["Authorization"] = f"Bearer {s.llm_api_key}"
            body = {"model": s.llm_model, "messages": messages, "temperature": 0, "seed": s.seed}
            if schema:
                body["response_format"] = {"type": "json_schema",
                                           "json_schema": {"name": prompt.task, "schema": schema}}
        else:
            url = s.local_llm_url.rstrip("/") + "/api/chat"
            body = {"model": s.local_llm_model, "messages": messages, "stream": False,
                    "think": False, "options": {"temperature": 0, "seed": s.seed}}
            if schema:
                body["format"] = schema
        req = urllib.request.Request(url, json.dumps(body).encode(), headers, method="POST")
        try:
            with self._slots, urllib.request.urlopen(req, timeout=s.llm_timeout_s) as resp:
                raw = resp.read(MAX_BODY + 1)
        except urllib.error.HTTPError as e:
            raise _Fail(f"HTTP {e.code}", e.code >= 500 or e.code == 429) from None
        except (urllib.error.URLError, TimeoutError, OSError) as e:
            raise _Fail(f"request failed: {type(e).__name__}", True) from None
        if len(raw) > MAX_BODY:
            raise _Fail("response exceeded 64 KB", True)
        try:
            data = json.loads(raw.decode("utf-8"))
            if provider == "openai":
                content = data["choices"][0]["message"]["content"]
            else:
                content = data["message"]["content"]
        except (ValueError, KeyError, IndexError, TypeError):
            raise _Fail("unexpected response envelope", True) from None
        if not isinstance(content, str):
            raise _Fail("unexpected response envelope", True)
        return content

    # ---- breaker
    def _open(self, provider: str) -> bool:
        with self._lock:
            fails, opened = self._breaker.get(provider, [0, 0.0])
            if fails < self.s.breaker_threshold:
                return False
            if time.monotonic() - opened >= self.s.breaker_cooldown_s:
                self._breaker[provider] = [0, 0.0]  # half-open: allow a fresh try
                return False
            return True

    def _record(self, provider: str, ok: bool) -> None:
        with self._lock:
            if ok:
                self._breaker[provider] = [0, 0.0]
                return
            fails = self._breaker.get(provider, [0, 0.0])[0] + 1
            self._breaker[provider] = [fails, time.monotonic()]

    # ---- cache (single SQLite kv file; every failure degrades to a miss)
    def _cache_key(self, provider: str, prompt: Prompt, messages: list[dict]) -> str:
        model = self.s.llm_model if provider == "openai" else self.s.local_llm_model
        schema = prompt.output_model.model_json_schema() if prompt.output_model else None
        key = json.dumps([provider, model, prompt.task, prompt.version, messages, schema],
                         sort_keys=True, ensure_ascii=False)
        return hashlib.sha256(key.encode()).hexdigest()

    def _cache_connect(self) -> sqlite3.Connection:
        path = self.s.cache_dir / "llm.sqlite"
        path.parent.mkdir(parents=True, exist_ok=True)
        con = sqlite3.connect(path, timeout=5)
        con.execute("PRAGMA busy_timeout=5000")
        con.execute("PRAGMA journal_mode=WAL")
        con.execute("CREATE TABLE IF NOT EXISTS kv(key TEXT PRIMARY KEY, value TEXT)")
        return con

    def _cache_read(self, key: str, prompt: Prompt):
        try:
            con = self._cache_connect()
            try:
                row = con.execute("SELECT value FROM kv WHERE key=?", (key,)).fetchone()
            finally:
                con.close()
            if row is None:
                return None
            value = json.loads(row[0])
            if prompt.output_model is not None:
                return prompt.output_model.model_validate(value)
            return value if isinstance(value, str) else None
        except (sqlite3.Error, OSError, ValueError):
            return None

    def _cache_write(self, key: str, out: BaseModel | str) -> None:
        try:
            payload = out.model_dump(mode="json") if isinstance(out, BaseModel) else out
            con = self._cache_connect()
            try:
                with con:
                    con.execute("INSERT OR REPLACE INTO kv(key, value) VALUES (?, ?)",
                                (key, json.dumps(payload, ensure_ascii=False)))
            finally:
                con.close()
        except (sqlite3.Error, OSError):
            pass  # cache is best-effort


_gateway: LLMGateway | None = None
_glock = threading.Lock()


def get_gateway() -> LLMGateway:
    global _gateway
    with _glock:
        if _gateway is None:
            _gateway = LLMGateway(get_settings())
        return _gateway


def reset_gateway() -> None:
    global _gateway
    with _glock:
        _gateway = None
