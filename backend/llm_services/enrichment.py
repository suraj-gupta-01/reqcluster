from __future__ import annotations

import asyncio
import hashlib
import json
import math
import os
import re
import tempfile
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Sequence

from .prompts import PROMPT_VERSION, normalize_domain_terms, normalize_plain_text
from .providers import (
    ParsedExpansion,
    ProviderResponseError,
    RequirementExpansionProvider,
    get_provider,
)
from .quality import build_quality_report, evaluate_expansion_quality
from .vocabulary import extract_domain_vocabulary


ENRICHMENT_CONFIG_VERSION = "reqcluster-llm-enrichment-v1"
MAX_INPUT_TEXT_CHARS = 12_000
SAFE_DIGEST_RE = re.compile(r"^[0-9a-f]{64}$")
CACHE_DIR = Path(__file__).resolve().parents[1] / "cache" / "llm_enrichment"


@dataclass(frozen=True)
class RequirementExpansionResult:
    requirement_id: str
    original_text_hash: str
    expanded_text: str
    domain_terms: list[str]
    functional_intent: str
    mentioned_components: list[str]
    assumptions: list[str]
    confidence: float
    warnings: list[str]
    provider: str
    model: str
    prompt_version: str
    created_at: str

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass(frozen=True)
class EnrichmentBatchResult:
    total: int
    succeeded: int
    failed: int
    results: list[RequirementExpansionResult]
    errors: list[dict]
    domain_vocabulary: list[str]
    quality_report: dict
    duration_ms: float

    def to_dict(self) -> dict:
        payload = asdict(self)
        payload["results"] = [result.to_dict() for result in self.results]
        return payload


def _utc_now_iso() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


def _bounded_int(value: int, default: int, low: int, high: int) -> int:
    try:
        parsed = int(value)
    except (TypeError, ValueError):
        parsed = default
    return min(max(parsed, low), high)


def _bounded_float(value: float, default: float, low: float, high: float) -> float:
    try:
        parsed = float(value)
    except (TypeError, ValueError):
        parsed = default
    return min(max(parsed, low), high)


def _original_hash(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _cache_key(
    original_text: str,
    provider_name: str,
    model_name: str,
    domain_vocabulary: Sequence[str],
) -> str:
    payload = {
        "config_version": ENRICHMENT_CONFIG_VERSION,
        "domain_vocabulary": list(domain_vocabulary),
        "model": model_name or "",
        "normalized_original_requirement_text": original_text,
        "prompt_version": PROMPT_VERSION,
        "provider": provider_name,
    }
    serialized = json.dumps(payload, ensure_ascii=False, separators=(",", ":"), sort_keys=True)
    return hashlib.sha256(serialized.encode("utf-8")).hexdigest()


def _cache_path(cache_key: str) -> Path:
    if not SAFE_DIGEST_RE.fullmatch(cache_key):
        raise ValueError("Invalid enrichment cache key.")
    return CACHE_DIR / f"llm_enrichment_{cache_key}.json"


def _dedupe(values: Sequence[str]) -> list[str]:
    result: list[str] = []
    seen: set[str] = set()
    for value in values:
        cleaned = normalize_plain_text(value, max_chars=1_000)
        if not cleaned:
            continue
        key = cleaned.casefold()
        if key in seen:
            continue
        seen.add(key)
        result.append(cleaned)
    return result


def _prepare_requirement_ids(
    requirement_ids: Sequence[str] | None,
    total: int,
) -> list[str]:
    if requirement_ids is None:
        return [f"REQ-{idx + 1:03d}" for idx in range(total)]
    ids = list(requirement_ids)
    if len(ids) != total:
        raise ValueError("requirement_ids must match the number of requirement texts.")
    prepared = []
    for idx, req_id in enumerate(ids):
        cleaned = normalize_plain_text(req_id, max_chars=128)
        prepared.append(cleaned or f"REQ-{idx + 1:03d}")
    return prepared


def _error(index: int, requirement_id: str, message: str, error_type: str) -> dict:
    return {
        "index": int(index),
        "requirement_id": requirement_id,
        "error_type": error_type,
        "message": normalize_plain_text(message, max_chars=500),
    }


def _result_from_payload(
    payload: ParsedExpansion,
    requirement_id: str,
    original_text: str,
    provider: RequirementExpansionProvider,
) -> RequirementExpansionResult:
    quality = evaluate_expansion_quality(
        original_text,
        payload.expanded_text,
        domain_terms=payload.domain_terms,
        provider_confidence=payload.confidence,
    )
    warnings = _dedupe([*payload.warnings, *quality.get("warnings", [])])
    return RequirementExpansionResult(
        requirement_id=requirement_id,
        original_text_hash=_original_hash(original_text),
        expanded_text=payload.expanded_text,
        domain_terms=list(payload.domain_terms),
        functional_intent=payload.functional_intent,
        mentioned_components=list(payload.mentioned_components),
        assumptions=list(payload.assumptions),
        confidence=round(float(payload.confidence), 6),
        warnings=warnings,
        provider=provider.name,
        model=provider.model,
        prompt_version=PROMPT_VERSION,
        created_at=_utc_now_iso(),
    )


def _cache_record(result: RequirementExpansionResult) -> dict:
    payload = result.to_dict()
    payload.pop("requirement_id", None)
    return {
        "schema_version": ENRICHMENT_CONFIG_VERSION,
        "result": payload,
    }


def _load_cached_result(cache_file: Path, requirement_id: str) -> RequirementExpansionResult | None:
    try:
        with cache_file.open("r", encoding="utf-8") as handle:
            record = json.load(handle)
    except Exception:
        return None

    if not isinstance(record, dict) or record.get("schema_version") != ENRICHMENT_CONFIG_VERSION:
        return None
    data = record.get("result")
    if not isinstance(data, dict):
        return None

    try:
        confidence = float(data.get("confidence", 0.0))
        if not math.isfinite(confidence):
            return None
        expanded_text = normalize_plain_text(data["expanded_text"], max_chars=12_000)
        if not expanded_text:
            return None
        prompt_version = normalize_plain_text(data.get("prompt_version", ""), max_chars=120)
        if prompt_version != PROMPT_VERSION:
            return None
        return RequirementExpansionResult(
            requirement_id=requirement_id,
            original_text_hash=str(data["original_text_hash"]),
            expanded_text=expanded_text,
            domain_terms=normalize_domain_terms(data.get("domain_terms", [])),
            functional_intent=normalize_plain_text(data.get("functional_intent", ""), max_chars=1_000),
            mentioned_components=[
                normalize_plain_text(item, max_chars=160)
                for item in data.get("mentioned_components", [])
                if normalize_plain_text(item, max_chars=160)
            ][:50],
            assumptions=[
                normalize_plain_text(item, max_chars=1_000)
                for item in data.get("assumptions", [])
                if normalize_plain_text(item, max_chars=1_000)
            ][:20],
            confidence=round(min(max(confidence, 0.0), 1.0), 6),
            warnings=[
                normalize_plain_text(item, max_chars=1_000)
                for item in data.get("warnings", [])
                if normalize_plain_text(item, max_chars=1_000)
            ][:30],
            provider=normalize_plain_text(data.get("provider", ""), max_chars=80),
            model=normalize_plain_text(data.get("model", ""), max_chars=160),
            prompt_version=prompt_version,
            created_at=normalize_plain_text(data.get("created_at", ""), max_chars=40),
        )
    except Exception:
        return None


def _atomic_write_json(cache_file: Path, payload: dict) -> None:
    cache_file.parent.mkdir(parents=True, exist_ok=True)
    tmp_name: str | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w",
            encoding="utf-8",
            dir=cache_file.parent,
            prefix=f".{cache_file.stem}.",
            suffix=".tmp",
            delete=False,
        ) as tmp:
            tmp_name = tmp.name
            json.dump(payload, tmp, ensure_ascii=False, sort_keys=True)
            tmp.flush()
            os.fsync(tmp.fileno())
        os.replace(tmp_name, cache_file)
    except Exception:
        if tmp_name and os.path.exists(tmp_name):
            try:
                os.remove(tmp_name)
            except OSError:
                pass
        raise


async def enrich_requirements(
    requirement_texts: Sequence[str],
    requirement_ids: Sequence[str] | None = None,
    provider_name: str = "mock",
    domain_vocabulary: Sequence[str] | None = None,
    batch_size: int = 8,
    max_concurrency: int = 4,
    timeout_seconds: float = 30,
    fail_fast: bool = False,
    use_cache: bool = True,
) -> EnrichmentBatchResult:
    """
    Expand cleaned requirement text into meaning-preserving enriched text.

    The service prepares text and quality metadata only. It does not run UMAP,
    HDBSCAN, labeling, graph construction, API writes, or database mutation.
    """
    start = time.perf_counter()
    if requirement_texts is None or isinstance(requirement_texts, (str, bytes)):
        raise ValueError("requirement_texts must be a sequence of strings.")

    raw_texts = list(requirement_texts)
    total = len(raw_texts)
    req_ids = _prepare_requirement_ids(requirement_ids, total)
    bounded_batch_size = _bounded_int(batch_size, 8, 1, 64)
    bounded_concurrency = _bounded_int(max_concurrency, 4, 1, 16)
    bounded_timeout = _bounded_float(timeout_seconds, 30.0, 1.0, 300.0)

    normalized_texts: list[str] = []
    validation_errors: dict[int, dict] = {}
    for idx, value in enumerate(raw_texts):
        if not isinstance(value, str):
            validation_errors[idx] = _error(idx, req_ids[idx], "Requirement text must be a string.", "validation_error")
            normalized_texts.append("")
            continue
        cleaned = normalize_plain_text(value, max_chars=MAX_INPUT_TEXT_CHARS)
        if not cleaned:
            validation_errors[idx] = _error(idx, req_ids[idx], "Requirement text is empty.", "validation_error")
        normalized_texts.append(cleaned)

    if fail_fast and validation_errors:
        first = validation_errors[min(validation_errors)]
        raise ValueError(first["message"])

    if domain_vocabulary is None:
        extracted = extract_domain_vocabulary(
            [text for idx, text in enumerate(normalized_texts) if idx not in validation_errors],
            top_n=50,
        )
        vocabulary = [str(term) for term in extracted]
    else:
        vocabulary = normalize_domain_terms(domain_vocabulary)

    provider = get_provider(provider_name)
    semaphore = asyncio.Semaphore(bounded_concurrency)
    results_by_index: list[RequirementExpansionResult | None] = [None] * total
    errors_by_index: dict[int, dict] = dict(validation_errors)

    async def process_one(index: int) -> tuple[int, RequirementExpansionResult | None, dict | None]:
        if index in validation_errors:
            return index, None, validation_errors[index]

        text = normalized_texts[index]
        req_id = req_ids[index]
        key = _cache_key(text, provider.name, provider.model, vocabulary)
        cache_file = _cache_path(key)

        async with semaphore:
            if use_cache and cache_file.exists():
                cached = _load_cached_result(cache_file, req_id)
                if cached is not None:
                    return index, cached, None

            try:
                payload = await asyncio.wait_for(
                    provider.expand_requirement(
                        text,
                        requirement_id=req_id,
                        domain_vocabulary=vocabulary,
                        timeout_seconds=bounded_timeout,
                    ),
                    timeout=bounded_timeout + 1.0,
                )
                if not isinstance(payload, ParsedExpansion):
                    raise ProviderResponseError("Provider returned an invalid expansion object.")
                result = _result_from_payload(payload, req_id, text, provider)
                if use_cache:
                    _atomic_write_json(cache_file, _cache_record(result))
                return index, result, None
            except Exception as exc:
                if fail_fast:
                    raise
                return index, None, _error(index, req_id, str(exc), exc.__class__.__name__)

    for start_idx in range(0, total, bounded_batch_size):
        batch_indices = range(start_idx, min(start_idx + bounded_batch_size, total))
        tasks = [asyncio.create_task(process_one(idx)) for idx in batch_indices]
        gathered = await asyncio.gather(*tasks, return_exceptions=not fail_fast)
        for item in gathered:
            if isinstance(item, Exception):
                if fail_fast:
                    raise item
                continue
            idx, result, error = item
            if result is not None:
                results_by_index[idx] = result
            if error is not None:
                errors_by_index[idx] = error

    ordered_results = [result for result in results_by_index if result is not None]
    ordered_errors = [errors_by_index[idx] for idx in sorted(errors_by_index)]
    quality_report = build_quality_report(normalized_texts, results_by_index, vocabulary)
    duration_ms = round((time.perf_counter() - start) * 1000.0, 3)

    return EnrichmentBatchResult(
        total=total,
        succeeded=len(ordered_results),
        failed=len(ordered_errors),
        results=ordered_results,
        errors=ordered_errors,
        domain_vocabulary=vocabulary,
        quality_report=quality_report,
        duration_ms=duration_ms,
    )


def extract_enriched_texts(batch_result: EnrichmentBatchResult | dict) -> list[str | None]:
    if isinstance(batch_result, dict):
        total = int(batch_result.get("total", 0))
        errors = batch_result.get("errors", [])
        results = batch_result.get("results", [])
    else:
        total = int(batch_result.total)
        errors = batch_result.errors
        results = batch_result.results

    failed_indices: set[int] = set()
    for error in errors:
        if not isinstance(error, dict):
            continue
        try:
            failed_indices.add(int(error.get("index")))
        except (TypeError, ValueError):
            continue
    output: list[str | None] = [None] * total
    result_iter = iter(results)
    for idx in range(total):
        if idx in failed_indices:
            continue
        try:
            result = next(result_iter)
        except StopIteration:
            break
        if isinstance(result, dict):
            output[idx] = result.get("expanded_text")
        else:
            output[idx] = result.expanded_text
    return output


async def enrich_then_prepare_pipeline_inputs(
    requirement_texts: Sequence[str],
    requirement_ids: Sequence[str] | None = None,
    provider_name: str = "mock",
    domain_vocabulary: Sequence[str] | None = None,
    batch_size: int = 8,
    max_concurrency: int = 4,
    timeout_seconds: float = 30,
    fail_fast: bool = False,
    use_cache: bool = True,
) -> dict:
    raw_texts = list(requirement_texts)
    req_ids = _prepare_requirement_ids(requirement_ids, len(raw_texts))
    original_texts = [normalize_plain_text(text, max_chars=MAX_INPUT_TEXT_CHARS) for text in raw_texts]
    batch_result = await enrich_requirements(
        requirement_texts=original_texts,
        requirement_ids=req_ids,
        provider_name=provider_name,
        domain_vocabulary=domain_vocabulary,
        batch_size=batch_size,
        max_concurrency=max_concurrency,
        timeout_seconds=timeout_seconds,
        fail_fast=fail_fast,
        use_cache=use_cache,
    )
    enriched_texts = extract_enriched_texts(batch_result)
    warnings = _dedupe(
        [
            *batch_result.quality_report.get("warnings", []),
            *[error.get("message", "") for error in batch_result.errors],
        ]
    )
    return {
        "original_texts": original_texts,
        "requirement_ids": req_ids,
        "enriched_texts": enriched_texts,
        "domain_vocabulary": batch_result.domain_vocabulary,
        "quality_report": batch_result.quality_report,
        "recommended_embedding_mode": "hybrid" if batch_result.succeeded else "base",
        "warnings": warnings,
    }
