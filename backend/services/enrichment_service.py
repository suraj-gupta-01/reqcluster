from __future__ import annotations

import hashlib
import json
import time
from datetime import datetime
from typing import Any, Callable, Sequence

from sqlalchemy.orm import Session as DBSession

from llm_services.enrichment import EnrichmentBatchResult, RequirementExpansionResult, enrich_requirements
from llm_services.prompts import PROMPT_VERSION, normalize_plain_text
from llm_services.providers import ProviderConfigurationError, get_provider
from llm_services.quality import summarize_quality_report
from llm_services.vocabulary import extract_domain_vocabulary
from models.database import EnrichedRequirement, Requirement, Session, utcnow


ProgressCallback = Callable[[str, int, str], None]
SUCCESS_STATUS = "succeeded"
FAILED_STATUS = "failed"
PENDING_STATUS = "pending"
MAX_WARNING_COUNT = 50


class EnrichmentServiceError(RuntimeError):
    def __init__(self, status_code: int, message: str) -> None:
        super().__init__(message)
        self.status_code = int(status_code)
        self.message = message


class EnrichmentAlignmentError(EnrichmentServiceError):
    pass


def _safe_message(value: object, max_chars: int = 500) -> str:
    return normalize_plain_text(value, max_chars=max_chars)


def _dedupe(values: Sequence[object], max_items: int = MAX_WARNING_COUNT) -> list[str]:
    result: list[str] = []
    seen: set[str] = set()
    for value in values:
        cleaned = _safe_message(value, max_chars=500)
        if not cleaned:
            continue
        key = cleaned.casefold()
        if key in seen:
            continue
        seen.add(key)
        result.append(cleaned)
        if len(result) >= max_items:
            break
    return result


def _requirement_text_hash(text: str) -> str:
    normalized = normalize_plain_text(text)
    return hashlib.sha256(normalized.encode("utf-8")).hexdigest()


def _json_list(value: Any) -> tuple[list, bool]:
    if value is None:
        return [], True
    parsed = value
    if isinstance(value, str):
        try:
            parsed = json.loads(value)
        except json.JSONDecodeError:
            return [], False
    if not isinstance(parsed, list):
        return [], False
    return list(parsed), True


def _json_dict(value: Any) -> tuple[dict, bool]:
    if value is None:
        return {}, True
    parsed = value
    if isinstance(value, str):
        try:
            parsed = json.loads(value)
        except json.JSONDecodeError:
            return {}, False
    if not isinstance(parsed, dict):
        return {}, False
    return dict(parsed), True


def _row_json_is_valid(row: EnrichedRequirement) -> bool:
    for value in (
        row.domain_terms_json,
        row.mentioned_components_json,
        row.assumptions_json,
        row.warnings_json,
    ):
        _, ok = _json_list(value)
        if not ok:
            return False
    _, ok = _json_dict(row.quality_report_json)
    return ok


def _row_is_valid_for_requirement(row: EnrichedRequirement, req: Requirement) -> bool:
    if row.status != SUCCESS_STATUS:
        return False
    if row.requirement_db_id != req.id:
        return False
    if row.requirement_text_hash != _requirement_text_hash(req.text):
        return False
    if row.prompt_version != PROMPT_VERSION:
        return False
    if not normalize_plain_text(row.expanded_text or ""):
        return False
    return _row_json_is_valid(row)


def _row_is_valid_for_provider(
    row: EnrichedRequirement,
    req: Requirement,
    provider: str,
    model: str,
) -> bool:
    return (
        row.provider == provider
        and row.model == model
        and _row_is_valid_for_requirement(row, req)
    )


def get_requirements_for_session_ordered(
    db: DBSession,
    session_id: int,
) -> tuple[Session, list[Requirement]]:
    session = db.query(Session).filter(Session.id == session_id).first()
    if not session:
        raise EnrichmentServiceError(404, f"Session {session_id} not found.")
    reqs = (
        db.query(Requirement)
        .filter(Requirement.session_id == session_id)
        .order_by(Requirement.id.asc())
        .all()
    )
    if not reqs:
        raise EnrichmentServiceError(400, "No requirements found for session.")
    return session, reqs


def _get_existing_row(
    db: DBSession,
    req: Requirement,
    provider: str,
    model: str,
) -> EnrichedRequirement | None:
    return (
        db.query(EnrichedRequirement)
        .filter(
            EnrichedRequirement.session_id == req.session_id,
            EnrichedRequirement.requirement_db_id == req.id,
            EnrichedRequirement.requirement_text_hash == _requirement_text_hash(req.text),
            EnrichedRequirement.provider == provider,
            EnrichedRequirement.model == model,
            EnrichedRequirement.prompt_version == PROMPT_VERSION,
        )
        .first()
    )


def _result_quality_from_batch(
    batch_result: EnrichmentBatchResult,
    subset_index: int,
) -> dict:
    rows = batch_result.quality_report.get("per_requirement", [])
    if isinstance(rows, list):
        for row in rows:
            if isinstance(row, dict) and row.get("index") == subset_index:
                return row
    return {}


def _upsert_success_row(
    db: DBSession,
    req: Requirement,
    result: RequirementExpansionResult,
    quality_report: dict,
    recommended_mode: str,
) -> EnrichedRequirement:
    row = _get_existing_row(db, req, result.provider, result.model)
    now = utcnow()
    if row is None:
        row = EnrichedRequirement(
            session_id=req.session_id,
            requirement_db_id=req.id,
            requirement_id=req.req_id,
            requirement_text_hash=_requirement_text_hash(req.text),
            provider=result.provider,
            model=result.model,
            prompt_version=result.prompt_version,
            created_at=now,
        )
        db.add(row)

    row.requirement_id = req.req_id
    row.embedding_mode_recommended = recommended_mode
    row.expanded_text = result.expanded_text
    row.domain_terms_json = list(result.domain_terms)
    row.functional_intent = result.functional_intent
    row.mentioned_components_json = list(result.mentioned_components)
    row.assumptions_json = list(result.assumptions)
    row.confidence = float(result.confidence)
    row.warnings_json = list(result.warnings)
    row.quality_report_json = quality_report or {}
    row.status = SUCCESS_STATUS
    row.error_message = None
    row.updated_at = now
    return row


def _upsert_failure_row(
    db: DBSession,
    req: Requirement,
    provider: str,
    model: str,
    message: str,
    quality_report: dict | None,
    recommended_mode: str,
) -> EnrichedRequirement:
    row = _get_existing_row(db, req, provider, model)
    now = utcnow()
    if row is None:
        row = EnrichedRequirement(
            session_id=req.session_id,
            requirement_db_id=req.id,
            requirement_id=req.req_id,
            requirement_text_hash=_requirement_text_hash(req.text),
            provider=provider,
            model=model,
            prompt_version=PROMPT_VERSION,
            created_at=now,
        )
        db.add(row)

    row.requirement_id = req.req_id
    row.embedding_mode_recommended = recommended_mode
    row.expanded_text = None
    row.domain_terms_json = []
    row.functional_intent = None
    row.mentioned_components_json = []
    row.assumptions_json = []
    row.confidence = None
    row.warnings_json = [_safe_message(message)]
    row.quality_report_json = quality_report or {
        "status": FAILED_STATUS,
        "warnings": [_safe_message(message)],
    }
    row.status = FAILED_STATUS
    row.error_message = _safe_message(message)
    row.updated_at = now
    return row


def _latest_row_by_requirement(
    rows: Sequence[EnrichedRequirement],
) -> dict[int, EnrichedRequirement]:
    latest: dict[int, EnrichedRequirement] = {}
    for row in sorted(
        rows,
        key=lambda item: (
            item.updated_at or item.created_at or datetime.min,
            item.id or 0,
        ),
        reverse=True,
    ):
        latest.setdefault(int(row.requirement_db_id), row)
    return latest


def _row_to_quality_item(
    req_index: int,
    row: EnrichedRequirement | None,
    req: Requirement,
) -> dict:
    if row is None:
        return {
            "index": int(req_index),
            "requirement_id": req.req_id,
            "status": FAILED_STATUS,
            "warnings": ["Requirement enrichment is missing."],
        }

    quality, ok = _json_dict(row.quality_report_json)
    warnings, warnings_ok = _json_list(row.warnings_json)
    if not warnings_ok:
        warnings = ["Persisted enrichment warnings were malformed."]
    if not ok:
        quality = {
            "index": int(req_index),
            "requirement_id": req.req_id,
            "status": row.status,
            "warnings": ["Persisted enrichment quality report was malformed."],
        }
    quality.setdefault("index", int(req_index))
    quality.setdefault("requirement_id", req.req_id)
    quality.setdefault("status", row.status)
    quality["warnings"] = _dedupe([*quality.get("warnings", []), *warnings])
    return quality


def _quality_report_for_rows(
    reqs: Sequence[Requirement],
    rows_by_index: dict[int, EnrichedRequirement],
) -> dict:
    return summarize_quality_report(
        [
            _row_to_quality_item(idx, rows_by_index.get(idx), req)
            for idx, req in enumerate(reqs)
        ]
    )


def _row_to_result_dict(row: EnrichedRequirement) -> dict:
    domain_terms, _ = _json_list(row.domain_terms_json)
    mentioned, _ = _json_list(row.mentioned_components_json)
    assumptions, _ = _json_list(row.assumptions_json)
    warnings, warnings_ok = _json_list(row.warnings_json)
    quality, quality_ok = _json_dict(row.quality_report_json)
    if not warnings_ok:
        warnings = ["Persisted enrichment warnings were malformed."]
    if not quality_ok:
        quality = {
            "status": row.status,
            "warnings": ["Persisted enrichment quality report was malformed."],
        }
    return {
        "requirement_id": row.requirement_id,
        "expanded_text": row.expanded_text,
        "domain_terms": [str(item) for item in domain_terms if isinstance(item, str)],
        "functional_intent": row.functional_intent,
        "mentioned_components": [str(item) for item in mentioned if isinstance(item, str)],
        "assumptions": [str(item) for item in assumptions if isinstance(item, str)],
        "confidence": row.confidence,
        "warnings": [str(item) for item in warnings if isinstance(item, str)],
        "quality_report": quality,
        "status": row.status,
    }


def _provider_for_request(provider_name: str):
    try:
        return get_provider(provider_name)
    except ProviderConfigurationError as exc:
        raise EnrichmentServiceError(
            400,
            f"Enrichment provider not configured: {_safe_message(exc)}",
        ) from exc
    except Exception as exc:
        raise EnrichmentServiceError(400, "Enrichment provider not configured.") from exc


def _progress(callback: ProgressCallback | None, step: str, pct: int, message: str) -> None:
    if callback:
        callback(step, pct, message)


async def run_and_persist_enrichment(
    db: DBSession,
    request: Any,
    progress_callback: ProgressCallback | None = None,
) -> dict:
    start = time.perf_counter()
    _progress(progress_callback, "loading_requirements", 5, "Loading requirements...")
    _session, reqs = get_requirements_for_session_ordered(db, int(request.session_id))
    texts = [normalize_plain_text(req.text) for req in reqs]
    req_ids = [str(req.req_id or f"REQ-{idx + 1:03d}") for idx, req in enumerate(reqs)]

    provider = _provider_for_request(request.provider_name)
    provider_name = provider.name
    model_name = provider.model

    _progress(progress_callback, "vocabulary_extraction", 15, "Extracting domain vocabulary...")
    domain_vocabulary = [str(term) for term in extract_domain_vocabulary(texts, top_n=50)]

    rows_by_index: dict[int, EnrichedRequirement] = {}
    missing_indices: list[int] = []
    if not request.force_refresh:
        for idx, req in enumerate(reqs):
            row = _get_existing_row(db, req, provider_name, model_name)
            if row is not None and _row_is_valid_for_provider(row, req, provider_name, model_name):
                rows_by_index[idx] = row
            else:
                missing_indices.append(idx)
    else:
        missing_indices = list(range(len(reqs)))

    batch_result: EnrichmentBatchResult | None = None
    if missing_indices:
        _progress(progress_callback, "llm_enrichment", 35, "Running requirement enrichment...")
        try:
            batch_result = await enrich_requirements(
                requirement_texts=[texts[idx] for idx in missing_indices],
                requirement_ids=[req_ids[idx] for idx in missing_indices],
                provider_name=request.provider_name,
                domain_vocabulary=domain_vocabulary,
                batch_size=request.batch_size,
                max_concurrency=request.max_concurrency,
                timeout_seconds=request.timeout_seconds,
                fail_fast=request.fail_fast,
                use_cache=request.use_cache,
            )
        except ProviderConfigurationError as exc:
            raise EnrichmentServiceError(
                400,
                f"Enrichment provider not configured: {_safe_message(exc)}",
            ) from exc
        except Exception as exc:
            raise EnrichmentServiceError(502, "Enrichment failed.") from exc

        _progress(progress_callback, "quality_evaluation", 75, "Evaluating enrichment quality...")
        failed_subset_indices = {
            int(error.get("index"))
            for error in batch_result.errors
            if isinstance(error, dict) and isinstance(error.get("index"), int)
        }
        result_iter = iter(batch_result.results)

        _progress(progress_callback, "persistence", 85, "Persisting enrichment results...")
        try:
            for subset_idx, req_idx in enumerate(missing_indices):
                req = reqs[req_idx]
                if subset_idx in failed_subset_indices:
                    error = next(
                        (
                            item
                            for item in batch_result.errors
                            if isinstance(item, dict) and item.get("index") == subset_idx
                        ),
                        {},
                    )
                    row = _upsert_failure_row(
                        db,
                        req,
                        provider_name,
                        model_name,
                        str(error.get("message") or "Requirement enrichment failed."),
                        _result_quality_from_batch(batch_result, subset_idx),
                        request.embedding_mode,
                    )
                    rows_by_index[req_idx] = row
                    continue

                result = next(result_iter)
                row = _upsert_success_row(
                    db,
                    req,
                    result,
                    _result_quality_from_batch(batch_result, subset_idx),
                    request.embedding_mode,
                )
                rows_by_index[req_idx] = row
            db.commit()
        except Exception:
            db.rollback()
            raise EnrichmentServiceError(500, "Failed to persist enrichment results.")

        for idx, row in list(rows_by_index.items()):
            try:
                db.refresh(row)
            except Exception:
                pass

    quality_report = _quality_report_for_rows(reqs, rows_by_index)
    succeeded = sum(1 for row in rows_by_index.values() if row.status == SUCCESS_STATUS)
    failed = len(reqs) - succeeded
    status = "complete" if succeeded == len(reqs) else "partial" if succeeded else "failed"
    warnings = _dedupe(
        [
            *quality_report.get("warnings", []),
            *(
                error.get("message", "")
                for error in (batch_result.errors if batch_result is not None else [])
                if isinstance(error, dict)
            ),
        ]
    )

    _progress(progress_callback, "complete", 100, "Enrichment complete.")
    return {
        "session_id": int(request.session_id),
        "status": status,
        "total": len(reqs),
        "succeeded": int(succeeded),
        "failed": int(failed),
        "provider": provider_name,
        "model": model_name,
        "prompt_version": PROMPT_VERSION,
        "domain_vocabulary": domain_vocabulary,
        "quality_report": quality_report,
        "warnings": warnings,
        "duration_ms": round((time.perf_counter() - start) * 1000.0, 3),
    }


def get_enrichment_status(db: DBSession, session_id: int) -> dict:
    session = db.query(Session).filter(Session.id == session_id).first()
    if not session:
        raise EnrichmentServiceError(404, f"Session {session_id} not found.")
    reqs = (
        db.query(Requirement)
        .filter(Requirement.session_id == session_id)
        .order_by(Requirement.id.asc())
        .all()
    )
    rows = (
        db.query(EnrichedRequirement)
        .filter(EnrichedRequirement.session_id == session_id)
        .order_by(EnrichedRequirement.updated_at.desc(), EnrichedRequirement.id.desc())
        .all()
    )
    latest = _latest_row_by_requirement(rows)
    succeeded = 0
    failed = 0
    warnings: list[str] = []
    for req in reqs:
        row = latest.get(req.id)
        if row is None:
            continue
        row_warnings, _ = _json_list(row.warnings_json)
        warnings.extend(str(item) for item in row_warnings if isinstance(item, str))
        if _row_is_valid_for_requirement(row, req):
            succeeded += 1
        elif row.status == FAILED_STATUS:
            failed += 1

    total = len(reqs)
    pending = max(total - succeeded - failed, 0)
    status = "not_started"
    if succeeded == total and total > 0:
        status = "complete"
    elif succeeded or failed:
        status = "partial" if succeeded else "failed"

    latest_row = rows[0] if rows else None
    return {
        "session_id": int(session_id),
        "status": status,
        "total": int(total),
        "succeeded": int(succeeded),
        "failed": int(failed),
        "pending": int(pending),
        "latest_run_created_at": latest_row.created_at if latest_row else None,
        "provider": latest_row.provider if latest_row else None,
        "model": latest_row.model if latest_row else None,
        "warnings": _dedupe(warnings),
    }


def get_enrichment_results(db: DBSession, session_id: int) -> list[dict]:
    _session, reqs = get_requirements_for_session_ordered(db, session_id)
    rows = (
        db.query(EnrichedRequirement)
        .filter(EnrichedRequirement.session_id == session_id)
        .order_by(EnrichedRequirement.updated_at.desc(), EnrichedRequirement.id.desc())
        .all()
    )
    latest = _latest_row_by_requirement(rows)
    output: list[dict] = []
    for req in reqs:
        row = latest.get(req.id)
        if row is not None:
            output.append(_row_to_result_dict(row))
    return output


def validate_enrichment_alignment(
    reqs: Sequence[Requirement],
    rows: Sequence[EnrichedRequirement],
) -> None:
    if len(reqs) != len(rows):
        raise EnrichmentAlignmentError(
            400,
            "Run /api/enrich for this session before clustering with enriched or hybrid embeddings.",
        )
    for req, row in zip(reqs, rows):
        if not _row_is_valid_for_requirement(row, req):
            raise EnrichmentAlignmentError(
                400,
                "Enrichment alignment mismatch. Run /api/enrich for this session before clustering with enriched or hybrid embeddings.",
            )


def _complete_enrichment_group(
    reqs: Sequence[Requirement],
    rows: Sequence[EnrichedRequirement],
) -> list[EnrichedRequirement] | None:
    req_by_id = {int(req.id): req for req in reqs}
    groups: dict[tuple[str, str, str], dict[int, EnrichedRequirement]] = {}
    group_updated_at: dict[tuple[str, str, str], datetime] = {}

    for row in sorted(
        rows,
        key=lambda item: (
            item.updated_at or item.created_at or datetime.min,
            item.id or 0,
        ),
        reverse=True,
    ):
        req = req_by_id.get(int(row.requirement_db_id))
        if req is None or not _row_is_valid_for_requirement(row, req):
            continue
        group = (row.provider, row.model, row.prompt_version)
        groups.setdefault(group, {})
        groups[group].setdefault(int(row.requirement_db_id), row)
        current_updated = row.updated_at or row.created_at or datetime.min
        previous_updated = group_updated_at.get(group, datetime.min)
        if current_updated > previous_updated:
            group_updated_at[group] = current_updated

    complete_groups = [
        group for group, by_req in groups.items() if len(by_req) == len(reqs)
    ]
    if not complete_groups:
        return None
    selected = sorted(
        complete_groups,
        key=lambda group: group_updated_at.get(group, datetime.min),
        reverse=True,
    )[0]
    rows_by_req = groups[selected]
    return [rows_by_req[int(req.id)] for req in reqs]


def get_enriched_texts_for_session_ordered(
    db: DBSession,
    session_id: int,
) -> list[str]:
    _session, reqs = get_requirements_for_session_ordered(db, session_id)
    rows = (
        db.query(EnrichedRequirement)
        .filter(
            EnrichedRequirement.session_id == session_id,
            EnrichedRequirement.status == SUCCESS_STATUS,
            EnrichedRequirement.prompt_version == PROMPT_VERSION,
        )
        .order_by(EnrichedRequirement.updated_at.desc(), EnrichedRequirement.id.desc())
        .all()
    )
    selected_rows = _complete_enrichment_group(reqs, rows)
    if selected_rows is None:
        raise EnrichmentAlignmentError(
            400,
            "Run /api/enrich for this session before clustering with enriched or hybrid embeddings.",
        )
    validate_enrichment_alignment(reqs, selected_rows)
    return [normalize_plain_text(row.expanded_text or "") for row in selected_rows]


def validate_session_enrichment_ready(
    db: DBSession,
    session_id: int,
) -> None:
    get_enriched_texts_for_session_ordered(db, session_id)


def build_enriched_texts_for_pipeline(
    db: DBSession,
    session_id: int,
    embedding_mode: str,
) -> list[str] | None:
    if embedding_mode == "base":
        return None
    if embedding_mode not in {"enriched", "hybrid"}:
        raise EnrichmentAlignmentError(400, "Invalid embedding mode.")
    return get_enriched_texts_for_session_ordered(db, session_id)
