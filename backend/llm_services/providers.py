from __future__ import annotations

import asyncio
import json
import math
import os
import re
import urllib.error
import urllib.parse
import urllib.request
from dataclasses import asdict, dataclass
from typing import Protocol, Sequence

from .prompts import build_requirement_expansion_prompt, normalize_domain_terms, normalize_plain_text
from .vocabulary import extract_domain_vocabulary


MAX_RESPONSE_CHARS = 64_000
MAX_EXPANDED_TEXT_CHARS = 12_000
MAX_SHORT_STRING_CHARS = 1_000
MAX_LIST_ITEM_CHARS = 160
MAX_DOMAIN_TERMS = 50
MAX_COMPONENTS = 50
MAX_ASSUMPTIONS = 20
MAX_WARNINGS = 30

_CONTROL_CHARS_RE = re.compile(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]")
_COMPONENT_RE = re.compile(r"\b(?:[A-Z][A-Za-z0-9-]{2,}|[A-Z]{2,}(?:-[A-Z0-9]+)*)\b")


class LLMProviderError(RuntimeError):
    """Base class for safe provider failures."""


class ProviderConfigurationError(LLMProviderError):
    """Raised when a provider is not configured safely."""


class ProviderResponseError(LLMProviderError):
    """Raised when a provider returns an invalid response."""


@dataclass(frozen=True)
class ParsedExpansion:
    expanded_text: str
    domain_terms: list[str]
    functional_intent: str
    mentioned_components: list[str]
    assumptions: list[str]
    confidence: float
    warnings: list[str]

    def to_dict(self) -> dict:
        return asdict(self)


class RequirementExpansionProvider(Protocol):
    name: str
    model: str

    async def expand_requirement(
        self,
        requirement_text: str,
        requirement_id: str | None = None,
        domain_vocabulary: Sequence[str] | None = None,
        timeout_seconds: float | None = None,
    ) -> ParsedExpansion:
        ...


def _bounded_int(value: str | None, default: int, low: int, high: int) -> int:
    try:
        parsed = int(value) if value is not None else default
    except (TypeError, ValueError):
        parsed = default
    return min(max(parsed, low), high)


def _bounded_float(value: str | None, default: float, low: float, high: float) -> float:
    try:
        parsed = float(value) if value is not None else default
    except (TypeError, ValueError):
        parsed = default
    return min(max(parsed, low), high)


def _sanitize_string(value: object, max_chars: int) -> tuple[str, bool]:
    text = normalize_plain_text(value, max_chars=0)
    truncated = len(text) > max_chars
    return text[:max_chars], truncated


def _parse_string_field(data: dict, field: str, max_chars: int, warnings: list[str]) -> str:
    value = data.get(field, "")
    if not isinstance(value, str):
        raise ProviderResponseError(f"Field '{field}' must be a string.")
    text, truncated = _sanitize_string(value, max_chars)
    if truncated:
        warnings.append(f"Field '{field}' was truncated to the maximum allowed length.")
    return text


def _parse_string_list(
    data: dict,
    field: str,
    max_items: int,
    item_max_chars: int,
    warnings: list[str],
) -> list[str]:
    value = data.get(field, [])
    if not isinstance(value, list):
        raise ProviderResponseError(f"Field '{field}' must be a list of strings.")

    result: list[str] = []
    seen: set[str] = set()
    for item in value:
        if not isinstance(item, str):
            warnings.append(f"Non-string item ignored in '{field}'.")
            continue
        cleaned, truncated = _sanitize_string(item, item_max_chars)
        if truncated:
            warnings.append(f"An item in '{field}' was truncated.")
        if not cleaned:
            continue
        key = cleaned.casefold()
        if key in seen:
            continue
        seen.add(key)
        result.append(cleaned)
        if len(result) >= max_items:
            if len(value) > max_items:
                warnings.append(f"Field '{field}' was truncated to {max_items} items.")
            break
    return result


def parse_expansion_response(raw_response: str) -> ParsedExpansion:
    """
    Strictly parse an untrusted LLM JSON response.

    This parser intentionally uses json.loads only. It rejects Markdown fences,
    top-level arrays, and huge responses before validating and bounding fields.
    """
    if not isinstance(raw_response, str):
        raise ProviderResponseError("LLM response must be a string.")
    if len(raw_response) > MAX_RESPONSE_CHARS:
        raise ProviderResponseError("LLM response exceeded the maximum allowed size.")
    if "```" in raw_response:
        raise ProviderResponseError("Markdown fenced JSON is not accepted.")

    raw = _CONTROL_CHARS_RE.sub("", raw_response).strip()
    if not raw:
        raise ProviderResponseError("LLM response was empty.")

    try:
        data = json.loads(raw)
    except json.JSONDecodeError as exc:
        raise ProviderResponseError("LLM response was not valid JSON.") from exc

    if isinstance(data, list):
        raise ProviderResponseError("Top-level JSON arrays are not accepted.")
    if not isinstance(data, dict):
        raise ProviderResponseError("LLM response must be a JSON object.")

    required_fields = {
        "expanded_text",
        "domain_terms",
        "functional_intent",
        "mentioned_components",
        "assumptions",
        "confidence",
        "warnings",
    }
    missing = sorted(required_fields - set(data.keys()))
    if missing:
        raise ProviderResponseError(f"LLM response missing required fields: {', '.join(missing)}.")

    parser_warnings: list[str] = []
    expanded_text = _parse_string_field(
        data,
        "expanded_text",
        MAX_EXPANDED_TEXT_CHARS,
        parser_warnings,
    )
    if not expanded_text:
        raise ProviderResponseError("Field 'expanded_text' must not be empty.")

    functional_intent = _parse_string_field(
        data,
        "functional_intent",
        MAX_SHORT_STRING_CHARS,
        parser_warnings,
    )
    domain_terms = normalize_domain_terms(
        _parse_string_list(
            data,
            "domain_terms",
            MAX_DOMAIN_TERMS,
            MAX_LIST_ITEM_CHARS,
            parser_warnings,
        ),
        max_terms=MAX_DOMAIN_TERMS,
    )
    mentioned_components = _parse_string_list(
        data,
        "mentioned_components",
        MAX_COMPONENTS,
        MAX_LIST_ITEM_CHARS,
        parser_warnings,
    )
    assumptions = _parse_string_list(
        data,
        "assumptions",
        MAX_ASSUMPTIONS,
        MAX_SHORT_STRING_CHARS,
        parser_warnings,
    )
    response_warnings = _parse_string_list(
        data,
        "warnings",
        MAX_WARNINGS,
        MAX_SHORT_STRING_CHARS,
        parser_warnings,
    )

    confidence_raw = data.get("confidence", 0.0)
    if isinstance(confidence_raw, bool) or not isinstance(confidence_raw, (int, float)):
        confidence = 0.0
        parser_warnings.append("Invalid confidence value; set to 0.0.")
    else:
        confidence_float = float(confidence_raw)
        if not math.isfinite(confidence_float):
            confidence = 0.0
            parser_warnings.append("Non-finite confidence value; set to 0.0.")
        else:
            confidence = min(max(confidence_float, 0.0), 1.0)
            if confidence != confidence_float:
                parser_warnings.append("Confidence value was clamped to the 0..1 range.")

    return ParsedExpansion(
        expanded_text=expanded_text,
        domain_terms=domain_terms,
        functional_intent=functional_intent,
        mentioned_components=mentioned_components,
        assumptions=assumptions,
        confidence=round(float(confidence), 6),
        warnings=_dedupe(response_warnings + parser_warnings, max_items=MAX_WARNINGS),
    )


def _dedupe(values: Sequence[str], max_items: int | None = None) -> list[str]:
    result: list[str] = []
    seen: set[str] = set()
    for value in values:
        cleaned = normalize_plain_text(value, max_chars=MAX_SHORT_STRING_CHARS)
        if not cleaned:
            continue
        key = cleaned.casefold()
        if key in seen:
            continue
        seen.add(key)
        result.append(cleaned)
        if max_items is not None and len(result) >= max_items:
            break
    return result


def _extract_components(text: str, terms: Sequence[str]) -> list[str]:
    components = [match.group(0) for match in _COMPONENT_RE.finditer(text)]
    components.extend(term for term in terms if any(part in term for part in ("module", "service", "api", "sensor", "controller")))
    return _dedupe(components, max_items=MAX_COMPONENTS)


def _derive_intent(text: str) -> str:
    cleaned = normalize_plain_text(text, max_chars=MAX_SHORT_STRING_CHARS)
    lowered = cleaned.lower()
    prefixes = (
        "the system shall ",
        "system shall ",
        "the system must ",
        "system must ",
        "the system should ",
        "system should ",
        "the system may ",
        "system may ",
    )
    for prefix in prefixes:
        if lowered.startswith(prefix):
            return cleaned[len(prefix) :].strip(" .") or cleaned
    return cleaned


class MockLLMProvider:
    """Deterministic, offline provider for tests and local development."""

    name = "mock"
    model = "mock-deterministic-v1"

    async def expand_requirement(
        self,
        requirement_text: str,
        requirement_id: str | None = None,
        domain_vocabulary: Sequence[str] | None = None,
        timeout_seconds: float | None = None,
    ) -> ParsedExpansion:
        del requirement_id, timeout_seconds
        original = normalize_plain_text(requirement_text)
        if not original:
            raise ProviderResponseError("Requirement text is empty.")

        local_terms = extract_domain_vocabulary([original], top_n=12)
        if not isinstance(local_terms, list):
            local_terms = []
        original_lower = original.lower()
        supplied_terms = [
            term
            for term in normalize_domain_terms(domain_vocabulary)
            if term in original_lower
        ]
        domain_terms = _dedupe([*supplied_terms, *[str(term) for term in local_terms]], max_items=MAX_DOMAIN_TERMS)
        components = _extract_components(original, domain_terms)
        intent = _derive_intent(original)

        parts = [f"Original requirement: {original}"]
        if intent and intent != original:
            parts.append(f"Functional intent: {intent}.")
        if components:
            parts.append(f"Explicitly mentioned components: {', '.join(components)}.")
        if domain_terms:
            parts.append(f"Domain vocabulary from the requirement: {', '.join(domain_terms[:12])}.")

        payload = {
            "expanded_text": " ".join(parts),
            "domain_terms": domain_terms,
            "functional_intent": intent,
            "mentioned_components": components,
            "assumptions": [],
            "confidence": 0.85,
            "warnings": [],
        }
        return parse_expansion_response(json.dumps(payload, ensure_ascii=False))


def _validate_http_url(value: str, env_name: str) -> str:
    parsed = urllib.parse.urlparse(value)
    if parsed.scheme not in {"http", "https"}:
        raise ProviderConfigurationError(f"{env_name} must use http or https.")
    if not parsed.netloc:
        raise ProviderConfigurationError(f"{env_name} must include a host.")
    return value.rstrip("/")


def _post_json(url: str, payload: dict, headers: dict, timeout_seconds: float) -> dict:
    req_headers = dict(headers)
    req_headers.setdefault(
        "User-Agent",
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
    )
    request = urllib.request.Request(
        url,
        data=json.dumps(payload).encode("utf-8"),
        headers=req_headers,
        method="POST",
    )
    try:
        with urllib.request.urlopen(request, timeout=timeout_seconds) as response:
            body = response.read(MAX_RESPONSE_CHARS + 1)
    except urllib.error.HTTPError as exc:
        raise ProviderResponseError(f"LLM provider HTTP error: {exc.code}.") from exc
    except urllib.error.URLError as exc:
        raise ProviderResponseError("LLM provider request failed.") from exc
    except TimeoutError as exc:
        raise ProviderResponseError("LLM provider request timed out.") from exc

    if len(body) > MAX_RESPONSE_CHARS:
        raise ProviderResponseError("LLM provider response exceeded the maximum allowed size.")
    try:
        parsed = json.loads(body.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ProviderResponseError("LLM provider response was not valid JSON.") from exc
    if not isinstance(parsed, dict):
        raise ProviderResponseError("LLM provider response must be a JSON object.")
    return parsed


def _extract_response_content(response: dict) -> str:
    if all(key in response for key in ("expanded_text", "domain_terms", "functional_intent")):
        return json.dumps(response, ensure_ascii=False)

    choices = response.get("choices")
    if isinstance(choices, list) and choices:
        first = choices[0]
        if isinstance(first, dict):
            message = first.get("message")
            if isinstance(message, dict) and isinstance(message.get("content"), str):
                return message["content"]
            if isinstance(first.get("text"), str):
                return first["text"]

    for key in ("response", "content", "text", "output"):
        value = response.get(key)
        if isinstance(value, str):
            return value
        if isinstance(value, dict):
            return json.dumps(value, ensure_ascii=False)

    raise ProviderResponseError("LLM provider response did not contain expansion content.")


class OpenAICompatibleProvider:
    name = "openai-compatible"

    def __init__(
        self,
        base_url: str,
        api_key: str,
        model: str,
        timeout_seconds: float = 30.0,
        max_retries: int = 2,
    ) -> None:
        self.base_url = _validate_http_url(base_url, "REQCLUSTER_LLM_BASE_URL")
        if not api_key:
            raise ProviderConfigurationError("REQCLUSTER_LLM_API_KEY is required.")
        if not model:
            raise ProviderConfigurationError("REQCLUSTER_LLM_MODEL is required.")
        self.api_key = api_key
        self.model = normalize_plain_text(model, max_chars=160)
        self.timeout_seconds = _bounded_float(str(timeout_seconds), 30.0, 1.0, 300.0)
        self.max_retries = _bounded_int(str(max_retries), 2, 0, 5)

    @classmethod
    def from_env(cls) -> "OpenAICompatibleProvider":
        return cls(
            base_url=os.getenv("REQCLUSTER_LLM_BASE_URL", ""),
            api_key=os.getenv("REQCLUSTER_LLM_API_KEY", ""),
            model=os.getenv("REQCLUSTER_LLM_MODEL", ""),
            timeout_seconds=_bounded_float(
                os.getenv("REQCLUSTER_LLM_TIMEOUT_SECONDS"),
                30.0,
                1.0,
                300.0,
            ),
            max_retries=_bounded_int(os.getenv("REQCLUSTER_LLM_MAX_RETRIES"), 2, 0, 5),
        )

    async def expand_requirement(
        self,
        requirement_text: str,
        requirement_id: str | None = None,
        domain_vocabulary: Sequence[str] | None = None,
        timeout_seconds: float | None = None,
    ) -> ParsedExpansion:
        del requirement_id
        prompt = build_requirement_expansion_prompt(requirement_text, domain_vocabulary)
        timeout = _bounded_float(
            str(timeout_seconds) if timeout_seconds is not None else None,
            self.timeout_seconds,
            1.0,
            300.0,
        )
        payload = {
            "model": self.model,
            "messages": [
                {"role": "system", "content": "You return conservative strict JSON only."},
                {"role": "user", "content": prompt},
            ],
            "temperature": 0,
            "response_format": {"type": "json_object"},
        }
        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {self.api_key}",
        }
        endpoint = f"{self.base_url}/chat/completions"

        last_error: Exception | None = None
        for attempt in range(self.max_retries + 1):
            try:
                response = await asyncio.to_thread(_post_json, endpoint, payload, headers, timeout)
                return parse_expansion_response(_extract_response_content(response))
            except ProviderResponseError as exc:
                last_error = exc
                if attempt >= self.max_retries:
                    break
                await asyncio.sleep(min(0.25 * (attempt + 1), 1.0))
        raise ProviderResponseError("OpenAI-compatible provider failed after bounded retries.") from last_error


class LocalLLMProvider:
    name = "local"

    def __init__(self, url: str, model: str = "", timeout_seconds: float = 30.0) -> None:
        self.url = _validate_http_url(url, "REQCLUSTER_LOCAL_LLM_URL")
        self.model = normalize_plain_text(model or "local-model", max_chars=160)
        self.timeout_seconds = _bounded_float(str(timeout_seconds), 30.0, 1.0, 300.0)

    @classmethod
    def from_env(cls) -> "LocalLLMProvider":
        return cls(
            url=os.getenv("REQCLUSTER_LOCAL_LLM_URL", ""),
            model=os.getenv("REQCLUSTER_LOCAL_LLM_MODEL", "local-model"),
            timeout_seconds=_bounded_float(
                os.getenv("REQCLUSTER_LOCAL_LLM_TIMEOUT_SECONDS"),
                30.0,
                1.0,
                300.0,
            ),
        )

    async def expand_requirement(
        self,
        requirement_text: str,
        requirement_id: str | None = None,
        domain_vocabulary: Sequence[str] | None = None,
        timeout_seconds: float | None = None,
    ) -> ParsedExpansion:
        del requirement_id
        prompt = build_requirement_expansion_prompt(requirement_text, domain_vocabulary)
        timeout = _bounded_float(
            str(timeout_seconds) if timeout_seconds is not None else None,
            self.timeout_seconds,
            1.0,
            300.0,
        )
        payload = {
            "model": self.model,
            "prompt": prompt,
            "temperature": 0,
            "stream": False,
        }
        headers = {"Content-Type": "application/json"}
        response = await asyncio.to_thread(_post_json, self.url, payload, headers, timeout)
        return parse_expansion_response(_extract_response_content(response))


def _extract_text_content(response: dict) -> str:
    choices = response.get("choices")
    if isinstance(choices, list) and choices:
        first = choices[0]
        if isinstance(first, dict):
            message = first.get("message")
            if isinstance(message, dict) and isinstance(message.get("content"), str):
                return message["content"]
            if isinstance(first.get("text"), str):
                return first["text"]
    for key in ("response", "content", "text", "output", "message"):
        value = response.get(key)
        if isinstance(value, str):
            return value
    raise ProviderResponseError("LLM provider response did not contain text content.")


def generate_completion(
    prompt: str,
    provider_name: str | None = None,
    *,
    system: str = "You are a precise systems-engineering analyst. Reply with concise plain prose, no markdown.",
    timeout_seconds: float | None = None,
    max_chars: int = 1200,
) -> str:
    """Free-text completion via an on-prem/openai-compatible LLM.

    Used for narrative summaries and rationales (e.g. Qwen served by Ollama or
    an OpenAI-compatible gateway). Synchronous; callers running on the event
    loop should wrap this in a worker thread. Raises ``LLMProviderError`` on any
    configuration or response problem so callers can fall back deterministically.
    """
    selected = normalize_plain_text(
        provider_name or os.getenv("REQCLUSTER_LLM_PROVIDER") or "mock",
        max_chars=80,
    ).lower()
    timeout = _bounded_float(
        str(timeout_seconds) if timeout_seconds is not None else None, 30.0, 1.0, 300.0
    )

    if selected in {"openai", "openai-compatible", "openai_compatible"}:
        base_url = _validate_http_url(
            os.getenv("REQCLUSTER_LLM_BASE_URL", ""), "REQCLUSTER_LLM_BASE_URL"
        )
        api_key = os.getenv("REQCLUSTER_LLM_API_KEY", "")
        model = os.getenv("REQCLUSTER_LLM_MODEL", "")
        if not api_key or not model:
            raise ProviderConfigurationError(
                "REQCLUSTER_LLM_API_KEY and REQCLUSTER_LLM_MODEL are required."
            )
        payload = {
            "model": model,
            "messages": [
                {"role": "system", "content": system},
                {"role": "user", "content": prompt},
            ],
            "temperature": 0,
        }
        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {api_key}",
        }
        endpoint = f"{base_url}/chat/completions"
    elif selected in {"local", "local-llm", "local_llm"}:
        endpoint = _validate_http_url(
            os.getenv("REQCLUSTER_LOCAL_LLM_URL", ""), "REQCLUSTER_LOCAL_LLM_URL"
        )
        model = os.getenv("REQCLUSTER_LOCAL_LLM_MODEL", "local-model")
        payload = {
            "model": model,
            "prompt": prompt,
            "system": system,
            "temperature": 0,
            "stream": False,
        }
        headers = {"Content-Type": "application/json"}
    else:
        raise ProviderConfigurationError(
            "Free-text generation requires the 'openai'/'openai-compatible' or 'local' provider."
        )

    response = _post_json(endpoint, payload, headers, timeout)
    text = normalize_plain_text(_extract_text_content(response), max_chars=max_chars)
    if not text:
        raise ProviderResponseError("LLM provider returned empty text.")
    return text


def get_provider(provider_name: str | None = "mock") -> RequirementExpansionProvider:
    selected = normalize_plain_text(
        provider_name or os.getenv("REQCLUSTER_LLM_PROVIDER") or "mock",
        max_chars=80,
    ).lower()
    if selected in {"mock", "offline", "test"}:
        return MockLLMProvider()
    if selected in {"openai", "openai-compatible", "openai_compatible"}:
        return OpenAICompatibleProvider.from_env()
    if selected in {"local", "local-llm", "local_llm"}:
        return LocalLLMProvider.from_env()
    raise ProviderConfigurationError("Unsupported LLM provider.")
