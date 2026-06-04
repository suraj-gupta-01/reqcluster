"""LLM enrichment helpers for Phase 2 requirement expansion."""

from .enrichment import (
    EnrichmentBatchResult,
    RequirementExpansionResult,
    enrich_requirements,
    enrich_then_prepare_pipeline_inputs,
    extract_enriched_texts,
)
from .providers import (
    LocalLLMProvider,
    MockLLMProvider,
    OpenAICompatibleProvider,
    ParsedExpansion,
    parse_expansion_response,
)

__all__ = [
    "EnrichmentBatchResult",
    "RequirementExpansionResult",
    "enrich_requirements",
    "enrich_then_prepare_pipeline_inputs",
    "extract_enriched_texts",
    "LocalLLMProvider",
    "MockLLMProvider",
    "OpenAICompatibleProvider",
    "ParsedExpansion",
    "parse_expansion_response",
]
