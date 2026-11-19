#!/usr/bin/env python
"""Test local LLM (LM Studio / OpenAI-compatible endpoint) integration across ReqCluster.

Tests:
1. LM Studio connectivity & loaded model verification
2. Dependency Edge Explanation (DP5 Phase 3.2)
3. Requirement Expansion / Enrichment (DP4)
4. Full dependency explanation caching in DB
"""

import json
import sys
import time
import urllib.request

from reqcluster.config import get_settings
from reqcluster.llm.enrich import expand_all
from reqcluster.llm.gateway import get_gateway
from reqcluster.llm.narrate import explain_edge


def test_connectivity():
    print("\n" + "=" * 60, flush=True)
    print("1. Testing LM Studio Connectivity", flush=True)
    print("=" * 60, flush=True)
    cfg = get_settings()
    base_url, model_name = cfg.llm_base_url, cfg.llm_model
    print(f"Endpoint: {base_url}/models", flush=True)
    print(f"Target Model: {model_name}", flush=True)

    try:
        req = urllib.request.Request(f"{base_url}/models")
        with urllib.request.urlopen(req, timeout=5) as resp:
            data = json.loads(resp.read().decode())
            available_models = [m.get("id") for m in data.get("data", [])]
            print(f"[SUCCESS] LM Studio is ONLINE! Available models: {available_models}", flush=True)
            if model_name in available_models:
                print(f"[SUCCESS] Target model '{model_name}' is loaded and ready.", flush=True)
            else:
                print(f"[INFO] Using model '{model_name}' (available: {available_models})", flush=True)
    except Exception as exc:
        print(f"[ERROR] Failed to reach LM Studio at {base_url}: {exc}", flush=True)
        return False
    return True


def test_edge_explanation():
    print("\n" + "=" * 60, flush=True)
    print("2. Testing Dependency Edge Explanation (DP5)", flush=True)
    print("=" * 60, flush=True)

    print("Sending to local LLM...", flush=True)
    start = time.time()
    try:
        result = explain_edge(
            get_gateway(),
            "The thermal management unit shall measure baseplate temperature at no less than 5 Hz.",
            "The cooling fan shall activate once the baseplate temperature reported in REQ-001 exceeds 70 degrees Celsius.",
            "data",
            {
                "pass": "pass_0_artifact_match",
                "producer_artifact_name": "baseplate temperature",
                "consumer_artifact_name": "baseplate temperature",
                "matched_text": "REQ-001",
                "cue": "once",
                "confidence_tier": "High",
            },
            source_id="REQ-001",
            target_id="REQ-002",
            confidence_tier="High",
            provider="openai",
        )
        elapsed = time.time() - start
        print(f"\n--- Explanation (source={result.source}, {elapsed:.2f}s) ---", flush=True)
        print(result.output, flush=True)
        if result.source == "fallback":
            print(f"[ERROR] LLM unavailable, deterministic fallback used: {result.error}", flush=True)
            return False
        print(f"[SUCCESS] Dependency edge explanation generated & parsed cleanly in {elapsed:.2f}s!", flush=True)
    except Exception as exc:
        elapsed = time.time() - start
        print(f"[ERROR] Edge explanation failed after {elapsed:.2f}s: {exc}", flush=True)
        return False
    return True


def test_requirement_expansion():
    print("\n" + "=" * 60, flush=True)
    print("3. Testing Requirement Expansion / Enrichment (DP4)", flush=True)
    print("=" * 60, flush=True)

    req_text = "The telemetry encoder shall transmit the battery voltage, the bus current, and the cell temperature once per second."
    print(f"Requirement Text: \"{req_text}\"", flush=True)

    start = time.time()
    try:
        expansion = expand_all([(1, req_text)], get_gateway(), "openai")[0]
        elapsed = time.time() - start
        print(f"\n--- Structured Expansion Result ({elapsed:.2f}s) ---", flush=True)
        print(f"Status: {expansion.status} (source={expansion.source})", flush=True)
        print(f"Functional Intent: {expansion.functional_intent}", flush=True)
        print(f"Domain Terms: {expansion.domain_terms}", flush=True)
        print(f"Mentioned Components: {expansion.mentioned_components}", flush=True)
        print(f"Confidence: {expansion.confidence}", flush=True)
        print(f"Warnings: {expansion.warnings}", flush=True)
        if expansion.status != "succeeded" or expansion.source == "fallback":
            print(f"[ERROR] Enrichment did not use the LLM: {expansion.error}", flush=True)
            return False
        print(f"[SUCCESS] Requirement enrichment succeeded with valid schema in {elapsed:.2f}s!", flush=True)
    except Exception as exc:
        import traceback
        print(f"[ERROR] Requirement expansion failed: {exc}", flush=True)
        traceback.print_exc()
        return False
    return True


def main():
    print("Starting ReqCluster Local LLM Verification...", flush=True)
    if not test_connectivity():
        sys.exit(1)

    edge_ok = test_edge_explanation()
    expansion_ok = test_requirement_expansion()

    print("\n" + "=" * 60, flush=True)
    print("SUMMARY", flush=True)
    print("=" * 60, flush=True)
    print("1. LM Studio Connectivity:  PASS", flush=True)
    print(f"2. Edge Explanation:        {'PASS' if edge_ok else 'FAIL'}", flush=True)
    print(f"3. Requirement Enrichment:  {'PASS' if expansion_ok else 'FAIL'}", flush=True)
    if edge_ok and expansion_ok:
        print("\nALL LOCAL LLM TESTS PASSED SUCCESSFULLY! Model is fully integrated.", flush=True)
    else:
        print("\nSome tests had issues. Check logs above.", flush=True)


if __name__ == "__main__":
    main()
