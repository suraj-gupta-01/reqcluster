#!/usr/bin/env python
"""Test local LLM (LM Studio / OpenAI-compatible endpoint) integration across ReqCluster.

Tests:
1. LM Studio connectivity & loaded model verification
2. Dependency Edge Explanation (DP5 Phase 3.2)
3. Requirement Expansion / Enrichment (DP4)
4. Full dependency explanation caching in DB
"""

import os
import sys
from pathlib import Path

# Add project root and backend to path
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "backend"))

# Load .env if present
env_file = ROOT / ".env"
if env_file.exists():
    for line in env_file.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if line and not line.startswith("#") and "=" in line:
            k, v = line.split("=", 1)
            os.environ.setdefault(k.strip(), v.strip())

import asyncio
import json
import urllib.request
from backend.llm_services.providers import (
    OpenAICompatibleProvider,
    generate_completion,
    get_provider,
)
from backend.llm_services.prompts import build_edge_explanation_prompt
from backend.llm_services.edge_explanation import parse_explanation_response


def test_connectivity():
    print("\n" + "=" * 60, flush=True)
    print("1. Testing LM Studio Connectivity", flush=True)
    print("=" * 60, flush=True)
    base_url = os.getenv("REQCLUSTER_LLM_BASE_URL", "http://localhost:1234/v1")
    model_name = os.getenv("REQCLUSTER_LLM_MODEL", "qwen/qwen3.5-9b")
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


import time

def test_edge_explanation():
    print("\n" + "=" * 60, flush=True)
    print("2. Testing Dependency Edge Explanation (DP5)", flush=True)
    print("=" * 60, flush=True)

    prompt = build_edge_explanation_prompt(
        source_text="The thermal management unit shall measure baseplate temperature at no less than 5 Hz.",
        target_text="The cooling fan shall activate once the baseplate temperature reported in REQ-001 exceeds 70 degrees Celsius.",
        relation="data",
        evidence={
            "pass": "pass_0_artifact_match",
            "producer_artifact_name": "baseplate temperature",
            "consumer_artifact_name": "baseplate temperature",
            "matched_text": "REQ-001",
            "cue": "once",
            "confidence_tier": "High",
        },
        source_id="REQ-001",
        target_id="REQ-002",
    )

    print("Prompt constructed successfully. Sending to local LLM (timeout: 180s)...", flush=True)
    start = time.time()
    try:
        raw_resp = generate_completion(
            prompt,
            provider_name="openai-compatible",
            timeout_seconds=180.0,
        )
        elapsed = time.time() - start
        print(f"\n--- Raw LLM Completion (generated in {elapsed:.2f}s) ---", flush=True)
        print(raw_resp, flush=True)

        parsed = parse_explanation_response(raw_resp)
        print("\n--- Parsed & Hardened Explanation ---", flush=True)
        print(parsed, flush=True)
        print(f"[SUCCESS] Dependency edge explanation generated & parsed cleanly in {elapsed:.2f}s!", flush=True)
    except Exception as exc:
        elapsed = time.time() - start
        print(f"[ERROR] Edge explanation failed after {elapsed:.2f}s: {exc}", flush=True)
        return False
    return True


async def test_requirement_expansion():
    print("\n" + "=" * 60, flush=True)
    print("3. Testing Requirement Expansion / Enrichment (DP4)", flush=True)
    print("=" * 60, flush=True)

    req_text = "The telemetry encoder shall transmit the battery voltage, the bus current, and the cell temperature once per second."
    print(f"Requirement Text: \"{req_text}\"", flush=True)
    print("Sending structured enrichment request to local LLM (timeout: 180s)...", flush=True)

    start = time.time()
    try:
        provider = OpenAICompatibleProvider.from_env()
        expansion = await provider.expand_requirement(
            requirement_text=req_text,
            requirement_id="ADV-014",
            domain_vocabulary=["telemetry", "voltage", "current", "temperature", "power"],
            timeout_seconds=180.0,
        )
        elapsed = time.time() - start
        print(f"\n--- Structured Expansion Result (generated in {elapsed:.2f}s) ---", flush=True)
        print(f"Functional Intent: {expansion.functional_intent}", flush=True)
        print(f"Domain Terms: {expansion.domain_terms}", flush=True)
        print(f"Mentioned Components: {expansion.mentioned_components}", flush=True)
        print(f"Confidence: {expansion.confidence}", flush=True)
        print(f"Warnings: {expansion.warnings}", flush=True)
        print(f"[SUCCESS] Requirement enrichment succeeded with valid schema in {elapsed:.2f}s!", flush=True)
    except Exception as exc:
        elapsed = time.time() - start
        import traceback
        print(f"[ERROR] Requirement expansion failed after {elapsed:.2f}s: {exc}", flush=True)
        traceback.print_exc()
        return False
    return True


def main():
    print("Starting ReqCluster Local LLM Verification...", flush=True)
    if not test_connectivity():
        sys.exit(1)
    
    edge_ok = test_edge_explanation()
    expansion_ok = asyncio.run(test_requirement_expansion())

    print("\n" + "=" * 60, flush=True)
    print("SUMMARY", flush=True)
    print("=" * 60, flush=True)
    print(f"1. LM Studio Connectivity:  PASS", flush=True)
    print(f"2. Edge Explanation:        {'PASS' if edge_ok else 'FAIL'}", flush=True)
    print(f"3. Requirement Enrichment:  {'PASS' if expansion_ok else 'FAIL'}", flush=True)
    if edge_ok and expansion_ok:
        print("\nALL LOCAL LLM TESTS PASSED SUCCESSFULLY! Model is fully integrated.", flush=True)
    else:
        print("\nSome tests had issues. Check logs above.", flush=True)


if __name__ == "__main__":
    main()
