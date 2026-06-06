"""ReqCluster HTTP stress smoke test.

Uploads a generated CSV, runs clustering, then measures read endpoint latency.
The script uses only the Python standard library so it can run in constrained
audit environments.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import random
import statistics
import tempfile
import time
import urllib.error
import urllib.parse
import urllib.request
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path


MODULES = [
    ("Command Authentication", "Command Freshness"),
    ("Mission Management", "Waypoint Validation"),
    ("Navigation", "GPS IRS Sensor Fusion"),
    ("Guidance", "LNAV VNAV"),
    ("Flight Control", "Automatic Mode"),
    ("Power Management", "Battery Readiness"),
    ("Communications", "Lost Link"),
    ("Payload Control", "Payload Constraints"),
    ("Health Monitoring", "POST Self Test"),
    ("Software Update", "Update Safety"),
    ("Telemetry", "Telemetry Generation"),
    ("FMS", "Fuel Displays"),
]

STEMS = [
    "validate state flags before enabling the controlled function",
    "reject stale inputs when the configured timer expires",
    "publish readiness status to the dependent subsystem",
    "log rejected requests with source timestamp and sequence number",
    "maintain the active buffer until replacement data is verified",
    "raise degraded status when required inputs are unavailable",
    "compute guidance outputs from the selected navigation source",
    "inhibit launch when configured safety limits are exceeded",
]


def percentile(values: list[float], pct: float) -> float:
    if not values:
        return math.nan
    ordered = sorted(values)
    idx = min(len(ordered) - 1, max(0, math.ceil((pct / 100.0) * len(ordered)) - 1))
    return ordered[idx]


def make_csv(path: Path, requirements: int, pages: int) -> None:
    random.seed(42)
    fields = ["id", "text", "module", "section", "page", "source_doc", "requirement_type", "criticality"]
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for idx in range(requirements):
            module, section = MODULES[idx % len(MODULES)]
            stem = STEMS[idx % len(STEMS)]
            page = 1 + (idx % max(pages, 1))
            writer.writerow({
                "id": f"LOAD-{idx + 1:05d}",
                "text": (
                    f"The {module.lower()} function shall {stem} for daily batch "
                    f"{1 + idx // 250} page {page} buffer {(idx % 29) + 1}."
                ),
                "module": module,
                "section": section,
                "page": page,
                "source_doc": "stress_test_reqcluster",
                "requirement_type": "functional",
                "criticality": "high" if module in {"Flight Control", "Power Management", "Software Update"} else "medium",
            })


def request_json(method: str, url: str, *, data: bytes | None = None, headers: dict[str, str] | None = None, timeout: float = 300.0) -> tuple[int, object]:
    req = urllib.request.Request(url, data=data, headers=headers or {}, method=method)
    try:
        with urllib.request.urlopen(req, timeout=timeout) as response:
            body = response.read()
            parsed = json.loads(body.decode("utf-8")) if body else None
            return response.status, parsed
    except urllib.error.HTTPError as exc:
        body = exc.read().decode("utf-8", errors="replace")
        try:
            parsed = json.loads(body)
        except json.JSONDecodeError:
            parsed = body[:500]
        return exc.code, parsed
    except TimeoutError:
        return 599, {"detail": "request timed out"}
    except urllib.error.URLError as exc:
        return 598, {"detail": str(exc.reason)[:500]}


def upload_csv(base_url: str, csv_path: Path) -> dict:
    boundary = "----ReqClusterStressBoundary"
    content = csv_path.read_bytes()
    body = (
        f"--{boundary}\r\n"
        f'Content-Disposition: form-data; name="file"; filename="{csv_path.name}"\r\n'
        "Content-Type: text/csv\r\n\r\n"
    ).encode("utf-8") + content + f"\r\n--{boundary}--\r\n".encode("utf-8")
    status, parsed = request_json(
        "POST",
        f"{base_url}/api/upload",
        data=body,
        headers={"Content-Type": f"multipart/form-data; boundary={boundary}"},
    )
    if status >= 400:
        raise RuntimeError(f"upload failed: HTTP {status}: {parsed}")
    return parsed


def timed_call(method: str, url: str, **kwargs) -> tuple[float, int, object]:
    start = time.perf_counter()
    status, parsed = request_json(method, url, **kwargs)
    return (time.perf_counter() - start) * 1000.0, status, parsed


def run(args: argparse.Namespace) -> dict:
    base_url = args.base_url.rstrip("/")
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    with tempfile.TemporaryDirectory() as tmp:
        csv_path = Path(tmp) / f"reqcluster_stress_{args.requirements}.csv"
        make_csv(csv_path, args.requirements, args.pages)

        _health_ms, health_status, health_body = timed_call("GET", f"{base_url}/health", timeout=10.0)
        if health_status >= 400:
            raise RuntimeError(f"health check failed: HTTP {health_status}: {health_body}")
        start = time.perf_counter()
        upload = upload_csv(base_url, csv_path)
        upload_duration_ms = (time.perf_counter() - start) * 1000.0
        session_id = int(upload["session_id"])

    cluster_payload = json.dumps({
        "session_id": session_id,
        "embedding_mode": "base",
        "similarity_threshold": args.similarity_threshold,
    }).encode("utf-8")
    cluster_ms, cluster_status, cluster_body = timed_call(
        "POST",
        f"{base_url}/api/cluster",
        data=cluster_payload,
        headers={"Content-Type": "application/json"},
        timeout=args.cluster_timeout,
    )

    endpoints = [
        f"{base_url}/api/sessions/{session_id}",
        f"{base_url}/api/requirements?session_id={session_id}",
        f"{base_url}/api/clusters?session_id={session_id}",
        f"{base_url}/api/graph?session_id={session_id}",
    ]
    latencies: list[float] = []
    failures: list[dict] = []
    with ThreadPoolExecutor(max_workers=args.concurrency) as pool:
        futures = [
            pool.submit(timed_call, "GET", endpoints[idx % len(endpoints)], timeout=args.read_timeout)
            for idx in range(args.read_requests)
        ]
        for future in as_completed(futures):
            ms, status, body = future.result()
            latencies.append(ms)
            if status >= 400:
                failures.append({"status": status, "body": body})

    result = {
        "base_url": base_url,
        "session_id": session_id,
        "requirements": args.requirements,
        "pages": args.pages,
        "upload_ms": round(upload_duration_ms, 3),
        "cluster_ms": round(cluster_ms, 3),
        "cluster_status": cluster_status,
        "cluster_summary": cluster_body if isinstance(cluster_body, dict) else {"body": cluster_body},
        "read_requests": args.read_requests,
        "concurrency": args.concurrency,
        "read_failures": len(failures),
        "read_latency_ms": {
            "min": round(min(latencies), 3) if latencies else None,
            "mean": round(statistics.fmean(latencies), 3) if latencies else None,
            "p50": round(percentile(latencies, 50), 3) if latencies else None,
            "p95": round(percentile(latencies, 95), 3) if latencies else None,
            "max": round(max(latencies), 3) if latencies else None,
        },
        "failures_sample": failures[:10],
    }

    timestamp = time.strftime("%Y%m%d-%H%M%S")
    json_path = out_dir / f"stress_{args.requirements}_{timestamp}.json"
    json_path.write_text(json.dumps(result, indent=2), encoding="utf-8")
    md_path = out_dir / f"stress_{args.requirements}_{timestamp}.md"
    md_path.write_text(
        "\n".join([
            "# ReqCluster Stress Result",
            "",
            f"- requirements: {args.requirements}",
            f"- upload_ms: {result['upload_ms']}",
            f"- cluster_ms: {result['cluster_ms']}",
            f"- read_requests: {args.read_requests}",
            f"- concurrency: {args.concurrency}",
            f"- read_failures: {result['read_failures']}",
            f"- p95_read_latency_ms: {result['read_latency_ms']['p95']}",
        ]),
        encoding="utf-8",
    )
    result["result_files"] = [str(json_path), str(md_path)]
    return result


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Stress test ReqCluster upload, clustering, and read endpoints.")
    parser.add_argument("--base-url", default="http://127.0.0.1:8000")
    parser.add_argument("--requirements", type=int, default=600)
    parser.add_argument("--pages", type=int, default=160)
    parser.add_argument("--read-requests", type=int, default=300)
    parser.add_argument("--concurrency", type=int, default=32)
    parser.add_argument("--similarity-threshold", type=float, default=0.65)
    parser.add_argument("--cluster-timeout", type=float, default=1800.0)
    parser.add_argument("--read-timeout", type=float, default=30.0)
    parser.add_argument("--output-dir", default="stress-results")
    args = parser.parse_args()
    if args.requirements < 1 or args.requirements > 100000:
        raise SystemExit("--requirements must be in 1..100000")
    if args.read_requests < 1 or args.concurrency < 1:
        raise SystemExit("--read-requests and --concurrency must be positive")
    return args


if __name__ == "__main__":
    print(json.dumps(run(parse_args()), indent=2))
