"""Simple concurrent load test for the `/chat` endpoint.

Usage examples:
    python scripts/load_test_chat.py --url http://127.0.0.1:8000/chat
    python scripts/load_test_chat.py --users 10 --requests 30 --question "什么是牛顿第二定律？"
"""

from __future__ import annotations

import argparse
import json
import statistics
import time
from dataclasses import dataclass
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import List
from urllib import error, request


@dataclass
class RequestResult:
    ok: bool
    status_code: int
    latency_ms: float
    error: str = ""


def _single_request(
    url: str,
    payload: dict,
    request_id: int,
    timeout_s: float,
) -> RequestResult:
    start = time.perf_counter()
    body = json.dumps(payload).encode("utf-8")
    req = request.Request(
        url,
        data=body,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    try:
        with request.urlopen(req, timeout=timeout_s) as response:
            status_code = response.getcode()
            response_body = response.read().decode("utf-8", errors="replace")
        latency_ms = (time.perf_counter() - start) * 1000
        ok = status_code == 200
        error = "" if ok else response_body.strip().replace("\n", " ")[:200]
        return RequestResult(
            ok=ok,
            status_code=status_code,
            latency_ms=latency_ms,
            error=error,
        )
    except error.HTTPError as exc:
        latency_ms = (time.perf_counter() - start) * 1000
        response_body = exc.read().decode("utf-8", errors="replace")
        return RequestResult(
            ok=False,
            status_code=exc.code,
            latency_ms=latency_ms,
            error=response_body.strip().replace("\n", " ")[:200],
        )
    except Exception as exc:
        latency_ms = (time.perf_counter() - start) * 1000
        return RequestResult(
            ok=False,
            status_code=0,
            latency_ms=latency_ms,
            error=f"{type(exc).__name__}: {exc}",
        )

def _percentile(values: List[float], pct: float) -> float:
    if not values:
        return 0.0
    if len(values) == 1:
        return values[0]
    ordered = sorted(values)
    index = round((len(ordered) - 1) * pct)
    return ordered[index]


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Concurrent load test for /chat")
    parser.add_argument("--url", default="http://127.0.0.1:8000/chat", help="Chat API URL")
    parser.add_argument("--users", type=int, default=5, help="Concurrent users")
    parser.add_argument("--requests", type=int, default=20, help="Total requests")
    parser.add_argument(
        "--question",
        default="这个知识库主要是做什么的？",
        help="Question sent to /chat",
    )
    parser.add_argument("--collection", default="default", help="Collection name")
    parser.add_argument("--top-k", type=int, default=5, help="Top-k retrieval count")
    parser.add_argument(
        "--disable-rerank",
        action="store_true",
        help="Disable rerank to isolate baseline retrieval/generation latency",
    )
    parser.add_argument("--timeout", type=float, default=120.0, help="Per-request timeout in seconds")
    return parser


def main() -> None:
    args = _build_parser().parse_args()

    if args.users <= 0:
        raise SystemExit("--users must be > 0")
    if args.requests <= 0:
        raise SystemExit("--requests must be > 0")

    payload = {
        "question": args.question,
        "collection": args.collection,
        "top_k": args.top_k,
        "use_rerank": not args.disable_rerank,
    }

    results: List[RequestResult] = []
    print("Starting load test")
    print(f"Target URL: {args.url}")
    print(f"Concurrent users: {args.users}")
    print(f"Total requests: {args.requests}")
    print(f"Payload: {payload}")

    started_at = time.perf_counter()
    with ThreadPoolExecutor(max_workers=args.users) as executor:
        future_map = {
            executor.submit(_single_request, args.url, payload, request_id, args.timeout): request_id
            for request_id in range(1, args.requests + 1)
        }
        for future in as_completed(future_map):
            request_id = future_map[future]
            result = future.result()
            results.append(result)
            print(
                f"[req={request_id:03d}] status={result.status_code} "
                f"latency={result.latency_ms:.1f}ms ok={result.ok}"
            )
    total_s = time.perf_counter() - started_at

    latencies = [r.latency_ms for r in results]
    success = [r for r in results if r.ok]
    failures = [r for r in results if not r.ok]

    print("\nSummary")
    print(f"Completed: {len(results)}/{args.requests}")
    print(f"Success: {len(success)}")
    print(f"Failures: {len(failures)}")
    print(f"Success rate: {(len(success) / len(results) * 100):.1f}%")
    print(f"Total time: {total_s:.2f}s")
    print(f"Throughput: {len(results) / total_s:.2f} req/s")
    print(f"Avg latency: {statistics.mean(latencies):.1f}ms")
    print(f"P50 latency: {_percentile(latencies, 0.50):.1f}ms")
    print(f"P95 latency: {_percentile(latencies, 0.95):.1f}ms")
    print(f"Max latency: {max(latencies):.1f}ms")

    if failures:
        print("\nFailure samples")
        for item in failures[:5]:
            print(f"- status={item.status_code} latency={item.latency_ms:.1f}ms error={item.error}")


if __name__ == "__main__":
    main()
