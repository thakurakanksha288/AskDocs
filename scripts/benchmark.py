"""
Benchmark script: fire N concurrent /ask requests and report p50/p95 latency.

Usage:
    python scripts/benchmark.py --url http://localhost:8000 --n 50 --concurrency 10 \
        --question "What is the main topic of the document?"
"""
import argparse
import asyncio
import statistics
import time

import httpx


async def single_request(client: httpx.AsyncClient, url: str, payload: dict) -> float:
    t0 = time.perf_counter()
    resp = await client.post(f"{url}/ask", json=payload, timeout=60.0)
    elapsed = (time.perf_counter() - t0) * 1000
    if resp.status_code != 200:
        print(f"  ERROR {resp.status_code}: {resp.text[:100]}")
    return elapsed


async def run_benchmark(
    url: str, n: int, concurrency: int, question: str, doc_ids: list[str]
) -> None:
    payload: dict = {"question": question, "top_k": 5}
    if doc_ids:
        payload["document_ids"] = doc_ids

    semaphore = asyncio.Semaphore(concurrency)

    async def bounded_request(client):
        async with semaphore:
            return await single_request(client, url, payload)

    print(f"\nBenchmarking {url}/ask")
    print(f"  N={n}, concurrency={concurrency}, question='{question[:50]}'\n")

    async with httpx.AsyncClient() as client:
        tasks = [bounded_request(client) for _ in range(n)]
        latencies = await asyncio.gather(*tasks)

    latencies_sorted = sorted(latencies)
    p50 = statistics.median(latencies_sorted)
    p95 = latencies_sorted[min(int(len(latencies_sorted) * 0.95), len(latencies_sorted) - 1)]
    p99 = latencies_sorted[min(int(len(latencies_sorted) * 0.99), len(latencies_sorted) - 1)]

    print("── Results ───────────────────────────────")
    print(f"  Requests   : {n}")
    print(f"  Concurrency: {concurrency}")
    print(f"  p50 latency: {p50:.1f} ms")
    print(f"  p95 latency: {p95:.1f} ms")
    print(f"  p99 latency: {p99:.1f} ms")
    print(f"  Min        : {min(latencies):.1f} ms")
    print(f"  Max        : {max(latencies):.1f} ms")
    print("──────────────────────────────────────────\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="AskDocs benchmark")
    parser.add_argument("--url", default="http://localhost:8000")
    parser.add_argument("--n", type=int, default=50)
    parser.add_argument("--concurrency", type=int, default=10)
    parser.add_argument("--question", default="What is the main topic of the document?")
    parser.add_argument("--doc-ids", nargs="*", default=[])
    args = parser.parse_args()
    asyncio.run(
        run_benchmark(args.url, args.n, args.concurrency, args.question, args.doc_ids)
    )
