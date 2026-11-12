"""
Retrieval evaluation script.

Q&A file format (eval_data.json):
[
  {
    "question": "What is the capital of France?",
    "expected_substring": "Paris",
    "document_id": "<uuid>"
  },
  ...
]

Usage:
    python scripts/eval_retrieval.py --url http://localhost:8000 \
        --qa-file scripts/eval_data.json
"""
import argparse
import asyncio
import json

import httpx


async def evaluate(url: str, qa_file: str, top_k: int = 3) -> None:
    with open(qa_file) as f:
        qa_pairs: list[dict] = json.load(f)

    hits = 0
    total = len(qa_pairs)

    async with httpx.AsyncClient() as client:
        for i, qa in enumerate(qa_pairs, 1):
            question = qa["question"]
            expected = qa["expected_substring"].lower()
            doc_ids = [qa["document_id"]] if "document_id" in qa else None

            payload: dict = {"question": question, "top_k": top_k}
            if doc_ids:
                payload["document_ids"] = doc_ids

            resp = await client.post(f"{url}/ask", json=payload, timeout=30.0)
            if resp.status_code != 200:
                print(f"[{i:02d}] ERROR {resp.status_code}: {question[:50]}")
                continue

            data = resp.json()
            sources_text = " ".join(s["text"].lower() for s in data.get("sources", []))
            hit = expected in sources_text
            hits += hit

            status = "v" if hit else "x"
            print(f"[{i:02d}] {status}  {question[:60]}")
            if not hit:
                print(f"       expected: '{expected[:60]}'")

    hit_rate = hits / total * 100 if total else 0
    print(f"\n── Retrieval Evaluation ──────────────────")
    print(f"  Questions : {total}")
    print(f"  Hits@{top_k}   : {hits}")
    print(f"  Hit rate  : {hit_rate:.1f}%")
    print(f"──────────────────────────────────────────")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="AskDocs retrieval evaluator")
    parser.add_argument("--url", default="http://localhost:8000")
    parser.add_argument("--qa-file", default="scripts/eval_data.json")
    parser.add_argument("--top-k", type=int, default=3)
    args = parser.parse_args()
    asyncio.run(evaluate(args.url, args.qa_file, args.top_k))
