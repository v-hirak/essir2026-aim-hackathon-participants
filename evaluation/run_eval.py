#!/usr/bin/env python3
"""Evaluate the three Level-3 questions against evaluation/gold.json."""

from __future__ import annotations

import argparse
import json
import re
import subprocess
import time
from pathlib import Path
from statistics import mean
from urllib.request import Request, urlopen

from bert_score import BERTScorer


ROOT = Path(__file__).resolve().parent.parent
GOLD_FILE = ROOT / "evaluation" / "gold.json"
RESULTS_DIR = ROOT / "results" / "evaluation"
API_URL = "http://localhost:8791"
NUMBER_RE = re.compile(r"(?<![\w.])\d+(?:\.\d+)?")


def api_json(path: str, payload: dict | None = None, timeout: float = 180) -> dict:
    data = json.dumps(payload).encode() if payload is not None else None
    headers = {"content-type": "application/json"} if data else {}
    request = Request(API_URL + path, data=data, headers=headers)
    with urlopen(request, timeout=timeout) as response:
        return json.loads(response.read())


def restart_app() -> None:
    """Clear Level-3 conversation memory while keeping the Qdrant index."""
    subprocess.run(["docker", "compose", "restart", "app"], check=True)
    for _ in range(90):
        try:
            if api_json("/health/ready", timeout=3).get("status") == "ready":
                return
        except Exception:
            pass
        time.sleep(1)
    raise RuntimeError("App did not become ready after restart")


def unique_numbers(text: str) -> list[str]:
    return list(dict.fromkeys(NUMBER_RE.findall(text)))


def score(question: dict, response: dict, bert_scorer: BERTScorer) -> dict:
    gold_pages = sorted({int(item["page"]) for item in question["gold_evidence"]})
    retrieved_pages = sorted({int(item["page"]) for item in response.get("sources", [])})
    page_hits = sorted(set(gold_pages) & set(retrieved_pages))

    gold_numbers = unique_numbers(question["gold_answer"])
    answer_numbers = unique_numbers(response.get("answer", ""))
    number_hits = sorted(set(gold_numbers) & set(answer_numbers))
    _, _, bert_f1 = bert_scorer.score([response.get("answer", "")], [question["gold_answer"]])

    return {
        "gold_pages": gold_pages,
        "retrieved_pages": retrieved_pages,
        "page_hits": page_hits,
        "gold_page_recall": round(len(page_hits) / len(gold_pages), 3),
        "gold_numbers": gold_numbers,
        "number_hits": number_hits,
        "gold_number_recall": (
            round(len(number_hits) / len(gold_numbers), 3) if gold_numbers else None
        ),
        "bertscore_f1": round(float(bert_f1[0]), 3),
        "latency_ms": (response.get("diagnostics") or {}).get("latency_ms"),
    }


def average(results: list[dict], metric: str) -> float | None:
    values = [item["metrics"][metric] for item in results if item.get("status") == "ok"]
    values = [value for value in values if value is not None]
    return round(mean(values), 3) if values else None


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-name", default="baseline-level3")
    args = parser.parse_args()

    gold = json.loads(GOLD_FILE.read_text())
    questions = [item for item in gold["questions"] if item["level"] == 3]
    bert_scorer = BERTScorer(model_type="roberta-base", lang="en")
    results: list[dict] = []
    output = RESULTS_DIR / f"{args.run_name}.json"

    for question in questions:
        print(f"Running {question['id']} ...", flush=True)
        try:
            restart_app()
            response = api_json(
                "/query",
                {"question": question["question"], "level": 3},
            )
            result = {
                "id": question["id"],
                "question": question["question"],
                "status": "ok",
                "gold": {
                    "answer": question["gold_answer"],
                    "evidence": question["gold_evidence"],
                },
                "metrics": score(question, response, bert_scorer),
                "response": response,
            }
        except Exception as error:
            result = {"id": question["id"], "status": "error", "error": str(error)}
        results.append(result)

        report = {
            "run_name": args.run_name,
            "document": gold["document"],
            "results": results,
            "summary": {
                "mean_gold_page_recall": average(results, "gold_page_recall"),
                "mean_gold_number_recall": average(results, "gold_number_recall"),
                "mean_bertscore_f1": average(results, "bertscore_f1"),
                "mean_latency_ms": average(results, "latency_ms"),
            },
        }
        RESULTS_DIR.mkdir(parents=True, exist_ok=True)
        output.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n")

    print("\nQuestion  Page recall  BERTScore F1  Number recall  Latency")
    for item in results:
        if item["status"] == "error":
            print(f"{item['id']:<9} ERROR: {item['error']}")
            continue
        metrics = item["metrics"]
        print(
            f"{item['id']:<9} {metrics['gold_page_recall']!s:<12} "
            f"{metrics['bertscore_f1']!s:<13} "
            f"{str(metrics['gold_number_recall']):<14} {metrics['latency_ms']} ms"
        )

    print("\nMean")
    print(f"Page recall:   {report['summary']['mean_gold_page_recall']}")
    print(f"BERTScore F1:  {report['summary']['mean_bertscore_f1']}")
    print(f"Number recall: {report['summary']['mean_gold_number_recall']}")
    print(f"Latency:       {report['summary']['mean_latency_ms']} ms")
    print(f"Saved: {output}")


if __name__ == "__main__":
    main()
