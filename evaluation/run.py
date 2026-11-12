import argparse
import json
import time
from pathlib import Path
from typing import Any, Literal

import chromadb

from app.config import Settings
from app.models import DocumentInput
from app.store import HybridDocumentStore
from evaluation.metrics import QueryMeasurement, measure_query, summarize


MODES: tuple[Literal["semantic", "keyword", "hybrid"], ...] = (
    "semantic",
    "keyword",
    "hybrid",
)


def load_dataset(path: Path) -> dict[str, Any]:
    with path.open(encoding="utf-8") as stream:
        dataset = json.load(stream)
    if not dataset.get("documents") or not dataset.get("queries"):
        raise ValueError("Dataset must contain non-empty documents and queries arrays")
    return dataset


def evaluate(dataset_path: Path, top_k: int) -> dict[str, dict[str, float | int]]:
    dataset = load_dataset(dataset_path)
    settings = Settings(collection_name="evaluation")
    store = HybridDocumentStore(settings, client=chromadb.EphemeralClient())
    store.add([DocumentInput.model_validate(item) for item in dataset["documents"]])

    report: dict[str, dict[str, float | int]] = {}
    for mode in MODES:
        measurements: list[QueryMeasurement] = []
        for query in dataset["queries"]:
            started = time.perf_counter()
            results = store.search(query["question"], top_k=top_k, mode=mode)
            elapsed_ms = (time.perf_counter() - started) * 1000
            measurements.append(
                measure_query(
                    query_id=query["id"],
                    results=results,
                    expected_sources=set(query["expected_sources"]),
                    latency_ms=elapsed_ms,
                )
            )
        report[mode] = summarize(measurements)
    return report


def print_report(report: dict[str, dict[str, float | int]], top_k: int) -> None:
    print(f"Retrieval benchmark (K={top_k})")
    print("mode       recall@k   MRR       mean ms   p95 ms")
    print("---------- ---------- --------- --------- ---------")
    for mode in MODES:
        row = report[mode]
        print(
            f"{mode:<10} {row['recall_at_k']:<10.3f} {row['mrr']:<9.3f} "
            f"{row['mean_latency_ms']:<9.2f} {row['p95_latency_ms']:<9.2f}"
        )


def main() -> None:
    parser = argparse.ArgumentParser(description="Compare RAG retrieval strategies")
    parser.add_argument(
        "--dataset",
        type=Path,
        default=Path("evaluation/sample_dataset.json"),
        help="Evaluation dataset JSON path",
    )
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument("--json-output", type=Path)
    args = parser.parse_args()
    if args.top_k < 1:
        parser.error("--top-k must be at least 1")

    report = evaluate(args.dataset, args.top_k)
    print_report(report, args.top_k)
    if args.json_output:
        args.json_output.parent.mkdir(parents=True, exist_ok=True)
        args.json_output.write_text(json.dumps(report, indent=2), encoding="utf-8")
        print(f"\nSaved JSON report to {args.json_output}")


if __name__ == "__main__":
    main()
