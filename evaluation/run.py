import argparse
import json
import time
from pathlib import Path
from typing import Any, Literal

import chromadb

from app.config import Settings
from app.models import DocumentInput
from app.reranking import CrossEncoderReranker
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


def evaluate_hybrid_weights(
    dataset_path: Path, top_k: int, step: float = 0.1
) -> list[dict[str, float | int]]:
    dataset = load_dataset(dataset_path)
    settings = Settings(collection_name="weight_tuning")
    store = HybridDocumentStore(settings, client=chromadb.EphemeralClient())
    store.add([DocumentInput.model_validate(item) for item in dataset["documents"]])

    configurations: list[dict[str, float | int]] = []
    steps = round(1.0 / step)
    for index in range(steps + 1):
        semantic_weight = round(index * step, 6)
        keyword_weight = round(1.0 - semantic_weight, 6)
        store.settings.semantic_weight = semantic_weight
        store.settings.keyword_weight = keyword_weight
        measurements: list[QueryMeasurement] = []
        for query in dataset["queries"]:
            started = time.perf_counter()
            results = store.search(query["question"], top_k=top_k, mode="hybrid")
            elapsed_ms = (time.perf_counter() - started) * 1000
            measurements.append(
                measure_query(
                    query_id=query["id"],
                    results=results,
                    expected_sources=set(query["expected_sources"]),
                    latency_ms=elapsed_ms,
                )
            )
        configurations.append(
            {
                "semantic_weight": semantic_weight,
                "keyword_weight": keyword_weight,
                **summarize(measurements),
            }
        )
    return sorted(
        configurations,
        key=lambda item: (
            -float(item["recall_at_k"]),
            -float(item["mrr"]),
            float(item["mean_latency_ms"]),
        ),
    )


def evaluate_reranker(
    dataset_path: Path, top_k: int, candidate_k: int, model_name: str
) -> dict[str, float | int | str]:
    dataset = load_dataset(dataset_path)
    settings = Settings(collection_name="reranker_evaluation")
    store = HybridDocumentStore(settings, client=chromadb.EphemeralClient())
    store.add([DocumentInput.model_validate(item) for item in dataset["documents"]])
    reranker = CrossEncoderReranker(model_name)
    measurements: list[QueryMeasurement] = []
    for query in dataset["queries"]:
        started = time.perf_counter()
        candidates = store.search(
            query["question"], top_k=candidate_k, mode="hybrid"
        )
        results = reranker.rerank(query["question"], candidates, top_k)
        elapsed_ms = (time.perf_counter() - started) * 1000
        measurements.append(
            measure_query(
                query_id=query["id"],
                results=results,
                expected_sources=set(query["expected_sources"]),
                latency_ms=elapsed_ms,
            )
        )
    return {
        "model": model_name,
        "candidate_k": candidate_k,
        **summarize(measurements),
    }


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
    parser.add_argument(
        "--rerank",
        action="store_true",
        help="Benchmark cross-encoder reranking after hybrid retrieval",
    )
    parser.add_argument("--candidate-k", type=int, default=20)
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument("--json-output", type=Path)
    parser.add_argument(
        "--tune",
        action="store_true",
        help="Search semantic/keyword weights from 0.0 to 1.0",
    )
    args = parser.parse_args()
    if args.top_k < 1:
        parser.error("--top-k must be at least 1")

    report: dict[str, Any] = evaluate(args.dataset, args.top_k)
    print_report(report, args.top_k)
    if args.tune:
        tuning = evaluate_hybrid_weights(args.dataset, args.top_k)
        best = tuning[0]
        report["weight_tuning"] = {"best": best, "all": tuning}
        print(
            "\nBest hybrid weights: "
            f"semantic={best['semantic_weight']:.1f}, "
            f"keyword={best['keyword_weight']:.1f} "
            f"(Recall@K={best['recall_at_k']:.3f}, MRR={best['mrr']:.3f})"
        )
    if args.rerank:
        settings = Settings()
        reranked = evaluate_reranker(
            args.dataset,
            args.top_k,
            args.candidate_k,
            settings.reranker_model,
        )
        report["reranked"] = reranked
        print(
            "\nReranked hybrid: "
            f"Recall@K={reranked['recall_at_k']:.3f}, "
            f"MRR={reranked['mrr']:.3f}, "
            f"mean={reranked['mean_latency_ms']:.2f} ms"
        )
    if args.json_output:
        args.json_output.parent.mkdir(parents=True, exist_ok=True)
        args.json_output.write_text(json.dumps(report, indent=2), encoding="utf-8")
        print(f"\nSaved JSON report to {args.json_output}")


if __name__ == "__main__":
    main()
