import argparse
import json
from pathlib import Path


def check_report(path: Path, min_recall: float, min_mrr: float) -> dict[str, float]:
    report = json.loads(path.read_text(encoding="utf-8"))
    best = report.get("weight_tuning", {}).get("best") or report["hybrid"]
    recall = float(best["recall_at_k"])
    mrr = float(best["mrr"])
    if recall < min_recall or mrr < min_mrr:
        raise ValueError(
            f"Retrieval regression: Recall@K={recall:.3f} (min {min_recall:.3f}), "
            f"MRR={mrr:.3f} (min {min_mrr:.3f})"
        )
    return {"recall_at_k": recall, "mrr": mrr}


def main() -> None:
    parser = argparse.ArgumentParser(description="Validate a committed RAG benchmark")
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--min-recall", type=float, default=0.95)
    parser.add_argument("--min-mrr", type=float, default=0.90)
    args = parser.parse_args()
    result = check_report(args.report, args.min_recall, args.min_mrr)
    print(f"Retrieval quality gate passed: Recall@K={result['recall_at_k']:.3f}, MRR={result['mrr']:.3f}")


if __name__ == "__main__":
    main()
