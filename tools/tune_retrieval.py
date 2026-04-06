from __future__ import annotations

import argparse
import copy
import json
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from app.config import Settings
from preprocessing.extract import Preprocessor
from tools.search import ManualSearchEngine


@dataclass
class EvalCase:
    case_id: str
    query: str
    acceptable_pages: list[dict[str, Any]]
    query_type: str = "general"
    preferred_top1: dict[str, Any] | None = None
    expected_structured_keys: list[str] | None = None
    needs_visual_page: bool = False
    notes: str = ""


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Tune hybrid retrieval weights against a local eval set.")
    parser.add_argument(
        "--cases",
        default=str(Path(__file__).with_name("retrieval_tuning_cases.json")),
        help="Path to eval cases JSON.",
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=4,
        help="Top-K page hits to evaluate.",
    )
    parser.add_argument(
        "--candidate-limit",
        type=int,
        default=12,
        help="Retrieval candidate pool size.",
    )
    parser.add_argument(
        "--sparse-grid",
        default="0.35,0.45,0.55,0.65",
        help="Comma-separated sparse weights to evaluate.",
    )
    parser.add_argument(
        "--semantic-grid",
        default="0.65,0.55,0.45,0.35",
        help="Comma-separated semantic weights to evaluate.",
    )
    parser.add_argument(
        "--normalize-weights",
        action="store_true",
        help="Normalize sparse/semantic pairs to sum to 1.0 before evaluation.",
    )
    parser.add_argument(
        "--preprocess",
        action="store_true",
        help="Rebuild cached retrieval assets before tuning.",
    )
    parser.add_argument(
        "--show-details",
        action="store_true",
        help="Print top hits and excerpts for each query.",
    )
    parser.add_argument(
        "--output",
        default="",
        help="Optional JSON path for the evaluation report.",
    )
    return parser.parse_args()


def load_cases(path: str) -> list[EvalCase]:
    raw = json.loads(Path(path).read_text(encoding="utf-8"))
    return [
        EvalCase(
            case_id=item["id"],
            query=item["query"],
            acceptable_pages=item.get("acceptable_pages", []),
            query_type=item.get("query_type", "general"),
            preferred_top1=item.get("preferred_top1"),
            expected_structured_keys=item.get("expected_structured_keys", []),
            needs_visual_page=bool(item.get("needs_visual_page", False)),
            notes=item.get("notes", ""),
        )
        for item in raw
    ]


def clone_settings(base: Settings, sparse_weight: float, semantic_weight: float, candidate_limit: int) -> Settings:
    settings = copy.deepcopy(base)
    settings.sparse_weight = sparse_weight
    settings.semantic_weight = semantic_weight
    settings.retrieval_candidate_pool = max(candidate_limit, settings.max_search_results)
    return settings


def normalize_pair(sparse: float, semantic: float) -> tuple[float, float]:
    total = sparse + semantic
    if total <= 0:
        return 0.5, 0.5
    return sparse / total, semantic / total


def evaluate_case(engine: ManualSearchEngine, case: EvalCase, top_k: int) -> dict[str, Any]:
    retrieval = engine.search(case.query, limit=top_k)
    pages = [
        {"doc": page.doc, "page": page.page, "score": page.score, "section": page.section}
        for page in retrieval.pages[:top_k]
    ]

    rank = None
    acceptable = {(item["doc"], int(item["page"])) for item in case.acceptable_pages}
    for idx, page in enumerate(retrieval.pages[:top_k], start=1):
        if (page.doc, page.page) in acceptable:
            rank = idx
            break

    top1 = pages[0] if pages else None
    top1_hit = bool(top1 and (top1["doc"], top1["page"]) in acceptable)

    preferred_top1_hit = None
    if case.preferred_top1:
        preferred_top1_hit = bool(
            top1
            and top1["doc"] == case.preferred_top1["doc"]
            and int(top1["page"]) == int(case.preferred_top1["page"])
        )

    retrieved_structured = sorted(retrieval.structured_hits.keys())
    expected_structured = sorted(case.expected_structured_keys or [])
    structured_match = all(key in retrieved_structured for key in expected_structured)

    visual_targets = {(item["doc"], int(item["page"])) for item in case.acceptable_pages}
    visual_page_present = any((page["doc"], int(page["page"])) in visual_targets for page in pages) or any(
        excerpt.lower().startswith("visual reference") for excerpt in retrieval.excerpts
    )

    score = 0.0
    if rank is not None:
        score += 1.0 / rank
    if top1_hit:
        score += 0.75
    if preferred_top1_hit is True:
        score += 0.5
    if expected_structured and structured_match:
        score += 0.25
    if case.needs_visual_page and visual_page_present:
        score += 0.25

    failure_reasons: list[str] = []
    if rank is None:
        failure_reasons.append("acceptable_page_missing")
    if case.preferred_top1 and not preferred_top1_hit:
        failure_reasons.append("preferred_top1_missing")
    if expected_structured and not structured_match:
        failure_reasons.append("structured_key_missing")
    if case.needs_visual_page and not visual_page_present:
        failure_reasons.append("visual_page_missing")

    return {
        "case_id": case.case_id,
        "query": case.query,
        "query_type": case.query_type,
        "notes": case.notes,
        "hit": rank is not None,
        "rank": rank,
        "mrr": 0.0 if rank is None else 1.0 / rank,
        "top1_hit": top1_hit,
        "preferred_top1_hit": preferred_top1_hit,
        "structured_match": structured_match,
        "visual_page_present": visual_page_present,
        "score": score,
        "failure_reasons": failure_reasons,
        "pages": pages,
        "excerpts": retrieval.excerpts[: min(2, len(retrieval.excerpts))],
        "structured_keys": retrieved_structured,
    }


def evaluate_combo(
    base_settings: Settings,
    cases: list[EvalCase],
    sparse_weight: float,
    semantic_weight: float,
    top_k: int,
    candidate_limit: int,
) -> dict[str, Any]:
    settings = clone_settings(base_settings, sparse_weight, semantic_weight, candidate_limit)
    engine = ManualSearchEngine(settings)
    results = [evaluate_case(engine, case, top_k=top_k) for case in cases]
    hit_count = sum(1 for result in results if result["hit"])
    mrr = sum(result["mrr"] for result in results) / max(len(results), 1)
    top1_accuracy = sum(1 for result in results if result["top1_hit"]) / max(len(results), 1)
    preferred_top1_rate = sum(
        1 for result in results if result["preferred_top1_hit"] is not False
    ) / max(len(results), 1)
    structured_match_rate = sum(1 for result in results if result["structured_match"]) / max(len(results), 1)
    visual_match_rate = sum(
        1
        for result in results
        if (not any(case.case_id == result["case_id"] and case.needs_visual_page for case in cases))
        or result["visual_page_present"]
    ) / max(len(results), 1)
    avg_score = sum(result["score"] for result in results) / max(len(results), 1)
    return {
        "weights": {
            "sparse": round(sparse_weight, 4),
            "semantic": round(semantic_weight, 4),
        },
        "hit_rate": hit_count / max(len(results), 1),
        "top1_accuracy": top1_accuracy,
        "preferred_top1_rate": preferred_top1_rate,
        "structured_match_rate": structured_match_rate,
        "visual_match_rate": visual_match_rate,
        "hits": hit_count,
        "case_count": len(results),
        "mrr": mrr,
        "avg_score": avg_score,
        "results": results,
    }


def print_combo_summary(report: dict[str, Any], show_details: bool) -> None:
    weights = report["weights"]
    print(
        f"sparse={weights['sparse']:.2f} semantic={weights['semantic']:.2f} "
        f"hit_rate={report['hit_rate']:.2%} top1={report['top1_accuracy']:.2%} "
        f"mrr={report['mrr']:.3f} score={report['avg_score']:.3f}"
    )
    if not show_details:
        return
    for result in report["results"]:
        print(
            f"  - {result['case_id']}: hit={result['hit']} rank={result['rank']} "
            f"top1={result['top1_hit']} structured={result['structured_match']} "
            f"visual={result['visual_page_present']}"
        )
        if result["failure_reasons"]:
            print(f"      fail: {', '.join(result['failure_reasons'])}")
        for page in result["pages"]:
            print(f"      page {page['doc']} p.{page['page']} score={page['score']}")
        for excerpt in result["excerpts"]:
            print(f"      excerpt: {excerpt[:220]}")


def main() -> None:
    args = parse_args()
    base_settings = Settings()

    if args.preprocess:
        manifest = Preprocessor(base_settings).run()
        print(f"Preprocessed cache: {manifest}")

    cases = load_cases(args.cases)
    sparse_grid = [float(item) for item in args.sparse_grid.split(",") if item.strip()]
    semantic_grid = [float(item) for item in args.semantic_grid.split(",") if item.strip()]

    reports: list[dict[str, Any]] = []
    for sparse in sparse_grid:
        for semantic in semantic_grid:
            if args.normalize_weights:
                sparse, semantic = normalize_pair(sparse, semantic)
            report = evaluate_combo(
                base_settings=base_settings,
                cases=cases,
                sparse_weight=sparse,
                semantic_weight=semantic,
                top_k=args.top_k,
                candidate_limit=args.candidate_limit,
            )
            reports.append(report)
            print_combo_summary(report, show_details=args.show_details)

    best = max(
        reports,
        key=lambda item: (
            item["avg_score"],
            item["hit_rate"],
            item["top1_accuracy"],
            item["mrr"],
        ),
    )
    print("\nBest combo:")
    print_combo_summary(best, show_details=True)

    if args.output:
        payload = {
            "best": best,
            "reports": reports,
            "top_k": args.top_k,
            "candidate_limit": args.candidate_limit,
            "cases_file": args.cases,
        }
        output_path = Path(args.output)
        output_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        print(f"\nSaved report to {output_path}")


if __name__ == "__main__":
    main()
