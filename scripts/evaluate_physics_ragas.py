#!/usr/bin/env python
"""Run a more realistic Ragas evaluation on the physics textbook collection."""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import replace
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.core.settings import EvaluationSettings, load_settings
from src.libs.llm.base_llm import Message
from src.libs.llm.llm_factory import LLMFactory
from src.observability.dashboard.pages.evaluation_panel import _try_create_hybrid_search
from src.observability.evaluation.eval_runner import EvalRunner
from src.observability.evaluation.ragas_evaluator import RagasEvaluator


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate physics textbook QA quality with Ragas.",
    )
    parser.add_argument(
        "--collection",
        default="physics_fast20",
        help="Target Chroma/BM25 collection (default: physics_fast20).",
    )
    parser.add_argument(
        "--test-set",
        default="tests/fixtures/physics_golden_test_set.json",
        help="Path to the physics golden test set JSON file.",
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=5,
        help="Number of chunks to retrieve per query (default: 5).",
    )
    parser.add_argument(
        "--json",
        action="store_true",
        help="Output raw JSON instead of a formatted summary.",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=0,
        help="Only evaluate the first N test cases (default: 0 = all).",
    )
    return parser.parse_args()


def build_answer_generator(settings):
    llm = LLMFactory.create(settings)

    def _extract_text(chunk) -> str:
        if isinstance(chunk, dict):
            return str(chunk.get("text", ""))
        return str(getattr(chunk, "text", ""))

    def answer_generator(query: str, chunks: list) -> str:
        context = "\n\n".join(_extract_text(chunk)[:700] for chunk in chunks[:3])
        messages = [
            Message(
                role="system",
                content=(
                    "你是高中物理助教。只能依据提供的检索上下文回答，"
                    "不能补充上下文之外的常识。请用中文简洁回答，"
                    "控制在80字内；如果上下文不足，就明确说信息不足。"
                ),
            ),
            Message(
                role="user",
                content=(
                    f"问题：{query}\n\n"
                    f"检索上下文：\n{context}\n\n"
                    "请直接作答。"
                ),
            ),
        ]
        response = llm.chat(messages, temperature=0.2, max_tokens=200)
        return response.content.strip()

    return answer_generator


def print_summary(report_dict: dict) -> None:
    print("=" * 60)
    print("  PHYSICS RAGAS EVALUATION")
    print("=" * 60)
    print(f"Queries: {report_dict.get('query_count', 0)}")
    print(f"Time:    {report_dict.get('total_elapsed_ms', 0):.1f} ms")
    print()
    print("Aggregate metrics:")
    for key, value in sorted(report_dict.get("aggregate_metrics", {}).items()):
        print(f"  - {key}: {value:.4f}")
    print()
    print("Per-query metrics:")
    for item in report_dict.get("query_results", []):
        metrics = ", ".join(
            f"{name}={value:.4f}" for name, value in sorted(item.get("metrics", {}).items())
        )
        print(f"  - {item['query']}: {metrics} ({item['elapsed_ms']:.1f} ms)")


def main() -> int:
    args = parse_args()

    test_set_path = Path(args.test_set)
    if args.limit and args.limit > 0:
        with test_set_path.open("r", encoding="utf-8") as f:
            payload = json.load(f)
        payload["test_cases"] = payload.get("test_cases", [])[: args.limit]
        limited_path = test_set_path.parent / f"{test_set_path.stem}.limit{args.limit}.json"
        with limited_path.open("w", encoding="utf-8") as f:
            json.dump(payload, f, ensure_ascii=False, indent=2)
        test_set_path = limited_path

    settings = load_settings()
    settings = replace(
        settings,
        vector_store=replace(
            settings.vector_store,
            provider="chroma",
            collection_name=args.collection,
        ),
        evaluation=EvaluationSettings(
            enabled=True,
            provider="ragas",
            metrics=["faithfulness", "answer_relevancy", "context_precision"],
        ),
    )

    hybrid_search = _try_create_hybrid_search(settings, args.collection)
    evaluator = RagasEvaluator(
        settings=settings,
        metrics=["faithfulness", "answer_relevancy", "context_precision"],
    )
    answer_generator = build_answer_generator(settings)

    runner = EvalRunner(
        settings=settings,
        hybrid_search=hybrid_search,
        evaluator=evaluator,
        answer_generator=answer_generator,
    )
    report = runner.run(str(test_set_path), top_k=args.top_k, collection=args.collection)
    report_dict = report.to_dict()

    if args.json:
        print(json.dumps(report_dict, ensure_ascii=False))
    else:
        print_summary(report_dict)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
