"""Retriever agent for lesson workflow."""

from __future__ import annotations

from typing import Any, Callable, Dict, List

from src.observability.logger import get_logger

from .agent_protocol import AgentMessage

logger = get_logger(__name__)


class RetrieverAgent:
    """Retrieve and filter evidence for lesson generation."""

    def __init__(
        self,
        *,
        hybrid_search: Any,
        reranker: Any,
        trace: Any,
        top_k: int,
        prioritize_visual_results: Callable[[List[Any], Any], List[Any]],
        relevance_check: Callable[[str, Any], bool],
        extract_image_resources: Callable[..., List[Any]],
        sanitize_source_path: Callable[[Any], str],
        image_storage: Any,
        collection: str,
        enable_rerank: bool = True,
        enable_image_extraction: bool = True,
        max_search_queries: int | None = None,
    ) -> None:
        self.hybrid_search = hybrid_search
        self.reranker = reranker
        self.trace = trace
        self.top_k = top_k
        self.prioritize_visual_results = prioritize_visual_results
        self.relevance_check = relevance_check
        self.extract_image_resources = extract_image_resources
        self.sanitize_source_path = sanitize_source_path
        self.image_storage = image_storage
        self.collection = collection
        self.enable_rerank = enable_rerank
        self.enable_image_extraction = enable_image_extraction
        self.max_search_queries = max_search_queries

    def run(self, message: AgentMessage) -> AgentMessage:
        query_plan = message.artifacts.get("query_plan") or {}
        topic = str(message.context.get("topic") or "")

        search_queries = list(query_plan.get("search_queries") or [])
        if not search_queries:
            search_queries = [str(query_plan.get("user_query") or topic)]
        if self.max_search_queries is not None:
            search_queries = search_queries[: max(1, self.max_search_queries)]

        per_query_top_k = max(6, min(self.top_k, (self.top_k // max(1, len(search_queries))) + 3))
        merged_results: List[Any] = []
        seen_result_keys = set()

        for search_query in search_queries:
            hybrid_result = self.hybrid_search.search(
                query=search_query,
                top_k=per_query_top_k,
                filters=None,
                trace=self.trace,
            )
            current_results = hybrid_result if not hasattr(hybrid_result, "results") else hybrid_result.results
            for item in current_results:
                metadata = item.metadata or {}
                key = (
                    metadata.get("chunk_id")
                    or metadata.get("id")
                    or f"{metadata.get('doc_hash')}::{metadata.get('page_num')}::{(item.text or '')[:80]}"
                )
                if key in seen_result_keys:
                    continue
                seen_result_keys.add(key)
                merged_results.append(item)

        merged_results = merged_results[: self.top_k]
        if merged_results and self.enable_rerank and self.reranker.is_enabled:
            rerank_result = self.reranker.rerank(
                query=str(query_plan.get("user_query") or topic),
                results=merged_results,
                top_k=self.top_k,
                trace=self.trace,
            )
            merged_results = rerank_result.results

        raw_results = self.prioritize_visual_results(merged_results, query_plan)[: self.top_k]
        relevant_results = [result for result in raw_results if self.relevance_check(topic, result)]

        citations: List[Dict[str, Any]] = []
        for result in relevant_results:
            citations.append(
                {
                    "source": self.sanitize_source_path((result.metadata or {}).get("source_path", "unknown")),
                    "score": round(float(getattr(result, "score", 0.0) or 0.0), 4),
                    "text": str(result.text or "")[:200],
                }
            )

        image_resources = []
        if self.enable_image_extraction:
            image_resources = self.extract_image_resources(
                relevant_results,
                image_storage=self.image_storage,
                collection=self.collection,
            )
        logger.info(
            "lesson_retriever.images topic=%s enabled=%s relevant_results=%s image_resources=%s collection=%s",
            topic,
            self.enable_image_extraction,
            len(relevant_results),
            len(image_resources),
            self.collection,
        )

        message.artifacts["raw_results"] = raw_results
        message.artifacts["relevant_results"] = relevant_results
        message.artifacts["citations"] = citations
        message.artifacts["image_resources"] = image_resources
        message.next_action = "write_review"
        return message
