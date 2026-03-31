"""Custom evaluator implementation for lightweight metrics.

This evaluator computes simple, deterministic metrics such as hit rate and MRR.
It is designed for fast regression checks and sanity validation.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence

from src.libs.evaluator.base_evaluator import BaseEvaluator


class CustomEvaluator(BaseEvaluator):
    """Custom evaluator for lightweight metrics (hit_rate, mrr).

    The evaluator expects retrieved chunks to contain an identifier field.
    Supported id fields: id, chunk_id, document_id, doc_id.
    """

    SUPPORTED_METRICS = {"hit_rate", "mrr"}
    _ID_FIELDS = ("id", "chunk_id", "document_id", "doc_id")

    def __init__(
        self,
        settings: Any = None,
        metrics: Optional[Sequence[str]] = None,
        **kwargs: Any,
    ) -> None:
        self.settings = settings
        self.kwargs = kwargs

        if metrics is None:
            metrics = self._metrics_from_settings(settings)

        normalized = [str(metric).strip().lower() for metric in (metrics or [])]
        if not normalized:
            normalized = ["hit_rate", "mrr"]

        unsupported = [metric for metric in normalized if metric not in self.SUPPORTED_METRICS]
        if unsupported:
            raise ValueError(
                "Unsupported custom metrics: "
                f"{', '.join(unsupported)}. Supported: {', '.join(sorted(self.SUPPORTED_METRICS))}"
            )

        self.metrics = normalized

    def evaluate(
        self,
        query: str,
        retrieved_chunks: List[Any],
        generated_answer: Optional[str] = None,
        ground_truth: Optional[Any] = None,
        trace: Optional[Any] = None,
        **kwargs: Any,
    ) -> Dict[str, float]:
        """Compute requested metrics for the given retrieval results.

        Args:
            query: The user query string.
            retrieved_chunks: Retrieved chunks or records.
            generated_answer: Optional generated answer (unused).
            ground_truth: Ground truth ids or structure.
            trace: Optional TraceContext (unused).
            **kwargs: Additional parameters (unused).

        Returns:
            Dictionary of metric name to float value.
        """
        self.validate_query(query)
        if not isinstance(retrieved_chunks, list):
            raise ValueError("retrieved_chunks must be a list")

        ground_truth_targets = self._extract_ground_truth_targets(ground_truth)
        ground_truth_ids = list(ground_truth_targets.get("ids") or [])
        ground_truth_sources = list(ground_truth_targets.get("sources") or [])

        if ground_truth_ids:
            retrieved_items = self._extract_ids(retrieved_chunks, label="retrieved_chunks")
            expected_items = ground_truth_ids
        elif ground_truth_sources:
            retrieved_items = self._extract_sources(retrieved_chunks, label="retrieved_chunks")
            expected_items = ground_truth_sources
        else:
            retrieved_items = self._extract_ids(retrieved_chunks, label="retrieved_chunks")
            expected_items = []

        results: Dict[str, float] = {}

        if "hit_rate" in self.metrics:
            results["hit_rate"] = self._compute_hit_rate(retrieved_items, expected_items)
        if "mrr" in self.metrics:
            results["mrr"] = self._compute_mrr(retrieved_items, expected_items)

        return results

    def _metrics_from_settings(self, settings: Any) -> List[str]:
        """Extract metrics list from settings if available."""
        if settings is None:
            return []
        metrics = getattr(getattr(settings, "evaluation", None), "metrics", None)
        if metrics is None:
            return []
        return [str(metric) for metric in metrics]

    def _extract_ground_truth_targets(self, ground_truth: Optional[Any]) -> Dict[str, List[str]]:
        """Extract ground truth targets from various input shapes."""
        if ground_truth is None:
            return {"ids": [], "sources": []}
        if isinstance(ground_truth, str):
            return {"ids": [ground_truth], "sources": []}
        if isinstance(ground_truth, dict):
            ids: List[str] = []
            sources: List[str] = []
            if "ids" in ground_truth and isinstance(ground_truth["ids"], list):
                ids = self._extract_ids(ground_truth["ids"], label="ground_truth.ids")
            if "sources" in ground_truth and isinstance(ground_truth["sources"], list):
                sources = self._normalise_sources(ground_truth["sources"])
            if ids or sources:
                return {"ids": ids, "sources": sources}
            return {"ids": self._extract_ids([ground_truth], label="ground_truth"), "sources": []}
        if isinstance(ground_truth, list):
            return {"ids": self._extract_ids(ground_truth, label="ground_truth"), "sources": []}

        raise ValueError(
            f"Unsupported ground_truth type: {type(ground_truth).__name__}. "
            "Expected str, dict, list, or None."
        )

    def _extract_sources(self, items: Iterable[Any], label: str) -> List[str]:
        """Extract normalized source file names from retrieved chunks."""
        sources: List[str] = []
        for index, item in enumerate(items):
            source_path: Optional[str] = None
            if isinstance(item, dict):
                metadata = item.get("metadata", {}) if isinstance(item.get("metadata"), dict) else {}
                source_path = metadata.get("source_path") or item.get("source_path")
            else:
                metadata = getattr(item, "metadata", None)
                if isinstance(metadata, dict):
                    source_path = metadata.get("source_path")
                if source_path is None and hasattr(item, "source_path"):
                    source_path = str(getattr(item, "source_path"))

            normalized = self._normalise_single_source(source_path)
            if normalized:
                sources.append(normalized)
                continue
            raise ValueError(
                f"Unable to extract source from {label}[{index}] of type "
                f"{type(item).__name__}"
            )
        return sources

    def _normalise_sources(self, values: Iterable[Any]) -> List[str]:
        out: List[str] = []
        for value in values:
            normalized = self._normalise_single_source(value)
            if normalized:
                out.append(normalized)
        return out

    @staticmethod
    def _normalise_single_source(value: Any) -> str:
        if value is None:
            return ""
        raw = str(value).strip()
        if not raw:
            return ""
        return Path(raw.replace("\\", "/")).name.lower()

    def _extract_ids(self, items: Iterable[Any], label: str) -> List[str]:
        """Extract ids from a list of items."""
        ids: List[str] = []
        for index, item in enumerate(items):
            if isinstance(item, str):
                ids.append(item)
                continue
            if isinstance(item, dict):
                for field in self._ID_FIELDS:
                    if field in item:
                        ids.append(str(item[field]))
                        break
                else:
                    raise ValueError(
                        f"Missing id field in {label}[{index}]. "
                        f"Expected one of {', '.join(self._ID_FIELDS)}"
                    )
                continue
            if hasattr(item, "chunk_id"):
                ids.append(str(getattr(item, "chunk_id")))
                continue
            if hasattr(item, "doc_id"):
                ids.append(str(getattr(item, "doc_id")))
                continue
            if hasattr(item, "document_id"):
                ids.append(str(getattr(item, "document_id")))
                continue
            if hasattr(item, "id"):
                ids.append(str(getattr(item, "id")))
                continue

            raise ValueError(
                f"Unable to extract id from {label}[{index}] of type "
                f"{type(item).__name__}"
            )

        return ids

    def _compute_hit_rate(self, retrieved_ids: Sequence[str], ground_truth_ids: Sequence[str]) -> float:
        """Compute hit rate (binary)."""
        if not ground_truth_ids:
            return 0.0
        return 1.0 if any(item in ground_truth_ids for item in retrieved_ids) else 0.0

    def _compute_mrr(self, retrieved_ids: Sequence[str], ground_truth_ids: Sequence[str]) -> float:
        """Compute Mean Reciprocal Rank (MRR)."""
        if not ground_truth_ids:
            return 0.0
        for rank, item in enumerate(retrieved_ids, start=1):
            if item in ground_truth_ids:
                return 1.0 / rank
        return 0.0
