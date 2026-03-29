"""Qdrant Cloud VectorStore implementation.

This module provides a production-ready implementation of BaseVectorStore
using Qdrant Cloud (https://qdrant.tech), a managed vector database service.
Data persists in Qdrant Cloud and survives Render container restarts.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Any, Dict, List, Optional

from src.libs.vector_store.base_vector_store import BaseVectorStore

if TYPE_CHECKING:
    from src.core.settings import Settings

logger = logging.getLogger(__name__)

# Lazy import – qdrant-client is installed alongside chromadb
try:
    from qdrant_client import QdrantClient
    from qdrant_client.http import models as qdrant_models
    from qdrant_client.http.exceptions import UnexpectedResponse
    QDRANT_AVAILABLE = True
except ImportError:
    QDRANT_AVAILABLE = False


class QdrantStore(BaseVectorStore):
    """Qdrant Cloud implementation of VectorStore.

    Design principles:
    - Production-ready: Cloud-native, survives container restarts.
    - Swappable: Same BaseVectorStore interface as ChromaStore.
    - Config-driven: All settings from settings.yaml / environment variables.
    - Idempotent: upsert with same ID overwrites.

    Required env vars (set in Render Dashboard):
        QDRANT_URL       – e.g. https://your-cluster.cloud.qdrant.io
        QDRANT_API_KEY   – your Qdrant API key
        QDRANT_COLLECTION_NAME – collection name (default: "default")
        QDRANT_VECTOR_DIM     – embedding dimension (e.g. 2560 for qwen3-embedding-4b)
    """

    # Default values – can be overridden via settings or kwargs
    DEFAULT_COLLECTION = "default"
    DEFAULT_VECTOR_DIM = 2560
    DISTANCE_MODEL = qdrant_models.Distance.COSINE if QDRANT_AVAILABLE else None

    def __init__(self, settings: Settings, **kwargs: Any) -> None:
        if not QDRANT_AVAILABLE:
            raise ImportError(
                "qdrant-client is required for QdrantStore. "
                "Install it with: pip install qdrant-client"
            )

        vs_cfg = getattr(settings, "vector_store", None)
        if vs_cfg is None:
            raise ValueError(
                "Missing required configuration: settings.vector_store. "
                "Ensure 'vector_store' section exists in settings.yaml"
            )

        # Resolve collection name (kwargs > settings > default)
        self.collection_name = kwargs.get(
            "collection_name",
            getattr(vs_cfg, "qdrant_collection_name", None)
            or getattr(vs_cfg, "collection_name", None)
            or self.DEFAULT_COLLECTION,
        )

        # Resolve vector dimension
        self.vector_dim = kwargs.get(
            "vector_dim",
            getattr(vs_cfg, "qdrant_vector_dim", None)
            or self.DEFAULT_VECTOR_DIM,
        )

        # Resolve Qdrant credentials (prefer env-override via settings)
        self.qdrant_url: str = kwargs.get(
            "qdrant_url",
            getattr(vs_cfg, "qdrant_url", None) or "",
        )
        self.qdrant_api_key: Optional[str] = kwargs.get(
            "qdrant_api_key",
            getattr(vs_cfg, "qdrant_api_key", None) or None,
        )

        if not self.qdrant_url:
            raise ValueError(
                "Qdrant URL is required. Set QDRANT_URL in environment or "
                "vector_store.qdrant_url in settings.yaml"
            )

        logger.info(
            f"Initializing QdrantStore: collection='{self.collection_name}', "
            f"url='{self.qdrant_url}', dim={self.vector_dim}"
        )

        # Initialize Qdrant client
        self._client = QdrantClient(
            url=self.qdrant_url,
            api_key=self.qdrant_api_key,
            timeout=30,
        )

        # Ensure collection exists (create if missing)
        self._ensure_collection()

        logger.info(
            f"QdrantStore initialized. "
            f"Points in collection: {self._client.get_collection(collection_name=self.collection_name).points_count}"
        )

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _ensure_collection(self) -> None:
        """Create the collection if it does not already exist."""
        collections = [c.name for c in self._client.get_collections().collections]
        if self.collection_name not in collections:
            logger.info(f"Creating Qdrant collection '{self.collection_name}'...")
            self._client.create_collection(
                collection_name=self.collection_name,
                vectors_config={
                    "": qdrant_models.VectorParams(
                        size=self.vector_dim,
                        distance=self.DISTANCE_MODEL,
                    )
                },
                # Enable payload index for metadata filtering
                optimizers_config=qdrant_models.OptimizersConfigDiff(
                    indexing_threshold=1000,
                ),
            )
            logger.info(f"Collection '{self.collection_name}' created.")
        else:
            logger.debug(f"Collection '{self.collection_name}' already exists.")

    def _sanitize_payload(self, metadata: Dict[str, Any]) -> Dict[str, Any]:
        """Remove None values and coerce types that Qdrant cannot store."""
        sanitized: Dict[str, Any] = {}
        for k, v in metadata.items():
            if v is None:
                continue
            if isinstance(v, (str, int, float, bool)):
                sanitized[k] = v
            elif isinstance(v, (list, tuple)):
                sanitized[k] = [str(x) for x in v]
            else:
                sanitized[k] = str(v)
        return sanitized

    # ------------------------------------------------------------------
    # BaseVectorStore interface
    # ------------------------------------------------------------------

    def upsert(
        self,
        records: List[Dict[str, Any]],
        trace: Optional[Any] = None,
        **kwargs: Any,
    ) -> None:
        self.validate_records(records)

        points = []
        for record in records:
            payload = self._sanitize_payload(record.get("metadata", {}))
            # Store the text in payload so we can retrieve it later
            if "text" in record:
                payload["text"] = record["text"]

            points.append(
                qdrant_models.PointStruct(
                    id=record["id"],
                    vector={"": record["vector"]},
                    payload=payload,
                )
            )

        self._client.upsert(
            collection_name=self.collection_name,
            points=points,
            wait=True,
        )
        logger.debug(f"Upserted {len(points)} points to Qdrant collection '{self.collection_name}'")

    def query(
        self,
        vector: List[float],
        top_k: int = 10,
        filters: Optional[Dict[str, Any]] = None,
        trace: Optional[Any] = None,
        **kwargs: Any,
    ) -> List[Dict[str, Any]]:
        self.validate_query_vector(vector, top_k)

        # Build Qdrant filter from metadata filters
        qdrant_filter: Optional[qdrant_models.Filter] = None
        if filters:
            must_clauses = []
            for key, value in filters.items():
                must_clauses.append(
                    qdrant_models.FieldCondition(
                        key=f"metadata.{key}",
                        match=qdrant_models.MatchValue(value=value),
                    )
                )
            qdrant_filter = qdrant_models.Filter(must=must_clauses)

        results = self._client.search(
            collection_name=self.collection_name,
            query_vector=("", vector),
            limit=top_k,
            query_filter=qdrant_filter,
            with_payload=True,
            with_vectors=False,
            score_threshold=0.0,
        )

        output: List[Dict[str, Any]] = []
        for result in results:
            output.append({
                "id": str(result.id),
                "score": float(result.score),
                "text": result.payload.get("text", ""),
                "metadata": {
                    k: v for k, v in result.payload.items()
                    if k != "text"
                },
            })
        return output

    def delete(
        self,
        ids: List[str],
        trace: Optional[Any] = None,
        **kwargs: Any,
    ) -> None:
        if not ids:
            raise ValueError("IDs list cannot be empty")
        self._client.delete(
            collection_name=self.collection_name,
            points_selector=qdrant_models.PointIdsList(points=ids),
            wait=True,
        )
        logger.debug(f"Deleted {len(ids)} points from Qdrant collection")

    def clear(
        self,
        collection_name: Optional[str] = None,
        trace: Optional[Any] = None,
        **kwargs: Any,
    ) -> None:
        target = collection_name or self.collection_name
        self._client.delete_collection(collection_name=target)
        logger.info(f"Deleted collection '{target}'")
        # Recreate if it was the active collection
        if target == self.collection_name:
            self._ensure_collection()

    def delete_by_metadata(
        self,
        filter_dict: Dict[str, Any],
        trace: Optional[Any] = None,
    ) -> int:
        if not filter_dict:
            raise ValueError("filter_dict cannot be empty")
        must_clauses = [
            qdrant_models.FieldCondition(
                key=f"metadata.{k}",
                match=qdrant_models.MatchValue(v),
            )
            for k, v in filter_dict.items()
        ]
        result = self._client.delete(
            collection_name=self.collection_name,
            points_selector=qdrant_models.FilterSelector(
                filter=qdrant_models.Filter(must=must_clauses)
            ),
            wait=True,
        )
        return result

    def get_by_ids(
        self,
        ids: List[str],
        trace: Optional[Any] = None,
        **kwargs: Any,
    ) -> List[Dict[str, Any]]:
        if not ids:
            raise ValueError("IDs list cannot be empty")
        results = self._client.retrieve(
            collection_name=self.collection_name,
            ids=ids,
            with_payload=True,
        )
        id_to_result = {r.id: r for r in results}
        output: List[Dict[str, Any]] = []
        for rid in ids:
            rec = id_to_result.get(rid)
            if rec:
                output.append({
                    "id": str(rec.id),
                    "text": rec.payload.get("text", ""),
                    "metadata": {
                        k: v for k, v in rec.payload.items()
                        if k != "text"
                    },
                })
            else:
                output.append({})
        return output

    def get_collection_stats(self) -> Dict[str, Any]:
        info = self._client.get_collection(collection_name=self.collection_name)
        return {
            "count": info.points_count,
            "name": self.collection_name,
            "vector_dim": info.config.params.vectors[""].size,
        }
