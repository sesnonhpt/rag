"""App runtime wiring: component bootstrap, lifespan, and global handlers."""

from __future__ import annotations

import os
from contextlib import asynccontextmanager
from typing import Any

from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse

from app.core.paths import PROJECT_ROOT
from app.core.runtime_helpers import build_api_error_detail
from src.agents import LessonHistoryStorage
from src.core.query_engine.dense_retriever import create_dense_retriever
from src.core.query_engine.hybrid_search import create_hybrid_search
from src.core.query_engine.query_processor import QueryProcessor
from src.core.query_engine.reranker import create_core_reranker
from src.core.query_engine.sparse_retriever import create_sparse_retriever
from src.core.settings import load_settings
from src.ingestion.storage.bm25_indexer import BM25Indexer
from src.ingestion.storage.image_storage import ImageStorage
from src.libs.embedding.embedding_factory import EmbeddingFactory
from src.libs.llm.llm_factory import LLMFactory
from src.libs.vector_store.vector_store_factory import VectorStoreFactory
from src.observability.logger import get_logger

logger = get_logger(__name__)
_ROOT = PROJECT_ROOT


def build_components(settings: Any, collection: str) -> tuple:
    vector_store = VectorStoreFactory.create(settings, collection_name=collection)
    embedding_client = EmbeddingFactory.create(settings)
    dense_retriever = create_dense_retriever(
        settings=settings,
        embedding_client=embedding_client,
        vector_store=vector_store,
    )
    bm25_indexer = BM25Indexer(index_dir=str(_ROOT / "data" / "db" / "bm25" / collection))
    sparse_retriever = create_sparse_retriever(
        settings=settings,
        bm25_indexer=bm25_indexer,
        vector_store=vector_store,
    )
    sparse_retriever.default_collection = collection
    query_processor = QueryProcessor()
    hybrid_search = create_hybrid_search(
        settings=settings,
        query_processor=query_processor,
        dense_retriever=dense_retriever,
        sparse_retriever=sparse_retriever,
    )
    reranker = create_core_reranker(settings=settings)
    return hybrid_search, reranker


def is_lesson_plan_mock_enabled() -> bool:
    # Mock lesson-plan responses are guarded by an explicit feature flag.
    # Keep the default disabled so only environments that opt in (for example
    # online debugging / controlled production rollout) can use this path.
    return os.environ.get("LESSON_PLAN_MOCK_ENABLED", "0").strip().lower() not in {"0", "false", "off", "no"}


@asynccontextmanager
async def lifespan(app: FastAPI):
    config_path = os.environ.get("CHAT_CONFIG", str(_ROOT / "config" / "settings.yaml"))
    logger.info("Loading settings from: %s", config_path)
    settings = load_settings(config_path)

    collection = settings.vector_store.collection_name
    hybrid_search, reranker = build_components(settings, collection)
    llm = LLMFactory.create(settings)
    image_storage = ImageStorage(
        db_path=str(_ROOT / "data" / "db" / "image_index.db"),
        images_root=str(_ROOT / "data" / "images"),
    )
    history_storage = LessonHistoryStorage(
        db_path=str(_ROOT / "data" / "db" / "lesson_history.db"),
    )

    app.state.settings = settings
    app.state.hybrid_search = hybrid_search
    app.state.reranker = reranker
    app.state.llm = llm
    app.state.image_storage = image_storage
    app.state.history_storage = history_storage
    app.state.default_collection = collection
    app.state.lesson_plan_mock_enabled = is_lesson_plan_mock_enabled()

    logger.info("Chat API components initialised successfully")
    yield
    logger.info("Chat API shutting down")
    image_storage.close()


async def global_exception_handler(request: Request, exc: Exception):
    logger.exception("Unhandled error on %s: %s", request.url, exc)
    return JSONResponse(
        status_code=500,
        content={
            "error": "服务器内部错误，请稍后重试",
            "detail": build_api_error_detail(
                code="INTERNAL_SERVER_ERROR",
                message="服务器内部错误，请稍后重试",
                stage="global",
            ),
        },
    )
