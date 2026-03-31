"""FastAPI API for lesson-plan generation and optional RAG chat endpoints."""

from __future__ import annotations

import asyncio
import json
import os
import sys
import time
import uuid
from contextlib import asynccontextmanager
from pathlib import Path
from dataclasses import asdict, is_dataclass
import re
from typing import Any, Callable, Dict, List, Optional

from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, HTMLResponse, JSONResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

# Ensure project root is on sys.path
_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_ROOT))

from src.core.query_engine.query_processor import QueryProcessor
from src.core.query_engine.hybrid_search import create_hybrid_search
from src.core.query_engine.dense_retriever import create_dense_retriever
from src.core.query_engine.sparse_retriever import create_sparse_retriever
from src.core.query_engine.reranker import create_core_reranker
from src.core.settings import load_settings
from src.core.trace import TraceContext
from src.core.templates import (
    TemplateManager, 
    TemplateConfig, 
    TemplateType, 
    GradeLevel, 
    LearningStyle
)
from src.agents import (
    ConversationAgent,
    LessonHistoryStorage,
    LessonOrchestrator,
    PlannerAgent,
    QueryAgent,
    RetrieverAgent,
    LessonTaskStorage,
    WriterReviewerAgent,
)
from src.ingestion.storage.bm25_indexer import BM25Indexer
from src.ingestion.storage.image_storage import ImageStorage
from src.libs.embedding.embedding_factory import EmbeddingFactory
from src.libs.llm.base_llm import Message
from src.libs.llm.llm_factory import LLMFactory
from src.libs.vector_store.vector_store_factory import VectorStoreFactory
from src.observability.logger import get_logger

logger = get_logger(__name__)

_STATIC_DIR = Path(__file__).parent / "static"
_GEMINI_MODEL_PREFIXES = ("gemini-", "gemma-")
_GEMINI_GATEWAY_BASE_URL = os.environ.get(
    "GEMINI_GATEWAY_BASE_URL",
    "https://gemini-gateway.xn--7dvnlw2c.top/v1",
)
_GEMINI_GATEWAY_API_KEY = os.environ.get(
    "GEMINI_GATEWAY_API_KEY",
    "sk-Ch@1-w3nch&^g-gemini-gateway-2025",
)

# ---------------------------------------------------------------------------
# Data models
# ---------------------------------------------------------------------------

class ChatRequest(BaseModel):
    question: str = Field(..., min_length=1, description="用户问题")
    collection: str = Field(default="default", description="知识库集合名称")
    top_k: Optional[int] = Field(default=None, description="检索数量，None 时使用配置文件默认值")
    use_rerank: bool = Field(default=True, description="是否启用重排序（仍受配置文件约束）")


class LessonPlanRequest(BaseModel):
    topic: str = Field(..., min_length=1, description="教案主题")
    collection: str = Field(default="default", description="知识库集合名称")
    model: Optional[str] = Field(default=None, description="LLM模型名称，不指定则使用默认配置")
    include_background: bool = Field(default=True, description="是否包含背景信息")
    include_facts: bool = Field(default=True, description="是否包含相关常识")
    include_examples: bool = Field(default=True, description="是否包含教学示例")
    template_category: Optional[str] = Field(default=None, description="模板类别")
    template_type: Optional[str] = Field(default=None, description="具体模板类型")
    grade_level: Optional[str] = Field(default=None, description="年级（个性化模板使用）")
    learning_style: Optional[str] = Field(default=None, description="学习风格（个性化模板使用）")
    conversation_state: Optional[Dict[str, Any]] = Field(default=None, description="轻量会话状态")


def _resolve_template_type(req: "LessonPlanRequest") -> Optional[str]:
    if req.template_type:
        return req.template_type
    if req.template_category == "guide":
        return "guide_master"
    if req.template_category == "comprehensive":
        return "comprehensive_master"
    return None


def _resolve_llm_auth_for_model(model_name: Optional[str], settings: Any) -> tuple[Optional[str], Optional[str]]:
    model = str(model_name or "").strip().lower()
    if model.startswith(_GEMINI_MODEL_PREFIXES):
        return _GEMINI_GATEWAY_API_KEY, _GEMINI_GATEWAY_BASE_URL
    return settings.llm.api_key, settings.llm.base_url


def _is_fast_mode_enabled() -> bool:
    return str(os.environ.get("LESSON_FAST_MODE", "false")).strip().lower() in {"1", "true", "yes", "on"}


def _build_api_error_detail(
    *,
    code: str,
    message: str,
    stage: Optional[str] = None,
    trace_id: Optional[str] = None,
) -> Dict[str, Any]:
    payload: Dict[str, Any] = {"code": code, "message": message}
    if stage:
        payload["stage"] = stage
    if trace_id:
        payload["trace_id"] = trace_id
    return payload


class Citation(BaseModel):
    source: str
    score: float
    text: str


class LessonImageResource(BaseModel):
    image_id: str
    url: str
    source: str
    page: Optional[int] = None
    caption: Optional[str] = None


class LessonReviewReportResponse(BaseModel):
    realism_score: int = 0
    pedagogy_score: int = 0
    structure_score: int = 0
    multimodal_score: int = 0
    strengths: List[str] = Field(default_factory=list)
    issues: List[str] = Field(default_factory=list)
    must_fix: List[str] = Field(default_factory=list)


class ChatResponse(BaseModel):
    answer: str
    citations: List[Citation] = Field(default_factory=list)


class LessonPlanResponse(BaseModel):
    topic: str
    subject: Optional[str] = None
    lesson_content: Optional[str] = None  # 完整的教案内容（Markdown格式）
    additional_resources: List[Citation] = Field(default_factory=list)
    image_resources: List[LessonImageResource] = Field(default_factory=list)
    review_report: Optional[LessonReviewReportResponse] = None
    conversation_state: Optional[Dict[str, Any]] = None
    history_records: List[Dict[str, Any]] = Field(default_factory=list)
    execution_plan: Optional[Dict[str, Any]] = None
    planning_mode: Optional[str] = None
    used_autonomous_fallback: bool = False


class LessonHistoryResponse(BaseModel):
    records: List[Dict[str, Any]] = Field(default_factory=list)


class HealthResponse(BaseModel):
    status: str
    components: dict


class LessonTaskCreateResponse(BaseModel):
    task_id: str
    status: str = "queued"


class LessonTaskStatusResponse(BaseModel):
    task_id: str
    status: str
    progress_stage: Optional[str] = None
    result: Optional[LessonPlanResponse] = None
    error: Optional[Dict[str, Any]] = None


# ---------------------------------------------------------------------------
# Component helpers
# ---------------------------------------------------------------------------

def _build_components(settings: Any, collection: str) -> tuple:
    """Initialise RAG components for a given collection (mirrors scripts/query.py)."""
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


def _extract_caption_lookup(metadata: Dict[str, Any]) -> Dict[str, str]:
    captions = metadata.get("image_captions", {})
    if isinstance(captions, dict):
        return {str(k): str(v) for k, v in captions.items()}
    if isinstance(captions, list):
        lookup: Dict[str, str] = {}
        for item in captions:
            if isinstance(item, dict) and item.get("id") and item.get("caption"):
                lookup[str(item["id"])] = str(item["caption"])
        return lookup
    return {}


def _extract_image_ids_from_text(text: Any) -> List[str]:
    if not text:
        return []
    matches = re.findall(r"\[IMAGE:\s*([^\]\s]+)\s*\]", str(text), flags=re.IGNORECASE)
    return list(dict.fromkeys(str(match).strip() for match in matches if str(match).strip()))


def _sanitize_source_path(source_path: Any) -> str:
    if not source_path:
        return "unknown"

    normalized = str(source_path).replace("\\", "/")
    data_index = normalized.find("/data/")
    if data_index != -1:
        return normalized[data_index:]

    relative_data_index = normalized.find("data/")
    if relative_data_index != -1:
        return "/" + normalized[relative_data_index:]

    return Path(normalized).name or "unknown"


def _normalize_image_caption(caption: Optional[str]) -> Optional[str]:
    if not caption:
        return None

    cleaned = " ".join(str(caption).strip().split())
    if not cleaned:
        return None

    boilerplate_patterns = [
        r"^the image contains\s+",
        r"^the image shows\s+",
        r"^the image depicts\s+",
        r"^the figure shows\s+",
        r"^the figure illustrates\s+",
        r"^the text includes\s+",
        r"^this image contains\s+",
        r"^this figure shows\s+",
        r"^图中包含\s*",
        r"^图片中包含\s*",
        r"^该图片展示了\s*",
        r"^该图展示了\s*",
    ]
    for pattern in boilerplate_patterns:
        cleaned = re.sub(pattern, "", cleaned, flags=re.IGNORECASE)

    replacements = {
        "The text includes ": "",
        "There is also ": "",
        "It appears to be ": "",
        "The image also includes ": "",
        "The text in the image includes ": "",
        "which appears to be ": "",
        "and the text ": "；文字：",
    }
    for source, target in replacements.items():
        cleaned = cleaned.replace(source, target)

    cleaned = cleaned.strip(" .;，。；:")
    cleaned = re.sub(r"\s*[:：]\s*", "：", cleaned)
    cleaned = re.sub(r"\s*[,，]\s*", "，", cleaned)
    cleaned = re.sub(r"\s*[;；]\s*", "；", cleaned)

    if len(cleaned) > 120:
        cleaned = cleaned[:117].rstrip("，。；: ") + "..."

    return cleaned or None


def _is_effective_lesson_image(
    caption: Optional[str],
    source_path: Any,
    page: Optional[int],
    image_info: Optional[Dict[str, Any]] = None,
) -> bool:
    text = " ".join(
        part.strip().lower()
        for part in [str(caption or ""), str(source_path or "")]
        if part
    )
    position = image_info.get("position", {}) if isinstance(image_info, dict) else {}
    image_type = str(position.get("type", "")).lower()
    width = int(position.get("width") or 0)
    height = int(position.get("height") or 0)
    image_area = width * height

    invalid_keywords = [
        "logo",
        "university logo",
        "school logo",
        "college logo",
        "watermark",
        "seal",
        "emblem",
        "cover page",
        "title page",
        "table of contents",
        "contents page",
        "目录页",
        "目录",
        "封面",
        "扉页",
        "华东师范大学",
        "east china normal university",
        "校徽",
        "校名",
        "校标",
        "does not contain any text or diagrams",
        "does not contain any text",
        "does not contain diagrams",
        "no text or diagrams",
        "simple graphic design",
        "geometric pattern",
        "abstract pattern",
        "decorative pattern",
        "white and red geometric pattern",
        "没有文字或图表",
        "没有文字或图示",
        "简单几何图形",
        "装饰图案",
        "抽象图案",
    ]
    if any(keyword in text for keyword in invalid_keywords):
        return False

    if page == 1 and image_type == "page_snapshot":
        return False

    if image_type == "page_snapshot" and any(keyword in text for keyword in ["目录", "contents", "table of contents"]):
        return False

    if image_type == "page_crop" and page == 1 and image_area and image_area < 120000:
        return False

    if image_area and image_area < 50000 and any(keyword in text for keyword in ["logo", "校徽", "watermark", "emblem"]):
        return False

    if page == 1 and caption:
        first_page_noise_keywords = [
            "university",
            "学院",
            "大学",
            "课程名称",
            "课件标题",
            "contains the logo",
            "presentation title",
            "course title",
            "institution name",
        ]
        if any(keyword.lower() in text for keyword in first_page_noise_keywords):
            return False

    return True


def _score_lesson_image(
    caption: Optional[str],
    page: Optional[int],
    image_info: Optional[Dict[str, Any]] = None,
) -> int:
    text = str(caption or "").strip().lower()
    position = image_info.get("position", {}) if isinstance(image_info, dict) else {}
    image_type = str(position.get("type", "")).lower()
    width = int(position.get("width") or 0)
    height = int(position.get("height") or 0)
    area = width * height

    score = 0

    high_value_keywords = [
        "架构图",
        "结构图",
        "流程图",
        "示意图",
        "原理图",
        "网络结构",
        "模型结构",
        "模型架构",
        "特征图",
        "卷积",
        "池化",
        "classification",
        "diagram",
        "workflow",
        "pipeline",
        "architecture",
        "chart",
        "graph",
        "plot",
        "result",
        "comparison",
        "accuracy",
        "loss",
        "confusion matrix",
        "heatmap",
    ]
    medium_value_keywords = [
        "实验",
        "结果",
        "案例",
        "对比",
        "曲线",
        "图表",
        "表格",
        "步骤",
        "过程",
        "分析",
        "example",
        "experiment",
        "visualization",
        "figure",
        "table",
    ]
    low_value_keywords = [
        "插图",
        "配图",
        "illustration",
        "screenshot",
    ]

    if any(keyword in text for keyword in high_value_keywords):
        score += 12
    if any(keyword in text for keyword in medium_value_keywords):
        score += 6
    if any(keyword in text for keyword in low_value_keywords):
        score += 2

    if image_type == "page_crop":
        score += 4
    elif image_type == "page_snapshot":
        score -= 3

    if area >= 600000:
        score += 4
    elif area >= 200000:
        score += 2
    elif area and area < 80000:
        score -= 2

    if page is not None:
        if 2 <= page <= 12:
            score += 3
        elif page > 20:
            score -= 1

    return score


def _extract_image_resources(
    results: List[Any],
    image_storage: Optional[ImageStorage] = None,
    collection: Optional[str] = None,
    max_images: int = 6,
) -> List[LessonImageResource]:
    seen_ids = set()
    scored_resources: List[tuple[int, LessonImageResource]] = []
    doc_hashes = []
    preferred_pages: Dict[str, List[int]] = {}
    direct_candidate_count = 0
    direct_kept_count = 0
    indexed_candidate_count = 0
    indexed_kept_count = 0

    for result_index, result in enumerate(results):
        metadata = result.metadata or {}
        images = metadata.get("images", [])
        doc_hash = metadata.get("doc_hash")
        if doc_hash and doc_hash not in doc_hashes:
            doc_hashes.append(doc_hash)
        page_num = metadata.get("page_num")
        if doc_hash and isinstance(page_num, int):
            preferred_pages.setdefault(str(doc_hash), [])
            if page_num not in preferred_pages[str(doc_hash)]:
                preferred_pages[str(doc_hash)].append(page_num)
        if not isinstance(images, list):
            continue

        caption_lookup = _extract_caption_lookup(metadata)
        source_path = metadata.get("source_path", "unknown")
        placeholder_image_ids = _extract_image_ids_from_text(result.text)

        for image_info in images:
            if not isinstance(image_info, dict):
                continue

            direct_candidate_count += 1
            image_id = image_info.get("id")
            image_path = image_info.get("path")
            page_num = image_info.get("page") or metadata.get("page_num")
            caption = _normalize_image_caption(caption_lookup.get(str(image_id)))
            if not image_id or not image_path or image_id in seen_ids:
                continue
            if not _is_effective_lesson_image(caption, source_path, page_num, image_info=image_info):
                continue

            path_obj = Path(image_path)
            if not path_obj.is_absolute():
                path_obj = (_ROOT / image_path).resolve()

            if not path_obj.exists() and image_storage is not None:
                indexed_path = image_storage.get_image_path(str(image_id))
                if indexed_path:
                    path_obj = Path(indexed_path)

            if not path_obj.exists():
                logger.info(
                    "lesson_image.direct_missing image_id=%s source=%s page=%s",
                    image_id,
                    _sanitize_source_path(source_path),
                    page_num,
                )
                continue

            seen_ids.add(image_id)
            rank_bonus = max(0, 8 - result_index)
            direct_kept_count += 1
            scored_resources.append(
                (
                    _score_lesson_image(caption, page_num, image_info=image_info) + rank_bonus,
                    LessonImageResource(
                        image_id=str(image_id),
                        url=f"/lesson-plan-image/{image_id}",
                        source=_sanitize_source_path(source_path),
                        page=page_num,
                        caption=caption,
                    ),
                )
            )

        for image_id in placeholder_image_ids:
            if image_id in seen_ids or image_storage is None:
                continue

            direct_candidate_count += 1
            indexed_path = image_storage.get_image_path(str(image_id))
            if not indexed_path:
                logger.info(
                    "lesson_image.placeholder_missing image_id=%s source=%s page=%s",
                    image_id,
                    _sanitize_source_path(source_path),
                    page_num,
                )
                continue

            if not _is_effective_lesson_image(caption_lookup.get(str(image_id)), source_path, page_num):
                continue

            path_obj = Path(indexed_path)
            if not path_obj.exists():
                logger.info(
                    "lesson_image.placeholder_file_missing image_id=%s source=%s page=%s",
                    image_id,
                    _sanitize_source_path(source_path),
                    page_num,
                )
                continue

            seen_ids.add(image_id)
            rank_bonus = max(0, 8 - result_index)
            direct_kept_count += 1
            scored_resources.append(
                (
                    _score_lesson_image(caption_lookup.get(str(image_id)), page_num) + rank_bonus,
                    LessonImageResource(
                        image_id=str(image_id),
                        url=f"/lesson-plan-image/{image_id}",
                        source=_sanitize_source_path(source_path),
                        page=page_num,
                        caption=_normalize_image_caption(caption_lookup.get(str(image_id))),
                    ),
                )
            )

    if image_storage is None:
        scored_resources.sort(
            key=lambda item: (
                -item[0],
                item[1].page if item[1].page is not None else 9999,
                item[1].image_id,
            )
        )
        return [resource for _, resource in scored_resources[:max_images]]

    # Fallback: some retrieved chunks don't carry image metadata after splitting/rerank.
    # Try to recover original PDF images by doc_hash from persistent image index.
    for doc_hash in doc_hashes:
        try:
            indexed_images = image_storage.list_images(
                collection=collection,
                doc_hash=doc_hash,
            )
        except Exception:
            indexed_images = []

        if not indexed_images:
            try:
                indexed_images = image_storage.list_images(doc_hash=doc_hash)
            except Exception:
                indexed_images = []

        for indexed in indexed_images:
            indexed_candidate_count += 1
            image_id = indexed.get("image_id")
            file_path = indexed.get("file_path")
            page_num = indexed.get("page_num")
            if not image_id or not file_path or image_id in seen_ids:
                continue
            if not _is_effective_lesson_image(None, file_path, page_num):
                continue

            path_obj = Path(file_path)
            if not path_obj.exists():
                continue

            seen_ids.add(image_id)
            page_bonus = 0
            doc_pages = preferred_pages.get(str(doc_hash), [])
            if isinstance(page_num, int) and doc_pages:
                page_distance = min(abs(page_num - candidate_page) for candidate_page in doc_pages)
                if page_distance == 0:
                    page_bonus = 10
                elif page_distance == 1:
                    page_bonus = 6
                elif page_distance == 2:
                    page_bonus = 3
            indexed_kept_count += 1
            scored_resources.append(
                (
                    _score_lesson_image(None, page_num) + page_bonus,
                    LessonImageResource(
                        image_id=str(image_id),
                        url=f"/lesson-plan-image/{image_id}",
                        source=_sanitize_source_path(path_obj),
                        page=page_num,
                        caption=None,
                    ),
                )
            )

    scored_resources.sort(
        key=lambda item: (
            -item[0],
            item[1].page if item[1].page is not None else 9999,
            item[1].image_id,
        )
    )
    final_resources = [resource for _, resource in scored_resources[:max_images]]
    logger.info(
        "lesson_image.extract results=%s doc_hashes=%s direct_candidates=%s direct_kept=%s indexed_candidates=%s indexed_kept=%s final=%s collection=%s",
        len(results),
        len(doc_hashes),
        direct_candidate_count,
        direct_kept_count,
        indexed_candidate_count,
        indexed_kept_count,
        len(final_resources),
        collection,
    )
    return final_resources


def _score_result_for_visual_lesson(result: Any, query_plan: Any) -> float:
    metadata = result.metadata or {}
    text = str(result.text or "")
    score = float(getattr(result, "score", 0.0) or 0.0)

    image_focus = False
    if isinstance(query_plan, dict):
        image_focus = bool(query_plan.get("image_focus", False))
    else:
        image_focus = bool(getattr(query_plan, "image_focus", False))
    if not query_plan or not image_focus:
        return score

    images = metadata.get("images", [])
    image_captions = metadata.get("image_captions", [])
    has_images = isinstance(images, list) and len(images) > 0
    has_captions = bool(image_captions)
    has_placeholder = "[IMAGE:" in text

    visual_keywords = [
        "结构图",
        "流程图",
        "示意图",
        "图解",
        "模型结构",
        "网络结构",
        "实验结果",
        "图表",
        "曲线",
        "对比图",
        "feature map",
        "architecture",
        "workflow",
        "diagram",
        "figure",
        "plot",
        "chart",
        "result",
    ]
    visual_hits = sum(1 for keyword in visual_keywords if keyword.lower() in text.lower())

    boost = 0.0
    if has_images:
        boost += 0.12
    if has_captions:
        boost += 0.08
    if has_placeholder:
        boost += 0.08
    if visual_hits:
        boost += min(0.12, visual_hits * 0.03)

    page_num = metadata.get("page_num")
    if isinstance(page_num, int) and 2 <= page_num <= 12:
        boost += 0.03

    return score + boost


def _prioritize_visual_results(results: List[Any], query_plan: Any) -> List[Any]:
    image_focus = False
    if isinstance(query_plan, dict):
        image_focus = bool(query_plan.get("image_focus", False))
    else:
        image_focus = bool(getattr(query_plan, "image_focus", False))
    if not results or not query_plan or not image_focus:
        return results

    return sorted(
        results,
        key=lambda item: _score_result_for_visual_lesson(item, query_plan),
        reverse=True,
    )


def _extract_topic_terms_for_filter(topic: str) -> List[str]:
    raw_terms = re.findall(r"[\u4e00-\u9fff]{2,}|[A-Za-z][A-Za-z0-9_-]{2,}", str(topic or ""))
    stop_terms = {
        "教案", "模板", "综合模板", "导学案", "综合教学模板", "教学", "内容", "主题",
        "lesson", "guide", "template", "teaching",
    }
    terms: List[str] = []
    for term in raw_terms:
        normalized = term.strip().lower()
        if not normalized or normalized in stop_terms:
            continue
        terms.append(normalized)
        if re.fullmatch(r"[\u4e00-\u9fff]{4,}", normalized):
            length = len(normalized)
            for window in (2, 3, 4):
                if length <= window:
                    continue
                for start in range(0, length - window + 1):
                    piece = normalized[start:start + window]
                    if piece not in stop_terms:
                        terms.append(piece)
    return list(dict.fromkeys(terms))


def _is_result_relevant_to_topic(topic: str, result: Any) -> bool:
    topic_terms = _extract_topic_terms_for_filter(topic)
    if not topic_terms:
        return True

    text_parts = [str(result.text or "").lower()]
    metadata = result.metadata or {}
    source_path = str(metadata.get("source_path", "") or "").lower()
    text_parts.append(source_path)
    combined_text = " ".join(text_parts)

    matched_terms = [term for term in topic_terms if term in combined_text]
    required_matches = 1 if len(topic_terms) <= 2 else 2
    return len(matched_terms) >= required_matches


def _looks_like_lesson_refusal(content: str) -> bool:
    text = str(content or "")
    refusal_patterns = [
        "基于当前上下文，无法",
        "基于提供的上下文，无法",
        "上下文内容并未涉及",
        "没有涉及",
        "因此，基于当前上下文，无法",
        "请补充相关",
        "很抱歉",
        "谢谢您的理解",
        "如需该主题的教案",
    ]
    return any(re.search(pattern, text, flags=re.IGNORECASE) for pattern in refusal_patterns)


def _build_comprehensive_image_markdown(image_resources: List[LessonImageResource]) -> str:
    if not image_resources:
        return ""

    lines = [
        "## 图文讲解素材",
        "以下配图来自检索到的 PDF 原始资料，可直接用于课堂讲解与投屏展示。",
        "",
    ]

    for index, image in enumerate(image_resources, start=1):
        title = f"### 配图{index}"
        if image.page:
            title += f"（第 {image.page} 页）"
        lines.append(title)
        lines.append(f"![配图{index}]({image.url})")

        caption_parts: List[str] = []
        if image.caption:
            caption_parts.append(image.caption.strip())
        if image.source:
            source_text = image.source
            if image.page:
                source_text += f" · 第 {image.page} 页"
            caption_parts.append(f"来源：{source_text}")
        if caption_parts:
            lines.append("")
            lines.append("> " + " | ".join(caption_parts))
        lines.append("")

    return "\n".join(lines).strip()


def _to_review_report_response(report: Any) -> Optional[LessonReviewReportResponse]:
    if report is None:
        return None
    if is_dataclass(report):
        payload = asdict(report)
    elif isinstance(report, dict):
        payload = report
    else:
        return None
    return LessonReviewReportResponse(**payload)


def _format_image_markdown_block(image: LessonImageResource, index: int) -> str:
    title = f"![配图{index}]({image.url})"
    caption_parts: List[str] = []
    if image.caption:
        caption_parts.append(image.caption.strip())
    if image.source:
        source_text = image.source
        if image.page:
            source_text += f" · 第 {image.page} 页"
        caption_parts.append(f"来源：{source_text}")
    if caption_parts:
        return f"{title}\n\n> " + " | ".join(caption_parts)
    return title


def _integrate_images_into_markdown(
    lesson_plan_content: str,
    image_resources: List[LessonImageResource],
) -> str:
    """Insert retrieved images near the first textual mention of 配图N.

    Falls back to an appendix block for images not explicitly referenced by the LLM.
    """
    if not lesson_plan_content or not image_resources:
        return lesson_plan_content

    lines = lesson_plan_content.splitlines()
    inserted_indices = set()

    for index, image in enumerate(image_resources, start=1):
        marker = f"配图{index}"
        insertion_block = _format_image_markdown_block(image, index).splitlines()

        for line_idx, line in enumerate(lines):
            if marker in line:
                insert_at = line_idx + 1
                if insert_at < len(lines) and lines[insert_at].strip().startswith("![配图"):
                    inserted_indices.add(index)
                    break

                block = [""] + insertion_block + [""]
                lines[insert_at:insert_at] = block
                inserted_indices.add(index)
                break

    remaining = [
        (idx, image)
        for idx, image in enumerate(image_resources, start=1)
        if idx not in inserted_indices
    ]
    if remaining:
        # Prefer in-body insertion: distribute unreferenced images across section anchors.
        # This avoids all images being appended to the end when the model didn't mention 配图N.
        section_anchor_indexes = [
            i + 1
            for i, line in enumerate(lines)
            if re.match(r"^\s*(#{1,6}\s+|\d+[\.、]\s+|[一二三四五六七八九十]+[、\.]\s+)", line.strip())
        ]

        if not section_anchor_indexes:
            # Fallback: choose evenly spaced insertion points in the body.
            total = max(len(lines), 1)
            section_anchor_indexes = [
                min(int((k + 1) * total / (len(remaining) + 1)), len(lines))
                for k in range(len(remaining))
            ]

        # Deduplicate and keep stable order
        dedup_anchors: List[int] = []
        for anchor in section_anchor_indexes:
            if anchor not in dedup_anchors:
                dedup_anchors.append(anchor)
        section_anchor_indexes = dedup_anchors or [len(lines)]

        # Spread images across anchors (round-robin), keeping global image numbering.
        inserted_offset = 0
        for item_idx, (global_index, image) in enumerate(remaining):
            anchor = section_anchor_indexes[item_idx % len(section_anchor_indexes)]
            insert_at = max(0, min(anchor + inserted_offset, len(lines)))
            block = [""] + _format_image_markdown_block(image, global_index).splitlines() + [""]
            lines[insert_at:insert_at] = block
            inserted_offset += len(block)

    return _remove_dangling_image_references("\n".join(lines))


def _remove_dangling_image_references(content: str) -> str:
    """Remove references like 配图4 when the corresponding image block is absent."""
    if not content:
        return content

    referenced_indices = {int(match) for match in re.findall(r"配图(\d+)", content)}
    rendered_indices = {
        int(match)
        for match in re.findall(r"!\[配图(\d+)\]", content)
    }
    dangling = referenced_indices - rendered_indices
    if not dangling:
        return content

    cleaned = content
    for index in sorted(dangling, reverse=True):
        cleaned = re.sub(rf"[（(]?\s*见?配图{index}\s*[）)]?", "", cleaned)
        cleaned = re.sub(rf"结合配图{index}", "结合示意内容", cleaned)
        cleaned = re.sub(rf"参考配图{index}", "参考相关示意", cleaned)
        cleaned = re.sub(rf"观察配图{index}", "观察相关示意", cleaned)
        cleaned = re.sub(rf"展示配图{index}", "展示相关示意", cleaned)
        cleaned = re.sub(rf"配图{index}", "相关示意", cleaned)

    cleaned = re.sub(r"\n{3,}", "\n\n", cleaned)
    cleaned = re.sub(r"[ \t]{2,}", " ", cleaned)
    return cleaned.strip()


def _build_prompt(question: str, contexts: List[Any]) -> List[Message]:
    """Build LLM messages from retrieved contexts."""
    context_text = "\n\n".join(
        f"[{i+1}] {(r.text or '').strip()[:500]}"
        for i, r in enumerate(contexts)
    )
    return [
        Message(
            role="system",
            content=(
                "你是一个知识库问答助手。请严格基于以下提供的上下文内容回答用户问题。"
                "如果上下文中没有足够信息，请明确告知用户，不要凭空捏造。"
            ),
        ),
        Message(
            role="user",
            content=f"上下文：\n{context_text}\n\n问题：{question}",
        ),
    ]


def _build_lesson_plan_prompt(topic: str, contexts: List[Any], include_background: bool, include_facts: bool, include_examples: bool) -> List[Message]:
    """Build LLM messages for lesson plan generation."""
    context_text = "\n\n".join(
        f"[{i+1}] {(r.text or '').strip()[:500]}"
        for i, r in enumerate(contexts)
    )
    
    instructions = []
    if include_background:
        instructions.append("""1. 背景信息（详细展开）：
   - 历史背景：详细介绍该主题的历史发展脉络，包括重要时间节点、关键人物和事件
   - 科学意义：阐述该主题在科学史上的重要地位和影响
   - 现实应用：说明该主题在现代社会的应用价值和实际意义
   - 相关概念：介绍与该主题相关的其他重要概念和理论
   
   **人物背景（如适用）**：
   - 生平简介：详细介绍相关科学家的生平、出生地、教育背景、主要成就
   - 时代背景：介绍科学家所处的时代背景、社会环境、科技发展水平
   - 研究历程：详细描述科学家的研究过程、遇到的困难、突破的关键时刻
   - 人物性格：介绍科学家的性格特点、研究风格、轶事趣闻
   - 历史评价：介绍该科学家在历史上的地位和影响
   
   **故事背景（如适用）**：
   - 发现过程：详细描述重要发现的过程，包括时间、地点、关键事件
   - 前因后果：介绍发现的背景、动机、以及后续影响
   - 争议与挑战：介绍发现过程中遇到的质疑、争议和挑战
   - 社会反响：介绍发现当时社会的反应和评价
   - 历史意义：阐述该发现对人类文明的深远影响""")
    
    if include_facts:
        instructions.append("""2. 核心知识点（详细列出）：
   - 基本概念：清晰定义该主题的核心概念和术语
   - 基本原理：详细阐述该主题的基本原理和规律
   - 重要公式：列出相关的数学公式和表达式（如有）
   - 实验现象：描述相关的实验现象和观测结果
   - 常见误区：指出学生容易产生的误解和错误认识
   
   **相关人物与事件**：
   - 同时代科学家：介绍同时期的其他重要科学家及其贡献
   - 前驱工作：介绍该发现之前的相关研究和理论
   - 后续发展：介绍该发现之后的重要进展和突破
   - 跨学科影响：介绍该发现对其他学科领域的影响""")
    
    if include_examples:
        instructions.append("""3. 教学活动设计（具体可行）：
   - 演示实验：提供具体的实验设计，包括所需材料、操作步骤和预期结果
   - 课堂活动：设计互动性强的课堂活动，激发学生兴趣
   - 案例分析：提供真实的案例和应用场景
   - 思考问题：设计启发性的思考问题，引导学生深入思考
   - 拓展阅读：推荐相关的课外阅读材料和资源
   
   **情境教学设计**：
   - 历史重现：设计让学生体验科学家发现过程的活动
   - 角色扮演：设计让学生扮演科学家、进行辩论或讨论的活动
   - 时间线构建：让学生构建相关发现的时间线
   - 对比分析：让学生对比不同科学家的贡献或不同理论的异同
   - 实地考察：推荐相关的博物馆、实验室等实地考察资源""")
    
    instructions_text = "\n\n".join(instructions)
    
    return [
        Message(
            role="system",
            content=(
                f"你是一名经验丰富的教师，擅长准备详细、深入的教案。请基于以下上下文内容，为主题'{topic}'生成一份详细的教案。\n"
                "请以上下文内容为主要依据，同时允许结合通用学科知识、课堂经验和教学设计方法做合理拓展。\n"
                "不要伪造具体出处、页码或事实来源；如果上下文不足，可补充教学性内容，但请避免编造可核验细节。\n"
                "最终输出必须为简体中文，不要包含英文句子或英文结尾。\n"
                "首先，请根据主题内容推断这属于哪个学科（如物理、化学、生物、数学、语文、英语、历史、地理等），并在教案开头明确指出。\n\n"
                "请按照以下结构组织教案：\n\n"
                f"{instructions_text}\n\n"
                "4. 参考资源：列出使用的参考资料，包括来源文档和页码"
            ),
        ),
        Message(
            role="user",
            content=f"上下文：\n{context_text}\n\n请为'{topic}'生成一份详细的教案。",
        ),
    ]


def _build_lesson_plan_prompt_fallback(req: "LessonPlanRequest") -> List[Message]:
    """Build LLM messages for lesson plan generation when no context is available."""
    instructions = []
    if req.include_background:
        instructions.append("""1. 背景信息（详细展开）：
   - 历史背景：详细介绍该主题的历史发展脉络，包括重要时间节点、关键人物和事件
   - 科学意义：阐述该主题在科学史上的重要地位和影响
   - 现实应用：说明该主题在现代社会的应用价值和实际意义
   - 相关概念：介绍与该主题相关的其他重要概念和理论""")
    
    if req.include_facts:
        instructions.append("""2. 核心知识点（详细列出）：
   - 基本概念：清晰定义该主题的核心概念和术语
   - 基本原理：详细阐述该主题的基本原理和规律
   - 重要公式：列出相关的数学公式和表达式（如有）
   - 实验现象：描述相关的实验现象和观测结果
   - 常见误区：指出学生容易产生的误解和错误认识""")
    
    if req.include_examples:
        instructions.append("""3. 教学活动设计（具体可行）：
   - 演示实验：提供具体的实验设计，包括所需材料、操作步骤和预期结果
   - 课堂活动：设计互动性强的课堂活动，激发学生兴趣
   - 案例分析：提供真实的案例和应用场景
   - 思考问题：设计启发性的思考问题，引导学生深入思考
   - 拓展阅读：推荐相关的课外阅读材料和资源""")
    
    instructions_text = "\n\n".join(instructions)
    
    return [
        Message(
            role="system",
            content=(
                f"你是一名经验丰富的教师，擅长准备详细、深入的教案。当前知识库中没有找到与主题'{req.topic}'相关的内容，\n"
                "请基于你自己的知识与教学经验为该主题生成一份详细的教案，并主动补充更有教学价值的延伸内容。\n\n"
                "最终输出必须为简体中文，不要包含英文句子或英文结尾。\n"
                "首先，请根据主题内容推断这属于哪个学科（如物理、化学、生物、数学、语文、英语、历史、地理等），并在教案开头明确指出。\n\n"
                "请按照以下结构组织教案：\n\n"
                f"{instructions_text}\n\n"
                "4. 参考资源：列出使用的参考资料（基于你的知识）"
            ),
        ),
        Message(
            role="user",
            content=f"请为'{req.topic}'生成一份详细的教案。",
        ),
    ]


# ---------------------------------------------------------------------------
# Lesson task helpers
# ---------------------------------------------------------------------------

def _task_set(task_id: str, **payload: Any) -> None:
    storage = payload.pop("_storage")
    current = storage.get(task_id) or {"task_id": task_id}
    current.update(payload)
    storage.upsert(
        task_id=task_id,
        status=str(current.get("status") or "queued"),
        progress_stage=current.get("progress_stage"),
        result=current.get("result") if isinstance(current.get("result"), dict) else None,
        error=current.get("error") if isinstance(current.get("error"), dict) else None,
        created_at=float(current.get("created_at") or time.time()),
        finished_at=float(current["finished_at"]) if current.get("finished_at") is not None else None,
    )


def _task_get(task_id: str, storage: Any) -> Optional[Dict[str, Any]]:
    return storage.get(task_id)


def _task_cleanup(storage: Any, max_age_seconds: int = 3600) -> None:
    storage.cleanup(max_age_seconds=max_age_seconds)


def _generate_lesson_plan_internal(
    req: LessonPlanRequest,
    request: Request,
    progress_callback: Optional[Callable[[str, Dict[str, Any]], None]] = None,
) -> LessonPlanResponse:
    started = time.monotonic()
    state = request.app.state
    settings = state.settings
    hybrid_search = state.hybrid_search
    reranker = state.reranker
    fast_mode = _is_fast_mode_enabled()
    logger.info(
        "lesson_plan.internal_start topic=%s template_category=%s model=%s fast_mode=%s collection=%s",
        req.topic,
        req.template_category,
        req.model or state.settings.llm.model,
        fast_mode,
        req.collection,
    )
    if progress_callback is not None:
        progress_callback(
            "internal_start",
            {
                "topic": req.topic,
                "template_category": req.template_category or "comprehensive",
                "model": req.model or state.settings.llm.model,
                "collection": req.collection,
            },
        )

    # 动态切换模型
    if req.model:
        from src.libs.llm.llm_factory import LLMFactory
        from dataclasses import replace as dc_replace
        from src.core.settings import LLMSettings
        resolved_api_key, resolved_base_url = _resolve_llm_auth_for_model(req.model, settings)

        # 创建新的LLM配置
        new_llm_config = LLMSettings(
            provider=settings.llm.provider,
            model=req.model,
            api_key=resolved_api_key,
            base_url=resolved_base_url,
            temperature=settings.llm.temperature,
            max_tokens=settings.llm.max_tokens,
        )
        new_settings = dc_replace(settings, llm=new_llm_config)
        llm = LLMFactory.create(new_settings)
        logger.info(f"Using custom model: {req.model}")
    else:
        llm = state.llm

    top_k = int(os.environ.get("LESSON_FAST_TOP_K", "8")) if fast_mode else 15
    trace = TraceContext(trace_type="lesson_plan")
    resolved_template_type = _resolve_template_type(req)
    conversation_agent = ConversationAgent()
    planner_agent = PlannerAgent(llm=None if fast_mode else llm)
    query_agent = QueryAgent()
    conversation_state = conversation_agent.prepare_state(req, req.conversation_state)
    retriever_agent = RetrieverAgent(
        hybrid_search=hybrid_search,
        reranker=reranker,
        trace=trace,
        top_k=top_k,
        prioritize_visual_results=_prioritize_visual_results,
        relevance_check=_is_result_relevant_to_topic,
        extract_image_resources=_extract_image_resources,
        sanitize_source_path=_sanitize_source_path,
        image_storage=request.app.state.image_storage,
        collection=req.collection,
        enable_rerank=not fast_mode,
        enable_image_extraction=not fast_mode,
        max_search_queries=2 if fast_mode else None,
    )
    writer_reviewer_agent = WriterReviewerAgent(
        llm=llm,
        template_manager=TemplateManager(),
        trace=trace,
        request=req,
        resolved_template_type=resolved_template_type,
        build_default_prompt=_build_lesson_plan_prompt,
        build_fallback_prompt=_build_lesson_plan_prompt_fallback,
        integrate_images=_integrate_images_into_markdown,
    )
    orchestrator = LessonOrchestrator(
        planner_agent=planner_agent,
        query_agent=query_agent,
        retriever_agent=retriever_agent,
        writer_reviewer_agent=writer_reviewer_agent,
        conversation_agent=conversation_agent,
        trace=trace,
        progress_callback=progress_callback,
    )
    trace.metadata["fast_mode"] = fast_mode
    if fast_mode:
        trace.record_stage(
            "lesson_fast_mode",
            {
                "enabled": True,
                "planner_llm_disabled": True,
                "rerank_disabled": True,
                "image_extraction_disabled": True,
                "top_k": top_k,
                "max_search_queries": 2,
            },
        )

    orchestration_output = orchestrator.run(
        topic=req.topic,
        template_category=req.template_category,
        conversation_state=conversation_state,
    )
    logger.info(
        "lesson_plan.orchestration_done elapsed_ms=%.1f topic=%s",
        (time.monotonic() - started) * 1000,
        req.topic,
    )
    if progress_callback is not None:
        progress_callback(
            "orchestration_done",
            {
                "elapsed_ms": (time.monotonic() - started) * 1000,
                "topic": req.topic,
            },
        )
    execution_plan = dict(orchestration_output["execution_plan"])
    query_plan = dict(orchestration_output["query_plan"])
    relevant_results = list(orchestration_output["results"] or [])
    image_resources = list(orchestration_output["image_resources"] or [])
    lesson_plan_content = str(orchestration_output["lesson_plan_content"] or "")
    subject = orchestration_output.get("subject")
    review_report_payload = orchestration_output.get("review_report")
    review_notes = list(orchestration_output.get("review_notes") or [])
    generation_metadata = dict(orchestration_output.get("generation_metadata") or {})
    finalized_conversation = orchestration_output["conversation_state"]
    citations = [
        Citation(
            source=str(item.get("source") or "unknown"),
            score=float(item.get("score") or 0.0),
            text=str(item.get("text") or ""),
        )
        for item in (orchestration_output.get("citations") or [])
    ]

    if _looks_like_lesson_refusal(lesson_plan_content):
        recovery_messages = [
            Message(
                role="system",
                content=(
                    "你是一名经验丰富的一线教师与教研组长。请直接基于主题与通用学科知识生成完整教案，"
                    "不要输出拒绝、资料不足、要求补充材料等语句。"
                ),
            ),
            Message(role="user", content=f"请为主题“{req.topic}”生成完整成稿。"),
        ]
        lesson_plan_content = llm.chat(recovery_messages).content
        trace.record_stage(
            "lesson_agent_final_refusal_recovery",
            {"applied": True},
        )

    # 添加元数据
    trace.metadata["topic"] = req.topic
    trace.metadata["collection"] = req.collection
    trace.metadata["model"] = req.model or settings.llm.model
    trace.metadata["include_background"] = req.include_background
    trace.metadata["include_facts"] = req.include_facts
    trace.metadata["include_examples"] = req.include_examples
    trace.metadata["template_category"] = req.template_category
    trace.metadata["template_type"] = req.template_type
    trace.metadata["session_id"] = finalized_conversation.session_id
    trace.metadata["agent_protocol"] = "lesson_agent_msg_v1"
    trace.metadata["query_plan"] = dict(query_plan)
    trace.metadata["execution_plan"] = dict(execution_plan)
    trace.metadata["fast_mode"] = fast_mode
    if req.grade_level:
        trace.metadata["grade_level"] = req.grade_level
    if req.learning_style:
        trace.metadata["learning_style"] = req.learning_style

    if isinstance(review_report_payload, dict):
        review_must_fix = list(review_report_payload.get("must_fix") or [])
    else:
        review_must_fix = list(getattr(review_report_payload, "must_fix", []) or [])

    trace.record_stage("lesson_agent_complete", {
        "model": req.model or settings.llm.model,
        "has_context": len(relevant_results) > 0,
        "image_count": len(image_resources),
        "subject": subject,
        "template_type": resolved_template_type,
        "planning_mode": execution_plan.get("generation_mode"),
        "review_must_fix_count": len(review_must_fix),
        "session_id": finalized_conversation.session_id,
    })

    # 保存trace
    trace.finish()
    from src.core.trace.trace_collector import TraceCollector
    TraceCollector().collect(trace)

    history_records: List[Dict[str, Any]] = []
    try:
        lesson_preview = re.sub(r"\s+", " ", lesson_plan_content or "").strip()[:180]
        request.app.state.history_storage.add_record(
            session_id=finalized_conversation.session_id,
            topic=req.topic,
            template_category=req.template_category,
            template_label="综合模板" if req.template_category == "comprehensive" else "导学案模板",
            subject=subject,
            created_at=finalized_conversation.updated_at,
            conversation_state=asdict(finalized_conversation),
            lesson_preview=lesson_preview,
            lesson_content=lesson_plan_content,
            planning_mode=execution_plan.get("generation_mode"),
            used_autonomous_fallback=bool(
                generation_metadata.get("forced_autonomous_retry_after_refusal")
                or generation_metadata.get("forced_autonomous_retry_after_review")
                or execution_plan.get("generation_mode") == "autonomous"
            ),
        )
        history_records = request.app.state.history_storage.list_records(
            limit=8,
            session_id=finalized_conversation.session_id,
        )
    except Exception:
        logger.exception("Failed to persist lesson history")

    return LessonPlanResponse(
        topic=req.topic,
        subject=subject,
        lesson_content=lesson_plan_content,
        additional_resources=citations,
        image_resources=image_resources,
        review_report=_to_review_report_response(review_report_payload),
        conversation_state=asdict(finalized_conversation),
        history_records=history_records,
        execution_plan=dict(execution_plan),
        planning_mode=execution_plan.get("generation_mode"),
        used_autonomous_fallback=bool(
            generation_metadata.get("forced_autonomous_retry_after_refusal")
            or generation_metadata.get("forced_autonomous_retry_after_review")
            or execution_plan.get("generation_mode") == "autonomous"
        ),
    )


def _format_sse_event(event: str, payload: Dict[str, Any]) -> bytes:
    return f"event: {event}\ndata: {json.dumps(payload, ensure_ascii=False)}\n\n".encode("utf-8")


# ---------------------------------------------------------------------------
# App lifecycle
# ---------------------------------------------------------------------------

@asynccontextmanager
async def lifespan(app: FastAPI):
    config_path = os.environ.get("CHAT_CONFIG", str(_ROOT / "config" / "settings.yaml"))
    logger.info(f"Loading settings from: {config_path}")
    settings = load_settings(config_path)

    collection = settings.vector_store.collection_name
    hybrid_search, reranker = _build_components(settings, collection)
    llm = LLMFactory.create(settings)
    image_storage = ImageStorage(
        db_path=str(_ROOT / "data" / "db" / "image_index.db"),
        images_root=str(_ROOT / "data" / "images"),
    )
    history_storage = LessonHistoryStorage(
        db_path=str(_ROOT / "data" / "db" / "lesson_history.db"),
    )
    task_storage = LessonTaskStorage(
        db_path=str(_ROOT / "data" / "db" / "lesson_tasks.db"),
    )

    app.state.settings = settings
    app.state.hybrid_search = hybrid_search
    app.state.reranker = reranker
    app.state.llm = llm
    app.state.image_storage = image_storage
    app.state.history_storage = history_storage
    app.state.task_storage = task_storage
    app.state.default_collection = collection

    logger.info("Chat API components initialised successfully")
    yield
    logger.info("Chat API shutting down")
    image_storage.close()


app = FastAPI(title="RAG Chat API", lifespan=lifespan)

# Mount static files
app.mount("/static", StaticFiles(directory=_STATIC_DIR), name="static")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost", "http://127.0.0.1"],
    allow_origin_regex=r"http://(localhost|127\.0\.0\.1)(:\d+)?",
    allow_methods=["*"],
    allow_headers=["*"],
)


# ---------------------------------------------------------------------------
# Exception handler
# ---------------------------------------------------------------------------

@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception):
    logger.exception(f"Unhandled error on {request.url}: {exc}")
    return JSONResponse(
        status_code=500,
        content={
            "error": "服务器内部错误，请稍后重试",
            "detail": _build_api_error_detail(
                code="INTERNAL_SERVER_ERROR",
                message="服务器内部错误，请稍后重试",
                stage="global",
            ),
        },
    )


# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------

@app.get("/", response_class=HTMLResponse)
async def serve_ui():
    html_file = _STATIC_DIR / "lesson-plan.html"
    if not html_file.exists():
        raise HTTPException(status_code=404, detail="UI file not found")
    return HTMLResponse(content=html_file.read_text(encoding="utf-8"))


@app.get("/lesson-plan.html", response_class=HTMLResponse)
async def serve_lesson_plan_ui():
    html_file = _STATIC_DIR / "lesson-plan.html"
    if not html_file.exists():
        raise HTTPException(status_code=404, detail="Lesson Plan UI file not found")
    return HTMLResponse(content=html_file.read_text(encoding="utf-8"))


@app.get("/lesson-plan-image/{image_id}")
async def serve_lesson_plan_image(image_id: str, request: Request):
    image_storage = request.app.state.image_storage
    image_path = image_storage.get_image_path(image_id)
    if not image_path:
        raise HTTPException(status_code=404, detail="Image not found")

    path = Path(image_path)
    if not path.exists() or not path.is_file():
        raise HTTPException(status_code=404, detail="Image file not found")

    return FileResponse(path)


@app.get("/health", response_model=HealthResponse)
async def health(request: Request):
    return HealthResponse(
        status="ok",
        components={
            "hybrid_search": request.app.state.hybrid_search is not None,
            "reranker": request.app.state.reranker is not None,
            "llm": request.app.state.llm is not None,
        },
    )


@app.get("/lesson-history", response_model=LessonHistoryResponse)
async def get_lesson_history(request: Request, session_id: Optional[str] = None, limit: int = 8):
    storage = request.app.state.history_storage
    safe_limit = max(1, min(limit, 20))
    records = storage.list_records(limit=safe_limit, session_id=session_id)
    return LessonHistoryResponse(records=records)


@app.delete("/lesson-history/{record_id}")
async def delete_lesson_history(record_id: int, request: Request):
    storage = request.app.state.history_storage
    storage.delete_record(record_id)
    return {"ok": True}


@app.post("/chat", response_model=ChatResponse)
async def chat(req: ChatRequest, request: Request):
    state = request.app.state
    settings = state.settings
    hybrid_search = state.hybrid_search
    reranker = state.reranker
    llm = state.llm

    top_k = req.top_k or settings.retrieval.fusion_top_k
    trace = TraceContext(trace_type="chat")

    # 1. Hybrid search
    try:
        hybrid_result = hybrid_search.search(
            query=req.question,
            top_k=top_k,
            filters=None,
            trace=trace,
        )
    except Exception as e:
        logger.exception("HybridSearch failed")
        raise HTTPException(
            status_code=500,
            detail=_build_api_error_detail(
                code="RETRIEVAL_ERROR",
                message="检索服务异常，请稍后重试",
                stage="chat_retrieval",
                trace_id=trace.trace_id,
            ),
        ) from e

    results = hybrid_result if not hasattr(hybrid_result, "results") else hybrid_result.results

    # 2. Optional rerank
    if results and req.use_rerank and reranker.is_enabled:
        try:
            rerank_result = reranker.rerank(query=req.question, results=results, top_k=top_k, trace=trace)
            results = rerank_result.results
        except Exception as e:
            logger.warning(f"Reranking failed, using original order: {e}")

    # 3. Build citations
    citations = [
        Citation(
            source=_sanitize_source_path((r.metadata or {}).get("source_path", "unknown")),
            score=round(r.score, 4),
            text=(r.text or "")[:200],
        )
        for r in results
    ]

    # 4. LLM generation
    try:
        if results:
            messages = _build_prompt(req.question, results)
        else:
            # 知识库中没有相关内容，使用LLM基于自身知识回答
            messages = [
                Message(
                    role="system",
                    content=(
                        "你是一个知识库问答助手。当前知识库中没有找到与用户问题相关的内容，"  
                        "请基于你自己的知识回答用户问题。请明确告知用户你是基于自身知识回答的，"  
                        "而不是基于知识库内容。"
                    ),
                ),
                Message(
                    role="user",
                    content=f"问题：{req.question}",
                ),
            ]
        llm_response = llm.chat(messages)
        answer = llm_response.content
    except Exception as e:
        logger.exception("LLM generation failed")
        raise HTTPException(
            status_code=500,
            detail=_build_api_error_detail(
                code="LLM_SERVICE_ERROR",
                message=f"LLM 服务异常: {str(e)}"[:280],
                stage="chat_generation",
                trace_id=trace.trace_id,
            ),
        ) from e

    return ChatResponse(answer=answer, citations=citations)


@app.post("/lesson-plan", response_model=LessonPlanResponse)
async def generate_lesson_plan(req: LessonPlanRequest, request: Request):
    lesson_timeout_sec = float(os.environ.get("LESSON_PLAN_TIMEOUT_SEC", "85"))
    try:
        response_payload = await asyncio.wait_for(
            asyncio.to_thread(
                _generate_lesson_plan_internal,
                req,
                request,
            ),
            timeout=lesson_timeout_sec,
        )
    except asyncio.TimeoutError:
        logger.exception("Lesson orchestration timed out")
        raise HTTPException(
            status_code=504,
            detail=_build_api_error_detail(
                code="LESSON_TIMEOUT",
                message=f"生成超时（>{int(lesson_timeout_sec)}s），请重试或切换更快模型",
                stage="lesson_orchestration",
            ),
        )
    except Exception as e:
        logger.exception("Lesson orchestration failed")
        raise HTTPException(
            status_code=500,
            detail=_build_api_error_detail(
                code="LESSON_ORCHESTRATION_ERROR",
                message=f"教案编排失败: {str(e)}"[:280],
                stage="lesson_orchestration",
            ),
        ) from e
    return response_payload


@app.post("/lesson-plan/stream")
async def stream_lesson_plan(req: LessonPlanRequest, request: Request):
    lesson_timeout_sec = float(os.environ.get("LESSON_PLAN_TIMEOUT_SEC", "180"))

    async def event_stream():
        queue: asyncio.Queue[Optional[bytes]] = asyncio.Queue()
        loop = asyncio.get_running_loop()

        def emit(stage: str, payload: Dict[str, Any]) -> None:
            logger.info("lesson_plan.stream_progress topic=%s stage=%s", req.topic, stage)
            loop.call_soon_threadsafe(
                queue.put_nowait,
                _format_sse_event("progress", {"stage": stage, **payload}),
            )

        async def run_generation() -> None:
            try:
                result = await asyncio.wait_for(
                    asyncio.to_thread(
                        _generate_lesson_plan_internal,
                        req,
                        request,
                        emit,
                    ),
                    timeout=lesson_timeout_sec,
                )
                result_payload = result.model_dump() if hasattr(result, "model_dump") else result.dict()
                await queue.put(_format_sse_event("result", result_payload))
            except asyncio.TimeoutError:
                await queue.put(
                    _format_sse_event(
                        "error",
                        {
                            "code": "LESSON_TIMEOUT",
                            "message": f"生成超时（>{int(lesson_timeout_sec)}s），请重试或切换更快模型",
                            "stage": "lesson_orchestration",
                        },
                    )
                )
            except Exception as exc:
                logger.exception("Lesson streaming orchestration failed")
                await queue.put(
                    _format_sse_event(
                        "error",
                        {
                            "code": "LESSON_ORCHESTRATION_ERROR",
                            "message": f"教案编排失败: {str(exc)}"[:280],
                            "stage": "lesson_orchestration",
                        },
                    )
                )
            finally:
                await queue.put(None)

        worker = asyncio.create_task(run_generation())
        yield _format_sse_event(
            "progress",
            {
                "stage": "queued",
                "topic": req.topic,
                "template_category": req.template_category or "comprehensive",
            },
        )

        try:
            while True:
                item = await queue.get()
                if item is None:
                    break
                yield item
        finally:
            if not worker.done():
                worker.cancel()

    return StreamingResponse(
        event_stream(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",
        },
    )


@app.post("/lesson-plan/tasks", response_model=LessonTaskCreateResponse)
async def create_lesson_plan_task(req: LessonPlanRequest, request: Request):
    """Create an async lesson-generation task and return immediately."""
    task_storage = request.app.state.task_storage
    _task_cleanup(task_storage)
    task_id = uuid.uuid4().hex
    logger.info(
        "lesson_task.create task_id=%s topic=%s template_category=%s",
        task_id,
        req.topic,
        req.template_category,
    )
    _task_set(
        task_id,
        _storage=task_storage,
        status="queued",
        progress_stage="queued",
        created_at=time.time(),
    )

    async def _runner() -> None:
        timeout_sec = float(os.environ.get("LESSON_PLAN_TASK_TIMEOUT_SEC", "180"))
        _task_set(task_id, _storage=task_storage, status="running", progress_stage="running")
        logger.info(
            "lesson_task.run_start task_id=%s timeout_sec=%s topic=%s",
            task_id,
            timeout_sec,
            req.topic,
        )
        try:
            task_started = time.monotonic()
            result = await asyncio.wait_for(
                asyncio.to_thread(_generate_lesson_plan_internal, req, request),
                timeout=timeout_sec,
            )
            result_payload = result.model_dump() if hasattr(result, "model_dump") else result.dict()
            _task_set(
                task_id,
                _storage=task_storage,
                status="succeeded",
                progress_stage="done",
                result=result_payload,
                finished_at=time.time(),
            )
            logger.info(
                "lesson_task.run_success task_id=%s elapsed_ms=%.1f",
                task_id,
                (time.monotonic() - task_started) * 1000,
            )
        except asyncio.TimeoutError:
            _task_set(
                task_id,
                _storage=task_storage,
                status="failed",
                progress_stage="timeout",
                error={
                    "code": "LESSON_TIMEOUT",
                    "message": f"任务超时（>{int(timeout_sec)}s）",
                    "stage": "lesson_orchestration",
                },
                finished_at=time.time(),
            )
            logger.warning(
                "lesson_task.run_timeout task_id=%s timeout_sec=%s",
                task_id,
                timeout_sec,
            )
        except Exception as exc:
            logger.exception("Async lesson task failed")
            _task_set(
                task_id,
                _storage=task_storage,
                status="failed",
                progress_stage="failed",
                error={
                    "code": "LESSON_ORCHESTRATION_ERROR",
                    "message": str(exc)[:280],
                    "stage": "lesson_orchestration",
                },
                finished_at=time.time(),
            )

    asyncio.create_task(_runner())
    return LessonTaskCreateResponse(task_id=task_id, status="queued")


@app.get("/lesson-plan/tasks/{task_id}", response_model=LessonTaskStatusResponse)
async def get_lesson_plan_task(task_id: str, request: Request):
    payload = _task_get(task_id, request.app.state.task_storage)
    if not payload:
        logger.warning("lesson_task.poll_miss task_id=%s", task_id)
        raise HTTPException(status_code=404, detail="Task not found")

    logger.info(
        "lesson_task.poll task_id=%s status=%s progress_stage=%s",
        task_id,
        payload.get("status"),
        payload.get("progress_stage"),
    )

    result_obj = None
    if isinstance(payload.get("result"), dict):
        try:
            result_obj = LessonPlanResponse(**payload["result"])
        except Exception:
            result_obj = None

    return LessonTaskStatusResponse(
        task_id=task_id,
        status=str(payload.get("status") or "unknown"),
        progress_stage=payload.get("progress_stage"),
        result=result_obj,
        error=payload.get("error") if isinstance(payload.get("error"), dict) else None,
    )
