"""FastAPI Chat API for Modular RAG MCP Server.

Provides a browser-accessible chat interface backed by the full RAG pipeline:
HybridSearch (Dense + Sparse + RRF) → optional Rerank → LLM generation.

Endpoints:
    GET  /         - Serve the Chat UI HTML page
    GET  /health   - Service health and component status
    POST /chat     - Execute RAG pipeline and return answer + citations
"""

from __future__ import annotations

import os
import sys
from contextlib import asynccontextmanager
from pathlib import Path
from dataclasses import asdict, is_dataclass
import re
from typing import Any, Dict, List, Optional

from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, HTMLResponse, JSONResponse
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
from src.agents import LessonAgent
from src.ingestion.storage.bm25_indexer import BM25Indexer
from src.ingestion.storage.image_storage import ImageStorage
from src.libs.embedding.embedding_factory import EmbeddingFactory
from src.libs.llm.base_llm import Message
from src.libs.llm.llm_factory import LLMFactory
from src.libs.vector_store.vector_store_factory import VectorStoreFactory
from src.observability.logger import get_logger

logger = get_logger(__name__)

_STATIC_DIR = Path(__file__).parent / "static"


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


def _resolve_template_type(req: "LessonPlanRequest") -> Optional[str]:
    if req.template_type:
        return req.template_type
    if req.template_category == "guide":
        return "guide_master"
    if req.template_category == "comprehensive":
        return "comprehensive_master"
    return None


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


class HealthResponse(BaseModel):
    status: str
    components: dict


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


def _extract_image_resources(
    results: List[Any],
    image_storage: Optional[ImageStorage] = None,
    collection: Optional[str] = None,
    max_images: int = 6,
) -> List[LessonImageResource]:
    seen_ids = set()
    image_resources: List[LessonImageResource] = []
    doc_hashes = []

    for result in results:
        metadata = result.metadata or {}
        images = metadata.get("images", [])
        doc_hash = metadata.get("doc_hash")
        if doc_hash and doc_hash not in doc_hashes:
            doc_hashes.append(doc_hash)
        if not isinstance(images, list):
            continue

        caption_lookup = _extract_caption_lookup(metadata)
        source_path = metadata.get("source_path", "unknown")

        for image_info in images:
            if not isinstance(image_info, dict):
                continue

            image_id = image_info.get("id")
            image_path = image_info.get("path")
            if not image_id or not image_path or image_id in seen_ids:
                continue

            path_obj = Path(image_path)
            if not path_obj.is_absolute():
                path_obj = (_ROOT / image_path).resolve()

            if not path_obj.exists():
                continue

            seen_ids.add(image_id)
            image_resources.append(
                LessonImageResource(
                    image_id=str(image_id),
                    url=f"/lesson-plan-image/{image_id}",
                    source=str(source_path),
                    page=image_info.get("page") or metadata.get("page_num"),
                    caption=caption_lookup.get(str(image_id)),
                )
            )

            if len(image_resources) >= max_images:
                return image_resources

    if len(image_resources) >= max_images or image_storage is None:
        return image_resources

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

        for indexed in indexed_images:
            image_id = indexed.get("image_id")
            file_path = indexed.get("file_path")
            if not image_id or not file_path or image_id in seen_ids:
                continue

            path_obj = Path(file_path)
            if not path_obj.exists():
                continue

            seen_ids.add(image_id)
            image_resources.append(
                LessonImageResource(
                    image_id=str(image_id),
                    url=f"/lesson-plan-image/{image_id}",
                    source=str(path_obj),
                    page=indexed.get("page_num"),
                    caption=None,
                )
            )
            if len(image_resources) >= max_images:
                return image_resources

    return image_resources


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
            source_text = Path(image.source).name
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
        source_text = Path(image.source).name
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
        image for idx, image in enumerate(image_resources, start=1)
        if idx not in inserted_indices
    ]
    if remaining:
        appendix = _build_comprehensive_image_markdown(remaining)
        if appendix:
            content = "\n".join(lines).rstrip()
            content = f"{content}\n\n{appendix}"
            return _remove_dangling_image_references(content)

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

    app.state.settings = settings
    app.state.hybrid_search = hybrid_search
    app.state.reranker = reranker
    app.state.llm = llm
    app.state.image_storage = image_storage
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
        content={"error": "服务器内部错误，请稍后重试"},
    )


# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------

@app.get("/", response_class=HTMLResponse)
async def serve_ui():
    html_file = _STATIC_DIR / "index.html"
    if not html_file.exists():
        raise HTTPException(status_code=404, detail="UI file not found")
    return HTMLResponse(content=html_file.read_text(encoding="utf-8"))


@app.get("/chat.html", response_class=HTMLResponse)
async def serve_chat_ui():
    html_file = _STATIC_DIR / "chat.html"
    if not html_file.exists():
        raise HTTPException(status_code=404, detail="Chat UI file not found")
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
        raise HTTPException(status_code=500, detail="检索服务异常，请稍后重试") from e

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
            source=(r.metadata or {}).get("source_path", "unknown"),
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
        raise HTTPException(status_code=500, detail="LLM 服务异常，请稍后重试") from e

    return ChatResponse(answer=answer, citations=citations)


@app.post("/lesson-plan", response_model=LessonPlanResponse)
async def generate_lesson_plan(req: LessonPlanRequest, request: Request):
    state = request.app.state
    settings = state.settings
    hybrid_search = state.hybrid_search
    reranker = state.reranker
    
    # 动态切换模型
    if req.model:
        from src.libs.llm.llm_factory import LLMFactory
        from dataclasses import replace as dc_replace
        from src.core.settings import LLMSettings
        
        # 创建新的LLM配置
        new_llm_config = LLMSettings(
            provider=settings.llm.provider,
            model=req.model,
            api_key=settings.llm.api_key,
            base_url=settings.llm.base_url,
            temperature=settings.llm.temperature,
            max_tokens=settings.llm.max_tokens,
        )
        new_settings = dc_replace(settings, llm=new_llm_config)
        llm = LLMFactory.create(new_settings)
        logger.info(f"Using custom model: {req.model}")
    else:
        llm = state.llm

    top_k = 15  # 教案需要更多的上下文信息
    trace = TraceContext(trace_type="lesson_plan")

    # 添加元数据
    trace.metadata["topic"] = req.topic
    trace.metadata["collection"] = req.collection
    trace.metadata["model"] = req.model or settings.llm.model
    trace.metadata["include_background"] = req.include_background
    trace.metadata["include_facts"] = req.include_facts
    trace.metadata["include_examples"] = req.include_examples
    trace.metadata["template_category"] = req.template_category
    trace.metadata["template_type"] = req.template_type
    if req.grade_level:
        trace.metadata["grade_level"] = req.grade_level
    if req.learning_style:
        trace.metadata["learning_style"] = req.learning_style

    # 1. 扩展查询以获取更多相关信息
    expanded_query = f"{req.topic} 背景 历史 原理 应用 实验 教学"
    
    # 2. Hybrid search
    try:
        hybrid_result = hybrid_search.search(
            query=expanded_query,
            top_k=top_k,
            filters=None,
            trace=trace,
        )
    except Exception as e:
        logger.exception("HybridSearch failed for lesson plan")
        raise HTTPException(status_code=500, detail="检索服务异常，请稍后重试") from e

    results = hybrid_result if not hasattr(hybrid_result, "results") else hybrid_result.results

    # 3. Optional rerank
    if results and reranker.is_enabled:
        try:
            rerank_result = reranker.rerank(query=expanded_query, results=results, top_k=top_k, trace=trace)
            results = rerank_result.results
        except Exception as e:
            logger.warning(f"Reranking failed for lesson plan, using original order: {e}")

    # 4. Build citations
    citations = [
        Citation(
            source=(r.metadata or {}).get("source_path", "unknown"),
            score=round(r.score, 4),
            text=(r.text or "")[:200],
        )
        for r in results
    ]
    image_resources = _extract_image_resources(
        results,
        image_storage=request.app.state.image_storage,
        collection=req.collection,
    )

    # 5. Agent generation for lesson plan
    try:
        template_manager = TemplateManager()
        resolved_template_type = _resolve_template_type(req)
        agent = LessonAgent(
            llm=llm,
            template_manager=template_manager,
            trace=trace,
            request=req,
            resolved_template_type=resolved_template_type,
            build_default_prompt=_build_lesson_plan_prompt,
            build_fallback_prompt=_build_lesson_plan_prompt_fallback,
            integrate_images=_integrate_images_into_markdown,
        )
        agent_state = agent.run(
            topic=req.topic,
            results=results,
            image_resources=image_resources,
            citations=citations,
        )
        lesson_plan_content = agent_state.final_content or agent_state.draft_content or ""
        subject = agent_state.subject
    except Exception as e:
        logger.exception("LLM generation failed for lesson plan")
        raise HTTPException(status_code=500, detail="LLM 服务异常，请稍后重试") from e

    trace.record_stage("lesson_agent_complete", {
        "model": req.model or settings.llm.model,
        "has_context": len(results) > 0,
        "image_count": len(image_resources),
        "subject": subject,
        "template_type": resolved_template_type,
        "review_must_fix_count": len(agent_state.review_report.must_fix) if agent_state.review_report else 0,
    })

    # 保存trace
    trace.finish()
    from src.core.trace.trace_collector import TraceCollector
    TraceCollector().collect(trace)

    return LessonPlanResponse(
        topic=req.topic,
        subject=subject,
        lesson_content=lesson_plan_content,
        additional_resources=citations,
        image_resources=image_resources,
        review_report=_to_review_report_response(agent_state.review_report),
    )
