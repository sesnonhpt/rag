"""Template management router - Phase 1 & 2: File listing and editing."""

from __future__ import annotations

import asyncio
import contextlib
import html
import json
import os
import tempfile
from pathlib import Path
from typing import List, Optional, Dict, Any, Tuple

from fastapi import APIRouter, HTTPException, Request, UploadFile, File, Form
from fastapi.responses import FileResponse, Response, StreamingResponse
from pydantic import BaseModel, Field

from app.core.paths import APP_ROOT
from app.core.runtime_helpers import format_sse_event
from app.services.template_database import TemplateDatabase
from app.services.file_parser_service import FileParserService
from app.services.template_export_service import TemplateExportService
from app.services.template_metadata_service import TemplateMetadataService
from app.services.teaching_thought_extractor import TeachingThoughtExtractor, PhysicsTeachingThoughtExtractor
from app.services.lesson_analyzer import LessonAnalyzer
from src.libs.llm.openai_llm import OpenAILLMError
from src.libs.llm.base_llm import Message
from src.observability.logger import get_logger

router = APIRouter()
logger = get_logger(__name__)

# Template storage directory
TEMPLATES_DIR = APP_ROOT.parent / "data" / "templates"
DB_PATH = APP_ROOT.parent / "data" / "db" / "template_index.db"

# Initialize services
template_db = TemplateDatabase(str(DB_PATH))
file_parser = FileParserService()
export_service = TemplateExportService()
metadata_service = TemplateMetadataService(TEMPLATES_DIR)


class TemplateFileInfo(BaseModel):
    """Template file information."""
    filename: str
    size_bytes: int
    size_display: str
    modified_at: str
    file_type: str
    desc: Optional[str] = None  # Metadata description
    keywords: Optional[List[str]] = None  # Metadata keywords
    relevance_score: Optional[float] = None  # Search relevance score


class TemplateListResponse(BaseModel):
    """Response for template list."""
    templates: List[TemplateFileInfo]
    total: int
    directory: str


class TemplateContentRequest(BaseModel):
    """Request to save template content."""
    content_html: str = Field(..., min_length=1)
    create_version: bool = True
    change_summary: Optional[str] = None


class TemplateContentResponse(BaseModel):
    """Response with template content."""
    template_id: str
    filename: str
    content_html: str
    version_id: Optional[str] = None
    metadata: Dict[str, Any]


class AIModifyRequest(BaseModel):
    """Request for AI content modification."""
    original_text: str = Field(..., min_length=1, max_length=5000)
    instruction: str = Field(..., min_length=1, max_length=500)


class AIModifyResponse(BaseModel):
    """Response for AI modification."""
    modified_text: str
    processing_time_ms: float


class ExportRequest(BaseModel):
    """Request for template export."""
    format: str = Field(..., pattern="^(docx|pdf|md)$")


class ExportResponse(BaseModel):
    """Response for export."""
    success: bool
    download_url: str
    format: str
    file_size: int


class VersionInfo(BaseModel):
    """Version information."""
    version_id: str
    created_at: str
    change_summary: Optional[str] = None


class VersionListResponse(BaseModel):
    """Response with version list."""
    versions: List[VersionInfo]
    total: int


class CourseDraftResponse(BaseModel):
    """Response for精品课/试点生成结果."""
    source_kind: str
    source_label: str
    subject: str
    grade: Optional[str] = None
    topic: str
    platform: Optional[str] = None
    teacher_name: Optional[str] = None
    duration_minutes: int = 45
    source_summary: str
    transcript_text: str
    draft_markdown: str


def _format_file_size(size_bytes: int) -> str:
    """Format file size in human-readable format."""
    for unit in ['B', 'KB', 'MB', 'GB']:
        if size_bytes < 1024.0:
            return f"{size_bytes:.1f} {unit}"
        size_bytes /= 1024.0
    return f"{size_bytes:.1f} TB"


def _get_file_type(filename: str) -> str:
    """Get file type from extension."""
    ext = Path(filename).suffix.lower()
    type_map = {
        '.doc': 'Word 文档 (.doc)',
        '.docx': 'Word 文档 (.docx)',
        '.pdf': 'PDF 文档',
        '.txt': '文本文件',
        '.md': 'Markdown 文档',
    }
    return type_map.get(ext, f'未知类型 ({ext})')


def _html_to_text(content_html: str) -> str:
    """Convert HTML content to plain text for LLM prompts."""
    try:
        from bs4 import BeautifulSoup

        soup = BeautifulSoup(content_html or "", "html.parser")
        return "\n".join(line.strip() for line in soup.get_text("\n").splitlines() if line.strip())
    except Exception:
        text = content_html or ""
        return text.replace("<br>", "\n").replace("<br/>", "\n").replace("<br />", "\n")


def _normalize_text_snippet(text: str, max_chars: int = 18000) -> str:
    """Keep long source text within a prompt-friendly size."""
    cleaned = "\n".join(line.strip() for line in str(text or "").splitlines() if line.strip())
    if len(cleaned) <= max_chars:
        return cleaned

    head = cleaned[: int(max_chars * 0.55)]
    middle_start = max(0, len(cleaned) // 2 - int(max_chars * 0.12))
    middle_end = min(len(cleaned), middle_start + int(max_chars * 0.24))
    tail = cleaned[-int(max_chars * 0.21):]
    return "\n".join([
        head,
        "\n[中间内容已压缩，保留代表性片段]\n",
        cleaned[middle_start:middle_end],
        "\n[结尾内容保留]\n",
        tail,
    ])


def _safe_markdown_to_html(markdown_text: str) -> str:
    """Minimal markdown-to-HTML conversion for editor preload when needed."""
    blocks = []
    for raw_line in str(markdown_text or "").splitlines():
        line = raw_line.strip()
        if not line:
            continue
        if line.startswith("### "):
            blocks.append(f"<h3>{html.escape(line[4:])}</h3>")
        elif line.startswith("## "):
            blocks.append(f"<h2>{html.escape(line[3:])}</h2>")
        elif line.startswith("# "):
            blocks.append(f"<h1>{html.escape(line[2:])}</h1>")
        elif line.startswith("- "):
            blocks.append(f"<p>• {html.escape(line[2:])}</p>")
        elif line[:3].isdigit() and line[1:3] == ". ":
            blocks.append(f"<p>{html.escape(line)}</p>")
        else:
            blocks.append(f"<p>{html.escape(line)}</p>")
    return "\n".join(blocks)


def _transcribe_audio_with_openai_compatible(
    *,
    llm: Any,
    audio_bytes: bytes,
    filename: str,
    content_type: Optional[str],
) -> str:
    """Transcribe audio via OpenAI-compatible audio endpoint when available."""
    import httpx

    api_key = os.environ.get("OPENAI_AUDIO_API_KEY") or getattr(llm, "api_key", None)
    if not api_key:
        raise RuntimeError("当前环境没有可用的语音转写密钥，请先上传文字稿或配置音频转写服务。")

    base_url = os.environ.get("OPENAI_AUDIO_BASE_URL") or getattr(llm, "base_url", None)
    if not base_url:
        raise RuntimeError("当前环境没有可用的语音转写地址，请先上传文字稿。")

    if getattr(llm, "_use_azure_auth", False) and not os.environ.get("OPENAI_AUDIO_BASE_URL"):
        raise RuntimeError("当前模型配置暂不支持直接语音转写，请先上传文字稿，或单独配置 OPENAI_AUDIO_BASE_URL。")

    endpoint = f"{str(base_url).rstrip('/')}/audio/transcriptions"
    model_candidates = [
        os.environ.get("OPENAI_AUDIO_TRANSCRIBE_MODEL"),
        "gpt-4o-mini-transcribe",
        "whisper-1",
    ]

    last_error: Optional[str] = None
    for model_name in [m for m in model_candidates if m]:
        files = {
            "file": (filename, audio_bytes, content_type or "application/octet-stream"),
        }
        data = {
            "model": model_name,
            "response_format": "text",
        }
        try:
            with httpx.Client(timeout=240.0) as client:
                response = client.post(
                    endpoint,
                    headers={"Authorization": f"Bearer {api_key}"},
                    data=data,
                    files=files,
                )
            if response.status_code == 200:
                transcript = response.text.strip()
                if transcript:
                    return transcript
                last_error = "转写结果为空"
                continue
            last_error = f"HTTP {response.status_code}: {response.text[:180]}"
        except Exception as exc:
            last_error = str(exc)

    raise RuntimeError(f"语音转写失败，请先上传文字稿或稍后重试。{last_error or ''}".strip())


def _extract_course_source(
    *,
    llm: Any,
    source_text: Optional[str],
    source_file: Optional[UploadFile],
) -> Tuple[str, str]:
    """Extract course text from pasted text, document, or audio."""
    pasted = str(source_text or "").strip()
    if pasted:
        return pasted, "text"

    if source_file is None:
        return "", "empty"

    suffix = Path(source_file.filename or "upload").suffix.lower()
    raw = source_file.file.read()
    if not raw:
        return "", "empty"

    if suffix in {".txt", ".md"}:
        return raw.decode("utf-8", errors="ignore"), "document"

    if suffix in {".docx", ".doc", ".pdf"}:
        with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
            tmp.write(raw)
            tmp_path = Path(tmp.name)
        try:
            parse_result = file_parser.parse_file(tmp_path)
            return _html_to_text(parse_result.html_content), "document"
        finally:
            with contextlib.suppress(FileNotFoundError):
                tmp_path.unlink()

    if suffix in {".mp3", ".wav", ".m4a", ".mp4", ".mpeg", ".mpga"}:
        transcript = _transcribe_audio_with_openai_compatible(
            llm=llm,
            audio_bytes=raw,
            filename=source_file.filename or "course-audio",
            content_type=source_file.content_type,
        )
        return transcript, "audio"

    raise RuntimeError(f"暂不支持该文件类型：{suffix}。目前支持 txt/md/doc/docx/pdf/mp3/wav/m4a/mp4。")


def _build_course_summary(
    *,
    llm: Any,
    platform: str,
    teacher_name: str,
    subject: str,
    grade: str,
    topic: str,
    duration_minutes: int,
    notes: str,
    source_text: str,
    prefer_physics_pilot: bool,
) -> str:
    prompt = f"""你是一位资深教研员，正在帮一线老师拆解精品课。

请阅读下面的课程文字稿或转写稿，提炼出老师真正需要的课堂信息。输出要求：
1. 用老师能看懂的话写，不要使用“向量检索、重排、token”等技术词。
2. 重点提炼：这节课的主线、关键教学环节、值得借鉴的提问、活动设计、学生易错点。
3. 如果是理科尤其物理，请更强调概念建构、实验观察、规律归纳、易错辨析。
4. 输出使用 Markdown，包含这 5 个二级标题：
## 课程主线
## 可借鉴的教学环节
## 值得保留的老师表达
## 学生可能卡住的地方
## 本课适合迁移到哪里

课程信息：
- 平台：{platform or '未注明'}
- 主讲老师：{teacher_name or '未注明'}
- 学科：{subject or '未注明'}
- 年级：{grade or '未注明'}
- 知识点：{topic or '未注明'}
- 课时：{duration_minutes} 分钟
- 补充要求：{notes or '无'}
- 是否物理试点：{'是' if prefer_physics_pilot else '否'}

课程原文如下：
{_normalize_text_snippet(source_text, 20000)}
"""
    messages = [
        Message(role="system", content="你擅长把精品课拆解成老师可直接使用的备课要点。"),
        Message(role="user", content=prompt),
    ]
    return llm.chat(messages).content.strip()


def _build_course_draft(
    *,
    llm: Any,
    platform: str,
    teacher_name: str,
    subject: str,
    grade: str,
    topic: str,
    duration_minutes: int,
    notes: str,
    course_summary: str,
    source_text: str,
    prefer_physics_pilot: bool,
) -> str:
    prompt = f"""你是一位教案设计专家，请基于精品课拆解结果，生成一份老师可以继续修改的“第一版教学设计”。

输出要求：
1. 目标用户是一线老师，语言要顺、能拿来备课，不要写成论文。
2. 同时考虑老师备课清晰、学生易懂。
3. 如果学科是理科，尤其物理，要强调：情境导入、实验/现象观察、概念形成、规律提炼、练习巩固、易错提醒。
4. 如果没有完整原文，也要根据学科和知识点先生成一版可用试点稿。
5. 输出 Markdown，必须包含这些标题：
# {topic or '第一版教学设计'}
## 一、教材与学情判断
## 二、教学目标
## 三、教学重点与难点
## 四、45分钟课堂流程
## 五、关键提问与师生活动
## 六、板书与练习建议
## 七、课后反思提示

附加要求：
- 每个课堂环节尽量写出时间分配。
- 关键提问要像老师在课堂里会说的话。
- 如果平台、老师信息有参考价值，可以自然融入“设计思路”里，不必生硬提及。
- 不要暴露任何技术处理过程。

课程信息：
- 平台：{platform or '未注明'}
- 主讲老师：{teacher_name or '未注明'}
- 学科：{subject or '未注明'}
- 年级：{grade or '未注明'}
- 知识点：{topic or '未注明'}
- 时长：{duration_minutes} 分钟
- 补充要求：{notes or '无'}
- 是否物理试点：{'是' if prefer_physics_pilot else '否'}

精品课拆解结果：
{course_summary or '暂无拆解结果'}

可参考的原始课程片段：
{_normalize_text_snippet(source_text, 12000) if source_text else '暂无原始课程文字稿，请按试点模式生成。'}
"""
    messages = [
        Message(role="system", content="你擅长把精品课内容整理成老师能直接继续打磨的第一版教学设计。"),
        Message(role="user", content=prompt),
    ]
    return llm.chat(messages).content.strip()


@router.get("/templates/list", response_model=TemplateListResponse)
async def list_templates(
    search: Optional[str] = None,
    use_metadata: bool = True
):
    """
    List all template files in the templates directory (including subdirectories).
    
    Args:
        search: Optional search query to filter filenames
        use_metadata: Use metadata search (desc + keywords) if True
    
    Returns:
        TemplateListResponse with file information
    """
    try:
        # Ensure templates directory exists
        TEMPLATES_DIR.mkdir(parents=True, exist_ok=True)
        
        files = []
        
        if search and use_metadata:
            # Layer 2: Metadata search
            logger.info(f"Searching with metadata: {search}")
            search_results = metadata_service.search(
                search,
                search_filename=True,
                search_metadata=True
            )
            
            # Build file info for search results
            for filename, score in search_results:
                file_path = TEMPLATES_DIR / filename
                if not file_path.exists():
                    continue
                
                stat = file_path.stat()
                metadata = metadata_service.get_metadata(filename)
                
                files.append(TemplateFileInfo(
                    filename=filename,
                    size_bytes=stat.st_size,
                    size_display=_format_file_size(stat.st_size),
                    modified_at=str(stat.st_mtime),
                    file_type=_get_file_type(file_path.name),
                    desc=metadata.desc if metadata else None,
                    keywords=metadata.keywords if metadata else None,
                    relevance_score=score
                ))
        else:
            # Layer 1: Simple filename search (or list all)
            for item in TEMPLATES_DIR.rglob('*'):
                if not item.is_file():
                    continue
                
                # Skip hidden files and README
                if item.name.startswith('.') or item.name == 'README.md':
                    continue
                
                # Apply simple filename filter if provided
                if search and search.lower() not in item.name.lower():
                    continue
                
                stat = item.stat()
                relative_path = item.relative_to(TEMPLATES_DIR)
                filename = str(relative_path)
                
                # Try to get metadata if available
                metadata = metadata_service.get_metadata(filename)
                
                files.append(TemplateFileInfo(
                    filename=filename,
                    size_bytes=stat.st_size,
                    size_display=_format_file_size(stat.st_size),
                    modified_at=str(stat.st_mtime),
                    file_type=_get_file_type(item.name),
                    desc=metadata.desc if metadata else None,
                    keywords=metadata.keywords if metadata else None
                ))
            
            # Sort by modified time (newest first) if no search
            if not search:
                files.sort(key=lambda x: float(x.modified_at), reverse=True)
        
        logger.info(f"Listed {len(files)} template files from {TEMPLATES_DIR}")
        
        return TemplateListResponse(
            templates=files,
            total=len(files),
            directory=str(TEMPLATES_DIR)
        )
    
    except Exception as e:
        logger.exception("Failed to list templates")
        raise HTTPException(
            status_code=500,
            detail=f"获取模板列表失败: {str(e)}"
        )


@router.get("/templates/download/{filename:path}")
async def download_template(filename: str):
    """
    Download a template file (supports subdirectories).
    
    Args:
        filename: Relative path to the file (e.g., "五数下导学案/1.1 等式与方程.docx")
    
    Returns:
        FileResponse with the requested file
    """
    try:
        # Security: prevent path traversal attacks
        # Normalize the path and ensure it doesn't escape templates directory
        safe_path = Path(filename).as_posix()
        if '..' in safe_path or safe_path.startswith('/'):
            raise HTTPException(
                status_code=403,
                detail="访问被拒绝：非法路径"
            )
        
        file_path = TEMPLATES_DIR / safe_path
        
        # Check if file exists
        if not file_path.exists() or not file_path.is_file():
            raise HTTPException(
                status_code=404,
                detail=f"文件不存在: {filename}"
            )
        
        # Check if file is within templates directory (security)
        if not str(file_path.resolve()).startswith(str(TEMPLATES_DIR.resolve())):
            raise HTTPException(
                status_code=403,
                detail="访问被拒绝"
            )
        
        logger.info(f"Downloading template file: {filename}")
        
        # Use just the filename (not path) for download
        download_filename = Path(filename).name
        
        return FileResponse(
            path=file_path,
            filename=download_filename,
            media_type='application/octet-stream'
        )
    
    except HTTPException:
        raise
    except Exception as e:
        logger.exception(f"Failed to download template: {filename}")
        raise HTTPException(
            status_code=500,
            detail=f"下载文件失败: {str(e)}"
        )


@router.get("/templates/{filename:path}/content", response_model=TemplateContentResponse)
async def get_template_content(filename: str):
    """
    Get template content for editing (supports subdirectories).
    Parse the file and return HTML content.
    
    Args:
        filename: Relative path to the file (e.g., "五数下导学案/1.1 等式与方程.docx")
    
    Returns:
        TemplateContentResponse with HTML content
    """
    try:
        # Security: prevent path traversal attacks
        safe_path = Path(filename).as_posix()
        if '..' in safe_path or safe_path.startswith('/'):
            raise HTTPException(
                status_code=403,
                detail="访问被拒绝：非法路径"
            )
        
        file_path = TEMPLATES_DIR / safe_path
        
        # Check if file exists
        if not file_path.exists() or not file_path.is_file():
            raise HTTPException(
                status_code=404,
                detail=f"文件不存在: {filename}"
            )
        
        # Check if file is within templates directory
        if not str(file_path.resolve()).startswith(str(TEMPLATES_DIR.resolve())):
            raise HTTPException(
                status_code=403,
                detail="访问被拒绝"
            )
        
        # Parse file to HTML
        parse_result = file_parser.parse_file(file_path)
        
        logger.info(f"Retrieved template content: {filename}")
        
        return TemplateContentResponse(
            template_id=filename,
            filename=filename,
            content_html=parse_result.html_content,
            version_id=None,
            metadata=parse_result.metadata
        )
    
    except HTTPException:
        raise
    except Exception as e:
        logger.exception(f"Failed to get template content: {filename}")
        raise HTTPException(
            status_code=500,
            detail=f"获取模板内容失败: {str(e)}"
        )


@router.put("/templates/{filename:path}/content")
async def save_template_content(filename: str, request: TemplateContentRequest):
    """
    Save edited template content (simplified - no versioning, supports subdirectories).
    Content is stored in memory for export only.
    
    Args:
        filename: Relative path to the file (e.g., "五数下导学案/1.1 等式与方程.docx")
        request: Content to save
    
    Returns:
        Success response
    """
    try:
        # Security: prevent path traversal attacks
        safe_path = Path(filename).as_posix()
        if '..' in safe_path or safe_path.startswith('/'):
            raise HTTPException(
                status_code=403,
                detail="访问被拒绝：非法路径"
            )
        
        # Store content in a simple in-memory cache for export
        # Use a hash of the full path to avoid directory issues in cache
        import hashlib
        cache_key = hashlib.md5(filename.encode()).hexdigest()
        
        cache_dir = Path(tempfile.gettempdir()) / "template_cache"
        cache_dir.mkdir(exist_ok=True)
        cache_file = cache_dir / f"{cache_key}.html"
        cache_file.write_text(request.content_html, encoding='utf-8')
        
        # Also store the original filename mapping
        mapping_file = cache_dir / f"{cache_key}.mapping"
        mapping_file.write_text(filename, encoding='utf-8')
        
        logger.info(f"Saved template content to cache: {filename}")
        
        return {
            "success": True,
            "template_id": filename,
            "message": "内容已缓存，可以导出"
        }
    
    except Exception as e:
        logger.exception(f"Failed to save template content: {filename}")
        raise HTTPException(
            status_code=500,
            detail=f"保存失败: {str(e)}"
        )


@router.post("/templates/ai-modify", response_model=AIModifyResponse)
async def ai_modify_content(request: AIModifyRequest, req: Request):
    """
    AI-assisted content modification.
    
    Args:
        request: Original text and modification instruction
        req: FastAPI request (to access LLM)
    
    Returns:
        AIModifyResponse with modified text
    """
    import time as time_module
    from src.libs.llm.base_llm import Message
    
    try:
        start_time = time_module.time()
        
        # Get LLM from app state
        llm = req.app.state.llm
        
        # Build prompt
        prompt = f"""你是一位经验丰富的教师。请根据以下指令修改文本内容。

原始文本：
{request.original_text}

修改指令：
{request.instruction}

请直接输出修改后的文本，不要添加额外说明。"""
        
        messages = [
            Message(role="system", content="你是一位专业的教学内容编辑助手。"),
            Message(role="user", content=prompt)
        ]
        
        # Call LLM
        response = llm.chat(messages)
        modified_text = response.content.strip()
        
        processing_time = (time_module.time() - start_time) * 1000
        
        logger.info(f"AI modified content, time: {processing_time:.1f}ms")
        
        return AIModifyResponse(
            modified_text=modified_text,
            processing_time_ms=processing_time
        )
    
    except OpenAILLMError as e:
        logger.exception("AI modification failed")
        message = str(e)
        if "HTTP 429" in message or "rate limited" in message.lower():
            raise HTTPException(
                status_code=429,
                detail="AI 修改暂时过载，请稍后重试或切换可用模型",
            )
        if "HTTP 404" in message or "no longer available" in message.lower():
            raise HTTPException(
                status_code=503,
                detail="AI 修改当前使用的模型已不可用，请更新模型配置后重试",
            )
        raise HTTPException(
            status_code=502,
            detail=f"AI 修改失败: {message}",
        )
    except Exception as e:
        logger.exception("AI modification failed")
        raise HTTPException(
            status_code=500,
            detail=f"AI 修改失败: {str(e)}"
        )


@router.post("/templates/extract-teaching-thoughts")
async def extract_teaching_thoughts(
    req: Request,
    content: str = Form(""),
    subject: str = Form("物理"),
    topic: str = Form(""),
    grade: str = Form(""),
):
    """
    从导学案内容中提取备课思路
    
    Args:
        content: 导学案文本内容
        subject: 学科
        topic: 课题
        grade: 年级
    
    Returns:
        5 个维度的备课思路
    """
    try:
        if not content.strip():
            raise HTTPException(
                status_code=400,
                detail="请提供导学案内容"
            )
        
        logger.info(f"extract_teaching_thoughts.start subject={subject} topic={topic}")
        
        # 使用 LLM 提取备课思路
        llm = req.app.state.llm
        
        # 根据学科选择合适的提取器
        if subject == "物理":
            thought_extractor = PhysicsTeachingThoughtExtractor(llm)
        else:
            thought_extractor = TeachingThoughtExtractor(llm)
        
        thoughts = await thought_extractor.extract_thoughts(
            course_text=content,
            subject=subject,
            topic=topic or "未指定课题",
            grade=grade,
            teacher_name=""
        )
        
        logger.info(f"extract_teaching_thoughts.success dimensions={len(thoughts)}")
        
        return {
            "success": True,
            "thoughts": [t.to_dict() for t in thoughts]
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.exception("extract_teaching_thoughts.failed")
        raise HTTPException(
            status_code=500,
            detail=f"提取备课思路失败: {str(e)}"
        )


@router.post("/templates/co-create-analyze")
async def analyze_lesson_content(
    req: Request,
    content: str = Form(""),
    file: UploadFile | None = File(default=None),
):
    """
    分析现有导学案内容
    
    Args:
        content: 粘贴的文本内容
        file: 上传的文件（doc/docx/pdf/txt）
    
    Returns:
        分析结果（JSON）
    """
    try:
        # 1. 提取文本
        if file:
            logger.info(f"co_create_analyze.file_upload filename={file.filename}")
            # 使用现有的文件解析服务
            with tempfile.NamedTemporaryFile(delete=False, suffix=Path(file.filename or "upload").suffix) as tmp:
                tmp.write(await file.read())
                tmp_path = Path(tmp.name)
            
            try:
                parse_result = file_parser.parse_file(tmp_path)
                text = _html_to_text(parse_result.html_content)
            finally:
                with contextlib.suppress(FileNotFoundError):
                    tmp_path.unlink()
        else:
            text = content
        
        if not text.strip():
            raise HTTPException(
                status_code=400,
                detail="请提供导学案内容（上传文件或粘贴文本）"
            )
        
        logger.info(f"co_create_analyze.analyzing text_length={len(text)}")
        
        # 2. 分析内容
        llm = req.app.state.llm
        analyzer = LessonAnalyzer(llm)
        analysis = await analyzer.analyze(text)
        
        logger.info(
            f"co_create_analyze.success topic={analysis.topic} subject={analysis.subject}"
        )
        
        return {
            "success": True,
            "analysis": analysis.to_dict()
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.exception("co_create_analyze.failed")
        raise HTTPException(
            status_code=500,
            detail=f"分析失败: {str(e)}"
        )


@router.post("/templates/course-to-draft/stream")
async def stream_course_to_draft(
    req: Request,
    platform: str = Form(""),
    teacher_name: str = Form(""),
    subject: str = Form("物理"),
    grade: str = Form(""),
    topic: str = Form(""),
    duration_minutes: int = Form(45),
    notes: str = Form(""),
    source_text: str = Form(""),
    prefer_physics_pilot: bool = Form(False),
    source_file: UploadFile | None = File(default=None),
):
    """
    Stream the process of turning精品课文字/语音 into a first draft teaching design.
    """

    async def event_stream():
        queue: asyncio.Queue[Optional[bytes]] = asyncio.Queue()
        loop = asyncio.get_running_loop()

        def emit(stage: str, payload: Dict[str, Any]) -> None:
            loop.call_soon_threadsafe(
                queue.put_nowait,
                format_sse_event("progress", {"stage": stage, **payload}),
            )

        async def run_generation() -> None:
            try:
                llm = req.app.state.llm

                emit("queued", {
                    "title": "开始接手精品课整理任务",
                    "detail": "先接收平台、老师、学科和课堂材料。",
                })

                emit("parsing_source", {
                    "title": "正在读取文字稿或语音稿",
                    "detail": "把上传内容整理成可分析的课堂原文。",
                })
                source_raw_text, source_kind = await asyncio.to_thread(
                    _extract_course_source,
                    llm=llm,
                    source_text=source_text,
                    source_file=source_file,
                )

                source_label_map = {
                    "text": "文字稿",
                    "document": "文档",
                    "audio": "语音转写",
                    "empty": "试点模式",
                }
                source_label = source_label_map.get(source_kind, "课堂材料")

                if not source_raw_text.strip() and not str(topic or "").strip():
                    raise HTTPException(status_code=400, detail="请至少提供课程文字/语音内容，或填写试点知识点。")

                emit("understanding_course", {
                    "title": "正在拆解这节课的主线",
                    "detail": "重点提炼教学环节、老师提问方式和学生容易卡住的地方。",
                    "source_kind": source_label,
                })
                summary = await asyncio.to_thread(
                    _build_course_summary,
                    llm=llm,
                    platform=platform,
                    teacher_name=teacher_name,
                    subject=subject,
                    grade=grade,
                    topic=topic,
                    duration_minutes=duration_minutes,
                    notes=notes,
                    source_text=source_raw_text or f"{subject} {topic}",
                    prefer_physics_pilot=prefer_physics_pilot,
                )

                # 新增：提取备课思路（5 个维度）
                emit("extracting_thoughts", {
                    "title": "正在理解这节课的设计思路",
                    "detail": "从名师课堂中提取 5 个核心备课维度：这节课要讲什么、怎么导入、易错点、课堂活动、检验理解。",
                })
                
                # 根据学科选择合适的提取器
                if subject == "物理":
                    thought_extractor = PhysicsTeachingThoughtExtractor(llm)
                else:
                    thought_extractor = TeachingThoughtExtractor(llm)
                
                thoughts = await thought_extractor.extract_thoughts(
                    course_text=source_raw_text or f"{subject} {topic} 试点课",
                    subject=subject,
                    topic=topic or "学科试点课",
                    grade=grade,
                    teacher_name=teacher_name
                )
                
                # 发送备课思路给前端
                await queue.put(format_sse_event("thoughts", {
                    "thoughts": [t.to_dict() for t in thoughts]
                }))

                emit("building_design", {
                    "title": "正在生成第一版教学设计",
                    "detail": "把 45 分钟课堂内容压缩成老师可继续修改的课堂设计初稿。",
                })
                draft_markdown = await asyncio.to_thread(
                    _build_course_draft,
                    llm=llm,
                    platform=platform,
                    teacher_name=teacher_name,
                    subject=subject,
                    grade=grade,
                    topic=topic or "学科试点课",
                    duration_minutes=duration_minutes,
                    notes=notes,
                    course_summary=summary,
                    source_text=source_raw_text,
                    prefer_physics_pilot=prefer_physics_pilot,
                )

                emit("teacher_rewrite", {
                    "title": "正在把内容改成老师好用的话",
                    "detail": "过滤技术味和论文腔，让课堂表达更顺。",
                })
                polished_markdown = await asyncio.to_thread(
                    lambda: llm.chat([
                        Message(role="system", content="你负责把教学设计改成一线老师更容易使用的课堂表达。"),
                        Message(role="user", content=f"""请只对下面这份教学设计做语言层面的优化：
1. 保留原有结构和信息。
2. 改掉过于书面、过于技术化、过于生硬的表达。
3. 让老师备课更清晰，学生更容易听懂。
4. 直接输出修改后的 Markdown。

原稿：
{draft_markdown}
"""),
                    ]).content.strip()
                )

                payload = CourseDraftResponse(
                    source_kind=source_kind,
                    source_label=source_label,
                    subject=subject or "未注明",
                    grade=grade or None,
                    topic=topic or "学科试点课",
                    platform=platform or None,
                    teacher_name=teacher_name or None,
                    duration_minutes=duration_minutes,
                    source_summary=summary,
                    transcript_text=source_raw_text,
                    draft_markdown=polished_markdown,
                ).model_dump()
                await queue.put(format_sse_event("result", payload))
            except HTTPException as exc:
                await queue.put(format_sse_event("error", {
                    "code": "COURSE_DRAFT_ERROR",
                    "message": str(exc.detail),
                    "stage": "course_to_draft",
                }))
            except OpenAILLMError as exc:
                await queue.put(format_sse_event("error", {
                    "code": "COURSE_DRAFT_LLM_ERROR",
                    "message": f"生成教学设计失败：{str(exc)[:260]}",
                    "stage": "course_to_draft",
                }))
            except Exception as exc:
                logger.exception("Failed to generate course draft")
                await queue.put(format_sse_event("error", {
                    "code": "COURSE_DRAFT_ERROR",
                    "message": f"精品课整理失败：{str(exc)[:260]}",
                    "stage": "course_to_draft",
                }))
            finally:
                await queue.put(None)

        worker = asyncio.create_task(run_generation())
        yield format_sse_event("progress", {
            "stage": "queued",
            "title": "开始接手精品课整理任务",
            "detail": "先接收平台、老师、学科和课堂材料。",
        })

        try:
            while True:
                item = await queue.get()
                if item is None:
                    break
                yield item
        finally:
            if not worker.done():
                worker.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await worker

    return StreamingResponse(
        event_stream(),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "Connection": "keep-alive"},
    )



@router.post("/templates/{filename:path}/export", response_model=ExportResponse)
async def export_template(filename: str, request: ExportRequest):
    """
    Export template to specified format (supports subdirectories).
    Uses cached edited content if available, otherwise parses original file.
    
    Args:
        filename: Relative path to the file (e.g., "五数下导学案/1.1 等式与方程.docx")
        request: Export format (docx/pdf/md)
    
    Returns:
        ExportResponse with download URL
    """
    import tempfile
    import uuid
    import hashlib
    from fastapi.responses import FileResponse
    
    try:
        # Security: prevent path traversal attacks
        safe_path = Path(filename).as_posix()
        if '..' in safe_path or safe_path.startswith('/'):
            raise HTTPException(
                status_code=403,
                detail="访问被拒绝：非法路径"
            )
        
        # Try to get cached edited content first
        cache_dir = Path(tempfile.gettempdir()) / "template_cache"
        cache_key = hashlib.md5(filename.encode()).hexdigest()
        cache_file = cache_dir / f"{cache_key}.html"
        
        if cache_file.exists():
            content_html = cache_file.read_text(encoding='utf-8')
            logger.info(f"Using cached content for export: {filename}")
        else:
            # Parse original file
            file_path = TEMPLATES_DIR / safe_path
            if not file_path.exists():
                raise HTTPException(
                    status_code=404,
                    detail=f"文件不存在: {filename}"
                )
            parse_result = file_parser.parse_file(file_path)
            content_html = parse_result.html_content
            logger.info(f"Using original parsed content for export: {filename}")
        
        # Export based on format
        export_format = request.format.lower()
        
        if export_format == 'docx':
            export_bytes = export_service.export_to_docx(content_html)
            content_type = 'application/vnd.openxmlformats-officedocument.wordprocessingml.document'
            file_ext = 'docx'
        elif export_format == 'pdf':
            export_bytes = export_service.export_to_pdf(content_html)
            content_type = 'application/pdf'
            file_ext = 'pdf'
        elif export_format == 'md':
            export_text = export_service.export_to_markdown(content_html)
            export_bytes = export_text.encode('utf-8')
            content_type = 'text/markdown'
            file_ext = 'md'
        else:
            raise HTTPException(
                status_code=400,
                detail=f"不支持的导出格式: {export_format}"
            )
        
        # Save to temporary file
        temp_dir = Path(tempfile.gettempdir()) / "template_exports"
        temp_dir.mkdir(exist_ok=True)
        
        # Generate unique filename (use just the base filename, not the path)
        export_id = str(uuid.uuid4())[:8]
        base_name = Path(filename).stem
        export_filename = f"{base_name}_{export_id}.{file_ext}"
        export_path = temp_dir / export_filename
        
        # Write export file
        if isinstance(export_bytes, bytes):
            export_path.write_bytes(export_bytes)
        else:
            export_path.write_text(export_bytes)
        
        logger.info(f"Exported template {filename} to {export_format}, size: {len(export_bytes)} bytes")
        
        # Return download URL (relative path)
        download_url = f"/templates/download-export/{export_filename}"
        
        return ExportResponse(
            success=True,
            download_url=download_url,
            format=export_format,
            file_size=len(export_bytes)
        )
    
    except HTTPException:
        raise
    except Exception as e:
        logger.exception(f"Failed to export template: {filename}")
        raise HTTPException(
            status_code=500,
            detail=f"导出失败: {str(e)}"
        )


@router.get("/templates/download-export/{export_filename}")
async def download_export(export_filename: str):
    """
    Download exported file.
    
    Args:
        export_filename: Name of the exported file
    
    Returns:
        FileResponse with the exported file
    """
    try:
        # Security: prevent path traversal
        safe_filename = Path(export_filename).name
        temp_dir = Path(tempfile.gettempdir()) / "template_exports"
        export_path = temp_dir / safe_filename
        
        # Check if file exists
        if not export_path.exists() or not export_path.is_file():
            raise HTTPException(
                status_code=404,
                detail=f"导出文件不存在: {safe_filename}"
            )
        
        # Check if file is within temp directory (security)
        if not str(export_path.resolve()).startswith(str(temp_dir.resolve())):
            raise HTTPException(
                status_code=403,
                detail="访问被拒绝"
            )
        
        # Determine media type
        suffix = export_path.suffix.lower()
        media_type_map = {
            '.docx': 'application/vnd.openxmlformats-officedocument.wordprocessingml.document',
            '.pdf': 'application/pdf',
            '.md': 'text/markdown',
        }
        media_type = media_type_map.get(suffix, 'application/octet-stream')
        
        logger.info(f"Downloading export file: {safe_filename}")
        
        return FileResponse(
            path=export_path,
            filename=safe_filename,
            media_type=media_type
        )
    
    except HTTPException:
        raise
    except Exception as e:
        logger.exception(f"Failed to download export: {export_filename}")
        raise HTTPException(
            status_code=500,
            detail=f"下载失败: {str(e)}"
        )


# ============================================================================
# Metadata Management Endpoints
# ============================================================================

@router.post("/templates/index")
async def index_templates(force: bool = False):
    """
    Index all template files to build metadata.
    
    Args:
        force: Force re-indexing of all files
    
    Returns:
        Status of indexing operation
    """
    try:
        logger.info(f"Starting template indexing (force={force})")
        metadata_service.index_all(force=force)
        
        return {
            "success": True,
            "message": "模板索引已更新",
            "indexed_count": len(metadata_service._index)
        }
    except Exception as e:
        logger.exception("Failed to index templates")
        raise HTTPException(
            status_code=500,
            detail=f"索引失败: {str(e)}"
        )


@router.post("/templates/{filename:path}/index")
async def index_single_template(filename: str, force: bool = False):
    """
    Index a single template file.
    
    Args:
        filename: Template filename (relative path)
        force: Force re-indexing
    
    Returns:
        Metadata for the indexed file
    """
    try:
        file_path = TEMPLATES_DIR / filename
        if not file_path.exists():
            raise HTTPException(status_code=404, detail="文件不存在")
        
        metadata = metadata_service.index_file(file_path, force=force)
        
        if metadata:
            return {
                "success": True,
                "metadata": metadata.to_dict()
            }
        else:
            raise HTTPException(
                status_code=500,
                detail="索引失败"
            )
    except HTTPException:
        raise
    except Exception as e:
        logger.exception(f"Failed to index template: {filename}")
        raise HTTPException(
            status_code=500,
            detail=f"索引失败: {str(e)}"
        )


class UpdateMetadataRequest(BaseModel):
    """Request to update template metadata."""
    desc: Optional[str] = None
    keywords: Optional[List[str]] = None


@router.put("/templates/{filename:path}/metadata")
async def update_template_metadata(filename: str, request: UpdateMetadataRequest):
    """
    Update metadata for a template file.
    
    Args:
        filename: Template filename (relative path)
        request: Metadata update request
    
    Returns:
        Updated metadata
    """
    try:
        metadata_service.update_metadata(
            filename,
            desc=request.desc,
            keywords=request.keywords
        )
        
        metadata = metadata_service.get_metadata(filename)
        
        return {
            "success": True,
            "metadata": metadata.to_dict() if metadata else None
        }
    except Exception as e:
        logger.exception(f"Failed to update metadata: {filename}")
        raise HTTPException(
            status_code=500,
            detail=f"更新元数据失败: {str(e)}"
        )


@router.get("/templates/{filename:path}/metadata")
async def get_template_metadata(filename: str):
    """
    Get metadata for a template file.
    
    Args:
        filename: Template filename (relative path)
    
    Returns:
        Template metadata
    """
    try:
        metadata = metadata_service.get_metadata(filename)
        
        if metadata:
            return {
                "success": True,
                "metadata": metadata.to_dict()
            }
        else:
            return {
                "success": False,
                "message": "未找到元数据，请先索引该文件"
            }
    except Exception as e:
        logger.exception(f"Failed to get metadata: {filename}")
        raise HTTPException(
            status_code=500,
            detail=f"获取元数据失败: {str(e)}"
        )
