"""Template management router - Phase 1 & 2: File listing and editing."""

from __future__ import annotations

import os
import tempfile
from pathlib import Path
from typing import List, Optional, Dict, Any

from fastapi import APIRouter, HTTPException, Request
from fastapi.responses import FileResponse, Response
from pydantic import BaseModel, Field

from app.core.paths import APP_ROOT
from app.services.template_database import TemplateDatabase
from app.services.file_parser_service import FileParserService
from app.services.template_export_service import TemplateExportService
from app.services.template_metadata_service import TemplateMetadataService
from src.libs.llm.openai_llm import OpenAILLMError
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
