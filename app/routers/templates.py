"""Template management router - Phase 1 & 2: File listing and editing."""

from __future__ import annotations

import os
from pathlib import Path
from typing import List, Optional, Dict, Any

from fastapi import APIRouter, HTTPException, Request
from fastapi.responses import FileResponse, Response
from pydantic import BaseModel, Field

from app.core.paths import APP_ROOT
from app.services.template_database import TemplateDatabase
from app.services.file_parser_service import FileParserService
from app.services.template_export_service import TemplateExportService
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


class TemplateFileInfo(BaseModel):
    """Template file information."""
    filename: str
    size_bytes: int
    size_display: str
    modified_at: str
    file_type: str


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
async def list_templates(search: Optional[str] = None):
    """
    List all template files in the templates directory.
    
    Args:
        search: Optional search query to filter filenames
    
    Returns:
        TemplateListResponse with file information
    """
    try:
        # Ensure templates directory exists
        TEMPLATES_DIR.mkdir(parents=True, exist_ok=True)
        
        # Get all files in directory
        files = []
        for item in TEMPLATES_DIR.iterdir():
            if item.is_file():
                # Apply search filter if provided
                if search and search.lower() not in item.name.lower():
                    continue
                
                stat = item.stat()
                files.append(TemplateFileInfo(
                    filename=item.name,
                    size_bytes=stat.st_size,
                    size_display=_format_file_size(stat.st_size),
                    modified_at=str(stat.st_mtime),
                    file_type=_get_file_type(item.name)
                ))
        
        # Sort by modified time (newest first)
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


@router.get("/templates/download/{filename}")
async def download_template(filename: str):
    """
    Download a template file.
    
    Args:
        filename: Name of the file to download
    
    Returns:
        FileResponse with the requested file
    """
    try:
        # Security: prevent path traversal
        safe_filename = Path(filename).name
        file_path = TEMPLATES_DIR / safe_filename
        
        # Check if file exists
        if not file_path.exists() or not file_path.is_file():
            raise HTTPException(
                status_code=404,
                detail=f"文件不存在: {safe_filename}"
            )
        
        # Check if file is within templates directory (security)
        if not str(file_path.resolve()).startswith(str(TEMPLATES_DIR.resolve())):
            raise HTTPException(
                status_code=403,
                detail="访问被拒绝"
            )
        
        logger.info(f"Downloading template file: {safe_filename}")
        
        return FileResponse(
            path=file_path,
            filename=safe_filename,
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


@router.get("/templates/{filename}/content", response_model=TemplateContentResponse)
async def get_template_content(filename: str):
    """
    Get template content for editing.
    Parse the file and return HTML content.
    
    Args:
        filename: Name of the file
    
    Returns:
        TemplateContentResponse with HTML content
    """
    try:
        # Security: prevent path traversal
        safe_filename = Path(filename).name
        file_path = TEMPLATES_DIR / safe_filename
        
        # Check if file exists
        if not file_path.exists() or not file_path.is_file():
            raise HTTPException(
                status_code=404,
                detail=f"文件不存在: {safe_filename}"
            )
        
        # Check if file is within templates directory
        if not str(file_path.resolve()).startswith(str(TEMPLATES_DIR.resolve())):
            raise HTTPException(
                status_code=403,
                detail="访问被拒绝"
            )
        
        # Parse file to HTML
        parse_result = file_parser.parse_file(file_path)
        
        logger.info(f"Retrieved template content: {safe_filename}")
        
        return TemplateContentResponse(
            template_id=safe_filename,
            filename=safe_filename,
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


@router.put("/templates/{filename}/content")
async def save_template_content(filename: str, request: TemplateContentRequest):
    """
    Save edited template content (simplified - no versioning).
    Content is stored in memory for export only.
    
    Args:
        filename: Name of the file
        request: Content to save
    
    Returns:
        Success response
    """
    try:
        safe_filename = Path(filename).name
        
        # Store content in a simple in-memory cache for export
        # In a real app, you might want to use Redis or similar
        import tempfile
        cache_dir = Path(tempfile.gettempdir()) / "template_cache"
        cache_dir.mkdir(exist_ok=True)
        cache_file = cache_dir / f"{safe_filename}.html"
        cache_file.write_text(request.content_html, encoding='utf-8')
        
        logger.info(f"Saved template content to cache: {safe_filename}")
        
        return {
            "success": True,
            "template_id": safe_filename,
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
    
    except Exception as e:
        logger.exception("AI modification failed")
        raise HTTPException(
            status_code=500,
            detail=f"AI 修改失败: {str(e)}"
        )



@router.post("/templates/{filename}/export", response_model=ExportResponse)
async def export_template(filename: str, request: ExportRequest):
    """
    Export template to specified format.
    Uses cached edited content if available, otherwise parses original file.
    
    Args:
        filename: Name of the file
        request: Export format (docx/pdf/md)
    
    Returns:
        ExportResponse with download URL
    """
    import tempfile
    import uuid
    from fastapi.responses import FileResponse
    
    try:
        safe_filename = Path(filename).name
        
        # Try to get cached edited content first
        cache_dir = Path(tempfile.gettempdir()) / "template_cache"
        cache_file = cache_dir / f"{safe_filename}.html"
        
        if cache_file.exists():
            content_html = cache_file.read_text(encoding='utf-8')
            logger.info(f"Using cached content for export: {safe_filename}")
        else:
            # Parse original file
            file_path = TEMPLATES_DIR / safe_filename
            if not file_path.exists():
                raise HTTPException(
                    status_code=404,
                    detail=f"文件不存在: {safe_filename}"
                )
            parse_result = file_parser.parse_file(file_path)
            content_html = parse_result.html_content
            logger.info(f"Using original parsed content for export: {safe_filename}")
        
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
        
        # Generate unique filename
        export_id = str(uuid.uuid4())[:8]
        base_name = Path(safe_filename).stem
        export_filename = f"{base_name}_{export_id}.{file_ext}"
        export_path = temp_dir / export_filename
        
        # Write export file
        if isinstance(export_bytes, bytes):
            export_path.write_bytes(export_bytes)
        else:
            export_path.write_text(export_bytes)
        
        logger.info(f"Exported template {safe_filename} to {export_format}, size: {len(export_bytes)} bytes")
        
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
