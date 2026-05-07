"""Document import and processing router."""

from __future__ import annotations

import asyncio
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

from fastapi import APIRouter, HTTPException, Request, UploadFile, File
from fastapi.responses import JSONResponse

from app.core.paths import TEMP_UPLOADS_DIR
from app.schemas.document_models import (
    DocumentUploadResponse,
    DocumentProcessingRequest,
    DocumentProcessingResponse,
    ProcessingHistoryResponse,
    ProcessingHistoryDeleteResponse,
)
from app.services.file_parser_service import FileParserService
from app.services.document_processor_service import DocumentProcessorService
from app.services.processing_history_service import ProcessingHistoryService
from app.services.docx_export_service import build_lesson_docx_bytes
from src.observability.logger import get_logger

router = APIRouter()
logger = get_logger(__name__)

# Initialize services
file_parser = FileParserService()
history_service = ProcessingHistoryService()

# Supported file formats
SUPPORTED_FORMATS = {'.doc', '.docx', '.pdf'}
MAX_FILE_SIZE = 10 * 1024 * 1024  # 10MB

# In-memory storage for uploaded documents (temporary)
# In production, this should be replaced with a database
_uploaded_documents = {}


@router.post("/documents/upload", response_model=DocumentUploadResponse)
async def upload_document(file: UploadFile = File(...)):
    """
    Upload and parse a document file.
    
    Supports: .doc, .docx, .pdf
    Max size: 10MB
    
    Args:
        file: Uploaded document file
    
    Returns:
        DocumentUploadResponse with document ID and parsed content preview
    """
    try:
        # Validate file format
        if not file.filename:
            raise HTTPException(
                status_code=400,
                detail="文件名不能为空"
            )
        
        file_suffix = Path(file.filename).suffix.lower()
        if file_suffix not in SUPPORTED_FORMATS:
            raise HTTPException(
                status_code=400,
                detail=f"不支持的文件格式,请上传Word或PDF文档。支持的格式: {', '.join(SUPPORTED_FORMATS)}"
            )
        
        # Read file content
        file_content = await file.read()
        file_size = len(file_content)
        
        # Validate file size
        if file_size > MAX_FILE_SIZE:
            raise HTTPException(
                status_code=400,
                detail=f"文件大小超过限制,请上传小于10MB的文件。当前文件大小: {file_size / 1024 / 1024:.1f}MB"
            )
        
        # Generate unique document ID
        document_id = str(uuid.uuid4())
        uploaded_at = datetime.now(timezone.utc).isoformat()
        
        # Ensure temp uploads directory exists
        TEMP_UPLOADS_DIR.mkdir(parents=True, exist_ok=True)
        
        # Save file temporarily
        temp_file_path = TEMP_UPLOADS_DIR / f"{document_id}{file_suffix}"
        temp_file_path.write_bytes(file_content)
        
        logger.info(f"Saved uploaded file: {file.filename} -> {temp_file_path}")
        
        # Parse document
        try:
            parse_result = file_parser.parse_file(temp_file_path)
            
            # Extract text from HTML
            from bs4 import BeautifulSoup
            soup = BeautifulSoup(parse_result.html_content, 'html.parser')
            full_text = soup.get_text(separator='\n', strip=True)
            
            # Create text preview (first 500 characters)
            text_preview = full_text[:500] if len(full_text) > 500 else full_text
            
            # Calculate word count
            word_count = len(full_text)
            
            # Store document info in memory
            _uploaded_documents[document_id] = {
                'document_id': document_id,
                'filename': file.filename,
                'file_path': str(temp_file_path),
                'file_size': file_size,
                'uploaded_at': uploaded_at,
                'full_text': full_text,
                'metadata': {
                    **parse_result.metadata,
                    'word_count': word_count,
                }
            }
            
            logger.info(
                f"Parsed document: {file.filename}, "
                f"size: {file_size} bytes, "
                f"words: {word_count}, "
                f"parser: {parse_result.metadata.get('parser')}"
            )
            
            return DocumentUploadResponse(
                document_id=document_id,
                filename=file.filename,
                file_size=file_size,
                uploaded_at=uploaded_at,
                text_preview=text_preview,
                metadata={
                    **parse_result.metadata,
                    'word_count': word_count,
                }
            )
        
        except Exception as parse_error:
            # Clean up temp file on parse error
            if temp_file_path.exists():
                temp_file_path.unlink()
            
            logger.exception(f"Failed to parse document: {file.filename}")
            raise HTTPException(
                status_code=422,
                detail=f"文档解析失败,请检查文件是否损坏或格式是否正确。错误: {str(parse_error)}"
            )
    
    except HTTPException:
        raise
    except Exception as e:
        logger.exception("Document upload failed")
        raise HTTPException(
            status_code=500,
            detail=f"文档上传失败: {str(e)}"
        )


@router.get("/documents/{document_id}")
async def get_document_info(document_id: str):
    """
    Get document information by ID.
    
    Args:
        document_id: Document ID
    
    Returns:
        Document information
    """
    if document_id not in _uploaded_documents:
        raise HTTPException(
            status_code=404,
            detail="文档不存在或已过期"
        )
    
    doc_info = _uploaded_documents[document_id]
    
    return {
        'document_id': doc_info['document_id'],
        'filename': doc_info['filename'],
        'file_size': doc_info['file_size'],
        'uploaded_at': doc_info['uploaded_at'],
        'metadata': doc_info['metadata'],
    }


@router.post("/documents/process", response_model=DocumentProcessingResponse)
async def process_document(request: DocumentProcessingRequest, req: Request):
    """
    Process document with AI using specified processing option.
    
    Args:
        request: Processing request with document_id, processing_option, and optional custom_prompt
        req: FastAPI request (to access LLM from app state)
    
    Returns:
        DocumentProcessingResponse with processing result
    """
    import time
    
    try:
        # Validate document exists
        if request.document_id not in _uploaded_documents:
            raise HTTPException(
                status_code=404,
                detail="文档不存在或已过期"
            )
        
        doc_info = _uploaded_documents[request.document_id]
        document_text = doc_info['full_text']
        
        # Validate processing option
        valid_options = {'extract_exercises', 'summarize', 'extract_teaching_thoughts', 'custom'}
        if request.processing_option not in valid_options:
            raise HTTPException(
                status_code=400,
                detail=f"无效的处理选项。有效选项: {', '.join(valid_options)}"
            )
        
        # Validate custom prompt for 'custom' option
        if request.processing_option == 'custom':
            if not request.custom_prompt:
                raise HTTPException(
                    status_code=400,
                    detail="自定义处理选项需要提供custom_prompt"
                )
            if len(request.custom_prompt) < 10 or len(request.custom_prompt) > 500:
                raise HTTPException(
                    status_code=400,
                    detail="自定义prompt长度必须在10到500字符之间"
                )
        
        # Get LLM from app state
        llm = req.app.state.llm
        
        # Create processor service
        processor = DocumentProcessorService(llm)
        
        # Determine timeout based on processing option
        # Increased timeout to allow for slower LLM responses
        timeout = 60.0  # 60 seconds for all processing options
        
        # Process document
        start_time = time.time()
        
        try:
            result = await processor.process_document(
                document_text=document_text,
                processing_option=request.processing_option,
                custom_prompt=request.custom_prompt,
                timeout=timeout
            )
        except asyncio.TimeoutError:
            raise HTTPException(
                status_code=408,
                detail=f"处理超时,请简化您的指令或稍后重试。(超时时间: {timeout}秒)"
            )
        
        processing_time_ms = (time.time() - start_time) * 1000
        
        # Generate processing ID
        processing_id = str(uuid.uuid4())
        processed_at = datetime.now(timezone.utc).isoformat()
        
        # Get model info from LLM
        model_name = getattr(llm, 'model', 'unknown')
        
        # Save to history
        history_service.save_history(
            processing_id=processing_id,
            document_filename=doc_info['filename'],
            processing_option=request.processing_option,
            result=result,
            custom_prompt=request.custom_prompt if request.processing_option == 'custom' else None
        )
        
        logger.info(
            f"Document processed: document_id={request.document_id}, "
            f"option={request.processing_option}, "
            f"time={processing_time_ms:.1f}ms, "
            f"result_length={len(result)}"
        )
        
        return DocumentProcessingResponse(
            processing_id=processing_id,
            result=result,
            processing_option=request.processing_option,
            processed_at=processed_at,
            metadata={
                'model': model_name,
                'processing_time_ms': processing_time_ms,
                'document_filename': doc_info['filename'],
                'custom_prompt': request.custom_prompt if request.processing_option == 'custom' else None,
            }
        )
    
    except HTTPException:
        raise
    except Exception as e:
        logger.exception("Document processing failed")
        raise HTTPException(
            status_code=500,
            detail=f"AI处理失败: {str(e)}"
        )



@router.get("/documents/processing-history", response_model=ProcessingHistoryResponse)
async def get_processing_history(limit: int = 10, offset: int = 0):
    """
    Get processing history list with pagination.
    
    Args:
        limit: Maximum number of items to return (default: 10, max: 50)
        offset: Number of items to skip (default: 0)
    
    Returns:
        ProcessingHistoryResponse with history items
    """
    try:
        items = history_service.get_history(limit=limit, offset=offset)
        total = history_service.get_total_count()
        has_more = (offset + len(items)) < total
        
        logger.info(f"Retrieved processing history: {len(items)} items, total={total}")
        
        return ProcessingHistoryResponse(
            items=items,
            total=total,
            has_more=has_more
        )
    
    except Exception as e:
        logger.exception("Failed to get processing history")
        raise HTTPException(
            status_code=500,
            detail=f"获取处理历史失败: {str(e)}"
        )


@router.get("/documents/processing-history/{processing_id}/result")
async def get_processing_result(processing_id: str):
    """
    Get full processing result by processing ID.
    
    Args:
        processing_id: Processing ID
    
    Returns:
        Full processing result
    """
    try:
        result = history_service.get_full_result(processing_id)
        
        if result is None:
            raise HTTPException(
                status_code=404,
                detail="处理记录不存在"
            )
        
        return {
            'processing_id': processing_id,
            'result': result
        }
    
    except HTTPException:
        raise
    except Exception as e:
        logger.exception(f"Failed to get processing result: {processing_id}")
        raise HTTPException(
            status_code=500,
            detail=f"获取处理结果失败: {str(e)}"
        )


@router.delete("/documents/processing-history", response_model=ProcessingHistoryDeleteResponse)
async def clear_processing_history():
    """
    Clear all processing history.
    
    Returns:
        ProcessingHistoryDeleteResponse with deleted count
    """
    try:
        deleted_count = history_service.clear_history()
        
        logger.info(f"Cleared processing history: {deleted_count} items deleted")
        
        return ProcessingHistoryDeleteResponse(
            deleted_count=deleted_count
        )
    
    except Exception as e:
        logger.exception("Failed to clear processing history")
        raise HTTPException(
            status_code=500,
            detail=f"清空处理历史失败: {str(e)}"
        )


@router.post("/documents/export-docx")
async def export_document_docx(request: Request):
    """
    Export document processing result as DOCX file.
    
    Args:
        request: FastAPI request with JSON body containing:
            - content_html: HTML content to export
            - title: Document title
    
    Returns:
        DOCX file as binary response
    """
    try:
        body = await request.json()
        content_html = body.get('content_html', '')
        title = body.get('title', '处理结果')
        
        if not content_html:
            raise HTTPException(
                status_code=400,
                detail="内容不能为空"
            )
        
        logger.info(f"Exporting document: {title}")
        logger.debug(f"HTML content preview (first 500 chars): {content_html[:500]}")
        
        # Generate DOCX bytes
        def resolve_image_path(src: str):
            # For document processing results, we don't have local images
            return None
        
        def resolve_image_bytes(src: str):
            # For document processing results, we don't have local images
            return None
        
        try:
            docx_bytes = build_lesson_docx_bytes(
                content_html=content_html,
                resolve_image_path=resolve_image_path,
                resolve_image_bytes=resolve_image_bytes,
            )
        except Exception as e:
            logger.exception(f"Failed to build DOCX: {str(e)}")
            raise HTTPException(
                status_code=500,
                detail=f"生成Word文档失败: {str(e)}"
            )
        
        # Return as downloadable file
        from fastapi.responses import Response
        import re
        from urllib.parse import quote
        
        # Sanitize filename
        safe_title = re.sub(r'[\\/:*?"<>|]+', '_', title).strip() or '处理结果'
        filename = f"{safe_title}.docx"
        
        # Encode filename for Content-Disposition header (RFC 5987)
        encoded_filename = quote(filename)
        
        logger.info(f"Exported document: {filename}, size: {len(docx_bytes)} bytes")
        
        return Response(
            content=docx_bytes,
            media_type='application/vnd.openxmlformats-officedocument.wordprocessingml.document',
            headers={
                'Content-Disposition': f"attachment; filename*=UTF-8''{encoded_filename}"
            }
        )
    
    except HTTPException:
        raise
    except Exception as e:
        logger.exception("Document export failed")
        raise HTTPException(
            status_code=500,
            detail=f"导出失败: {str(e)}"
        )
