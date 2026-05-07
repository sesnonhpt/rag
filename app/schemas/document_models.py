"""Pydantic models for document import and processing."""

from __future__ import annotations

from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field


class DocumentUploadResponse(BaseModel):
    """Response model for document upload."""
    document_id: str = Field(..., description="唯一文档ID")
    filename: str = Field(..., description="文件名")
    file_size: int = Field(..., description="文件大小(字节)")
    uploaded_at: str = Field(..., description="上传时间(ISO格式)")
    text_preview: str = Field(..., description="文本预览(前500字符)")
    metadata: Dict[str, Any] = Field(default_factory=dict, description="文档元数据")


class DocumentProcessingRequest(BaseModel):
    """Request model for document processing."""
    document_id: str = Field(..., description="文档ID")
    processing_option: str = Field(
        ...,
        description="处理选项: extract_exercises, summarize, extract_teaching_thoughts, custom"
    )
    custom_prompt: Optional[str] = Field(
        None,
        min_length=10,
        max_length=500,
        description="自定义prompt(当processing_option为custom时必填)"
    )


class DocumentProcessingResponse(BaseModel):
    """Response model for document processing."""
    processing_id: str = Field(..., description="处理记录ID")
    result: str = Field(..., description="处理结果文本")
    processing_option: str = Field(..., description="使用的处理选项")
    processed_at: str = Field(..., description="处理完成时间(ISO格式)")
    metadata: Dict[str, Any] = Field(default_factory=dict, description="处理元数据")


class ProcessingHistoryItem(BaseModel):
    """Model for a single processing history item."""
    processing_id: str = Field(..., description="处理记录ID")
    document_filename: str = Field(..., description="文档文件名")
    processing_option: str = Field(..., description="处理选项")
    custom_prompt: Optional[str] = Field(None, description="自定义prompt(如有)")
    result_preview: str = Field(..., description="结果预览(前200字符)")
    processed_at: str = Field(..., description="处理时间(ISO格式)")


class ProcessingHistoryResponse(BaseModel):
    """Response model for processing history list."""
    items: List[ProcessingHistoryItem] = Field(default_factory=list, description="历史记录列表")
    total: int = Field(..., description="总记录数")
    has_more: bool = Field(..., description="是否有更多记录")


class ProcessingHistoryDeleteResponse(BaseModel):
    """Response model for clearing processing history."""
    deleted_count: int = Field(..., description="删除的记录数")
