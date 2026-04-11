"""Pydantic request/response schemas for API routes."""

from __future__ import annotations

from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field


class ChatRequest(BaseModel):
    question: str = Field(..., min_length=1, description="用户问题")
    collection: str = Field(default="default", description="知识库集合名称")
    top_k: Optional[int] = Field(default=None, description="检索数量，None 时使用配置文件默认值")
    use_rerank: bool = Field(default=True, description="是否启用重排序（仍受配置文件约束）")


class LessonPlanRequest(BaseModel):
    topic: str = Field(..., min_length=1, description="教案主题")
    notes: Optional[str] = Field(default=None, description="教师备注或补充要求")
    collection: str = Field(default="default", description="知识库集合名称")
    model: Optional[str] = Field(default=None, description="LLM模型名称，不指定则使用默认配置")
    include_background: bool = Field(default=True, description="是否包含背景信息")
    include_facts: bool = Field(default=True, description="是否包含相关常识")
    include_examples: bool = Field(default=True, description="是否包含教学示例")
    template_category: Optional[str] = Field(default=None, description="模板类别")
    conversation_state: Optional[Dict[str, Any]] = Field(default=None, description="轻量会话状态")
    allow_ai_visuals: bool = Field(
        default=False,
        description="兼容旧请求字段。当前前端已改为根据备注自动判断是否需要 AI 示意图。",
    )


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
    source_type: str = "retrieved"
    role: Optional[str] = None
    model: Optional[str] = None


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
    lesson_content: Optional[str] = None
    additional_resources: List[Citation] = Field(default_factory=list)
    image_resources: List[LessonImageResource] = Field(default_factory=list)
    review_report: Optional[LessonReviewReportResponse] = None
    conversation_state: Optional[Dict[str, Any]] = None
    history_records: List[Dict[str, Any]] = Field(default_factory=list)
    execution_plan: Optional[Dict[str, Any]] = None
    planning_mode: Optional[str] = None
    used_autonomous_fallback: bool = False


class ExportDocxRequest(BaseModel):
    title: str = Field(..., min_length=1, description="导出文件标题")
    content_html: str = Field(..., min_length=1, description="当前教案正文 HTML")


class LessonHistoryResponse(BaseModel):
    records: List[Dict[str, Any]] = Field(default_factory=list)


class LessonTemplateCategoryItem(BaseModel):
    category: str
    template_type: str
    label: str
    description: str
    default: bool = False


class LessonTemplateCategoriesResponse(BaseModel):
    templates: List[LessonTemplateCategoryItem] = Field(default_factory=list)


class HealthResponse(BaseModel):
    status: str
    components: dict


class ClientConfigResponse(BaseModel):
    lesson_plan_mock_enabled: bool = False
    image_generation_enabled: bool = False


class ImageGenerationExperimentRequest(BaseModel):
    prompt: str = Field(..., min_length=1, description="图片生成提示词")
    topic: Optional[str] = Field(default=None, description="关联主题")
    style: str = Field(default="diagram_clean", description="试验风格")


class ImageGenerationExperimentResponse(BaseModel):
    image_url: str
    image_path: str
    filename: str
    model: Optional[str] = None
    style: str
    prompt: str
    topic: Optional[str] = None
