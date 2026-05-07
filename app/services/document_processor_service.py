"""Document intelligent processing service."""

from __future__ import annotations

import asyncio
from pathlib import Path
from typing import Dict, Optional

from app.core.paths import PROJECT_ROOT
from src.libs.llm.base_llm import BaseLLM, Message
from src.observability.logger import get_logger

logger = get_logger(__name__)

# Prompt templates directory
PROMPTS_DIR = PROJECT_ROOT / "config" / "prompts" / "document_processing"


class DocumentProcessorService:
    """Service for intelligent document processing using LLM."""
    
    def __init__(self, llm: BaseLLM):
        """
        Initialize document processor service.
        
        Args:
            llm: LLM instance for processing
        """
        self.llm = llm
        self.preset_prompts = self._load_preset_prompts()
    
    def _load_preset_prompts(self) -> Dict[str, str]:
        """
        Load preset prompt templates from files.
        
        Returns:
            Dictionary mapping processing option to prompt template
        """
        prompts = {}
        
        # Ensure prompts directory exists
        PROMPTS_DIR.mkdir(parents=True, exist_ok=True)
        
        # Load each preset prompt file
        prompt_files = {
            'extract_exercises': 'extract_exercises.txt',
            'summarize': 'summarize.txt',
            'extract_teaching_thoughts': 'extract_teaching_thoughts.txt',
        }
        
        for option, filename in prompt_files.items():
            prompt_file = PROMPTS_DIR / filename
            if prompt_file.exists():
                prompts[option] = prompt_file.read_text(encoding='utf-8')
                logger.info(f"Loaded preset prompt: {option}")
            else:
                logger.warning(f"Preset prompt file not found: {prompt_file}")
                # Provide fallback prompts
                prompts[option] = self._get_fallback_prompt(option)
        
        return prompts
    
    def _get_fallback_prompt(self, option: str) -> str:
        """
        Get fallback prompt template if file doesn't exist.
        
        Args:
            option: Processing option
        
        Returns:
            Fallback prompt template
        """
        fallback_prompts = {
            'extract_exercises': """请从以下文档中提取所有习题内容。

要求:
1. 识别选择题、填空题、简答题、计算题等题型
2. 提取题干、选项(如有)和答案(如有)
3. 按题型分类组织
4. 保持题目的完整性和准确性

文档内容:
{document_text}

请按以下格式输出:

## 选择题
1. [题目内容]
   A. [选项A]
   B. [选项B]
   C. [选项C]
   D. [选项D]
   答案: [正确答案]

## 填空题
1. [题目内容]
   答案: [答案]

## 简答题
1. [题目内容]
   答案: [答案要点]

## 计算题
1. [题目内容]
   答案: [解答过程]
""",
            'summarize': """请对以下文档进行归纳总结。

要求:
1. 提取核心知识点和关键概念
2. 总结主要论述内容
3. 生成不超过500字的总结
4. 按"核心内容"、"关键知识点"、"教学建议"三个部分组织
5. 保持客观准确,不添加文档中不存在的信息

文档内容:
{document_text}

请按以下格式输出:

## 核心内容
[总结文档的核心内容和主题]

## 关键知识点
- [知识点1]
- [知识点2]
- [知识点3]

## 教学建议
[基于文档内容提供的教学建议]
""",
            'extract_teaching_thoughts': """请从以下教案文档中提取教学思路。

要求:
1. 识别教学环节、教学方法和设计意图
2. 提取教学导入方式、知识展开逻辑、师生互动设计和课堂收束方法
3. 按教学流程的时间顺序组织
4. 包含每个教学环节的目的说明和实施要点

文档内容:
{document_text}

请按以下格式输出:

## 教学导入
- 导入方式: [描述]
- 设计意图: [说明]
- 预计时间: [分钟]

## 知识展开
- 展开逻辑: [描述]
- 教学方法: [说明]
- 预计时间: [分钟]

## 师生互动
- 互动设计: [描述]
- 活动形式: [说明]
- 预计时间: [分钟]

## 课堂收束
- 收束方法: [描述]
- 总结要点: [说明]
- 预计时间: [分钟]
""",
        }
        
        return fallback_prompts.get(option, "请处理以下文档:\n\n{document_text}")
    
    def _build_prompt(
        self,
        document_text: str,
        processing_option: str,
        custom_prompt: Optional[str] = None
    ) -> str:
        """
        Build complete LLM prompt.
        
        Args:
            document_text: Document text content
            processing_option: Processing option (extract_exercises, summarize, etc.)
            custom_prompt: Custom prompt (for 'custom' option)
        
        Returns:
            Complete prompt string
        """
        if processing_option == 'custom' and custom_prompt:
            # For custom option, use user's prompt
            return f"{custom_prompt}\n\n文档内容:\n{document_text}"
        
        # For preset options, use template
        template = self.preset_prompts.get(processing_option, "{document_text}")
        return template.format(document_text=document_text)
    
    def _format_result(self, raw_result: str, processing_option: str) -> str:
        """
        Format processing result.
        
        Args:
            raw_result: Raw LLM output
            processing_option: Processing option used
        
        Returns:
            Formatted result string
        """
        # Basic formatting: strip whitespace
        formatted = raw_result.strip()
        
        # Option-specific formatting can be added here
        # For now, just return the cleaned result
        
        return formatted
    
    async def process_document(
        self,
        document_text: str,
        processing_option: str,
        custom_prompt: Optional[str] = None,
        timeout: float = 15.0
    ) -> str:
        """
        Process document content using LLM.
        
        Args:
            document_text: Document text content
            processing_option: Processing option (extract_exercises, summarize, extract_teaching_thoughts, custom)
            custom_prompt: Custom prompt (required for 'custom' option)
            timeout: Processing timeout in seconds (default: 15s)
        
        Returns:
            Processed result text
        
        Raises:
            ValueError: If processing_option is invalid or custom_prompt is missing for 'custom' option
            asyncio.TimeoutError: If processing exceeds timeout
            RuntimeError: If LLM processing fails
        """
        # Validate processing option
        valid_options = {'extract_exercises', 'summarize', 'extract_teaching_thoughts', 'custom'}
        if processing_option not in valid_options:
            raise ValueError(
                f"Invalid processing option: {processing_option}. "
                f"Valid options: {', '.join(valid_options)}"
            )
        
        # Validate custom prompt for 'custom' option
        if processing_option == 'custom' and not custom_prompt:
            raise ValueError("Custom prompt is required for 'custom' processing option")
        
        logger.info(
            f"Processing document: option={processing_option}, "
            f"text_length={len(document_text)}, "
            f"timeout={timeout}s"
        )
        
        try:
            # Build prompt
            prompt = self._build_prompt(document_text, processing_option, custom_prompt)
            
            # Prepare messages
            system_message = Message(
                role="system",
                content="你是一位专业的教学内容分析助手,擅长从教学资料中提取和整理有价值的信息。"
            )
            user_message = Message(role="user", content=prompt)
            messages = [system_message, user_message]
            
            # Call LLM with timeout
            def _call_llm():
                return self.llm.chat(messages)
            
            response = await asyncio.wait_for(
                asyncio.to_thread(_call_llm),
                timeout=timeout
            )
            
            # Extract and format result
            raw_result = response.content
            formatted_result = self._format_result(raw_result, processing_option)
            
            logger.info(
                f"Document processing completed: option={processing_option}, "
                f"result_length={len(formatted_result)}"
            )
            
            return formatted_result
        
        except asyncio.TimeoutError:
            logger.error(f"Document processing timed out after {timeout}s")
            raise asyncio.TimeoutError(
                f"处理超时,请简化您的指令或稍后重试。(超时时间: {timeout}秒)"
            )
        except Exception as e:
            logger.exception(f"Document processing failed: {e}")
            raise RuntimeError(f"文档处理失败: {str(e)}")
