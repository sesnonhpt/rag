"""
导学案分析服务

分析现有导学案，提取关键信息
"""

import json
import re
from typing import Dict, Any, Optional
from dataclasses import dataclass, asdict

from src.observability.logger import get_logger

logger = get_logger(__name__)


@dataclass
class LessonAnalysis:
    """导学案分析结果"""
    topic: str  # 知识点
    subject: str  # 学科
    grade: str  # 年级
    difficulty: str  # 难度（简单/中等/困难）
    question_types: Dict[str, int]  # 题型分布
    teaching_sections: list[str]  # 教学环节
    key_points: list[str]  # 重点内容
    
    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


class LessonAnalyzer:
    """导学案分析器"""
    
    def __init__(self, llm):
        self.llm = llm
    
    async def analyze(self, content: str) -> LessonAnalysis:
        """
        分析导学案内容
        
        Args:
            content: 导学案文本内容
        
        Returns:
            LessonAnalysis: 分析结果
        """
        logger.info("lesson_analyzer.analyze_start content_length=%d", len(content))
        
        try:
            # 构建提示词
            prompt = self._build_prompt(content)
            
            # 调用 LLM
            from src.libs.llm.base_llm import Message
            response = await self.llm.chat([
                Message(
                    role="system",
                    content="你是教学内容分析专家，擅长分析导学案并提取关键信息。"
                ),
                Message(role="user", content=prompt)
            ])
            
            # 解析返回内容
            analysis = self._parse_response(response.content, content)
            
            logger.info(
                "lesson_analyzer.analyze_success topic=%s difficulty=%s",
                analysis.topic, analysis.difficulty
            )
            
            return analysis
            
        except Exception as e:
            logger.exception("lesson_analyzer.analyze_failed")
            # 降级：返回基础分析
            return self._get_basic_analysis(content)
    
    def _build_prompt(self, content: str) -> str:
        """构建分析提示词"""
        
        # 截取前 5000 字
        text_preview = content[:5000] if len(content) > 5000 else content
        
        prompt = f"""请分析这份导学案，提取以下信息。

要求：
1. 输出 JSON 格式（不要输出其他内容）
2. 如果某些信息无法确定，使用合理的默认值

输出格式：
{{
  "topic": "知识点名称（如：牛顿第一定律）",
  "subject": "学科（物理/化学/生物/数学/语文/英语等）",
  "grade": "年级（小学/初中/高中，如果能确定具体年级更好）",
  "difficulty": "难度（简单/中等/困难）",
  "question_types": {{
    "choice": 选择题数量（数字）,
    "fill": 填空题数量（数字）,
    "application": 应用题数量（数字）,
    "other": 其他题型数量（数字）
  }},
  "teaching_sections": [
    "教学环节1（如：导入新课）",
    "教学环节2（如：概念讲解）",
    "教学环节3（如：例题分析）"
  ],
  "key_points": [
    "重点1（一句话概括）",
    "重点2（一句话概括）",
    "重点3（一句话概括）"
  ]
}}

导学案内容：
{text_preview}
"""
        return prompt
    
    def _parse_response(self, content: str, original_content: str) -> LessonAnalysis:
        """解析 LLM 返回的内容"""
        
        try:
            # 提取 JSON（可能包含在 markdown 代码块中）
            json_text = content.strip()
            if json_text.startswith("```json"):
                json_text = json_text[7:]
            if json_text.startswith("```"):
                json_text = json_text[3:]
            if json_text.endswith("```"):
                json_text = json_text[:-3]
            json_text = json_text.strip()
            
            data = json.loads(json_text)
            
            return LessonAnalysis(
                topic=data.get("topic", "未知知识点"),
                subject=data.get("subject", "未知学科"),
                grade=data.get("grade", "未知年级"),
                difficulty=data.get("difficulty", "中等"),
                question_types=data.get("question_types", {
                    "choice": 0,
                    "fill": 0,
                    "application": 0,
                    "other": 0
                }),
                teaching_sections=data.get("teaching_sections", []),
                key_points=data.get("key_points", [])
            )
            
        except (json.JSONDecodeError, KeyError) as e:
            logger.warning(f"lesson_analyzer.parse_failed error={str(e)}")
            # 降级到基础分析
            return self._get_basic_analysis(original_content)
    
    def _get_basic_analysis(self, content: str) -> LessonAnalysis:
        """降级方案：基础分析"""
        
        logger.info("lesson_analyzer.using_basic_analysis")
        
        # 简单的关键词匹配
        topic = "未知知识点"
        subject = "未知学科"
        
        # 尝试识别学科
        subject_keywords = {
            "物理": ["力", "运动", "能量", "电", "磁", "光", "声"],
            "化学": ["化学", "反应", "元素", "分子", "原子", "酸", "碱"],
            "生物": ["生物", "细胞", "遗传", "进化", "生态", "植物", "动物"],
            "数学": ["方程", "函数", "几何", "代数", "概率", "统计"],
            "语文": ["课文", "阅读", "写作", "古诗", "文言文"],
            "英语": ["English", "单词", "语法", "阅读理解"]
        }
        
        for subj, keywords in subject_keywords.items():
            if any(kw in content for kw in keywords):
                subject = subj
                break
        
        # 统计题型
        choice_count = len(re.findall(r'[A-D][\s\.、]', content))
        fill_count = len(re.findall(r'_{3,}|（\s*）', content))
        
        return LessonAnalysis(
            topic=topic,
            subject=subject,
            grade="未知年级",
            difficulty="中等",
            question_types={
                "choice": min(choice_count, 20),  # 限制最大值
                "fill": min(fill_count, 20),
                "application": 0,
                "other": 0
            },
            teaching_sections=["导入新课", "知识讲解", "练习巩固"],
            key_points=["待分析"]
        )
