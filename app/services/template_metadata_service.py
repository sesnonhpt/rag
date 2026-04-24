"""Template metadata service for indexing and searching templates."""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Dict, List, Optional
from datetime import datetime

from src.observability.logger import get_logger
from app.services.file_parser_service import FileParserService

logger = get_logger(__name__)


class TemplateMetadata:
    """Metadata for a template file."""
    
    def __init__(
        self,
        filename: str,
        desc: str = "",
        keywords: List[str] = None,
        content_preview: str = "",
        indexed_at: str = None,
        file_modified_at: float = None
    ):
        self.filename = filename
        self.desc = desc
        self.keywords = keywords or []
        self.content_preview = content_preview
        self.indexed_at = indexed_at or datetime.now().isoformat()
        self.file_modified_at = file_modified_at
    
    def to_dict(self) -> dict:
        return {
            "filename": self.filename,
            "desc": self.desc,
            "keywords": self.keywords,
            "content_preview": self.content_preview,
            "indexed_at": self.indexed_at,
            "file_modified_at": self.file_modified_at
        }
    
    @classmethod
    def from_dict(cls, data: dict) -> TemplateMetadata:
        return cls(**data)


class TemplateMetadataService:
    """Service for managing template metadata index."""
    
    def __init__(self, templates_dir: Path, metadata_file: Path = None, use_llm: bool = True):
        self.templates_dir = templates_dir
        self.metadata_file = metadata_file or templates_dir / ".metadata_index.json"
        self.parser = FileParserService()
        self.use_llm = use_llm
        self._index: Dict[str, TemplateMetadata] = {}
        self._llm = None
        
        # Initialize LLM if enabled
        if self.use_llm:
            self._init_llm()
        
        self._load_index()
    
    def _init_llm(self):
        """Initialize MiniMax LLM for metadata generation."""
        try:
            from src.libs.llm.openai_llm import OpenAILLM
            from types import SimpleNamespace
            
            # Get MiniMax config from environment
            api_key = os.environ.get("MINIMAX_API_KEY")
            base_url = os.environ.get("MINIMAX_API_URL", "https://api.minimax.io/v1")
            
            if not api_key:
                logger.warning("MINIMAX_API_KEY not found, LLM metadata generation disabled")
                self.use_llm = False
                return
            
            # Create a minimal settings object for OpenAILLM
            settings = SimpleNamespace(
                llm=SimpleNamespace(
                    model="abab6.5s-chat",  # MiniMax model
                    temperature=0.7,
                    max_tokens=500,
                    api_key=api_key
                )
            )
            
            self._llm = OpenAILLM(settings, base_url=base_url)
            
            # Test the connection
            from src.libs.llm.base_llm import Message
            test_msg = [Message(role="user", content="测试")]
            self._llm.chat(test_msg, max_tokens=10)
            
            logger.info("MiniMax LLM initialized and tested successfully")
        except Exception as e:
            logger.warning(f"Failed to initialize LLM, using rule-based fallback: {e}")
            self.use_llm = False
    
    def _load_index(self):
        """Load metadata index from file."""
        if self.metadata_file.exists():
            try:
                with open(self.metadata_file, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                    self._index = {
                        k: TemplateMetadata.from_dict(v) 
                        for k, v in data.items()
                    }
                logger.info(f"Loaded {len(self._index)} metadata entries")
            except Exception as e:
                logger.error(f"Failed to load metadata index: {e}")
                self._index = {}
        else:
            self._index = {}
    
    def _save_index(self):
        """Save metadata index to file."""
        try:
            with open(self.metadata_file, 'w', encoding='utf-8') as f:
                data = {k: v.to_dict() for k, v in self._index.items()}
                json.dump(data, f, ensure_ascii=False, indent=2)
            logger.info(f"Saved {len(self._index)} metadata entries")
        except Exception as e:
            logger.error(f"Failed to save metadata index: {e}")
    
    def _extract_text_from_html(self, html: str, max_length: int = 500) -> str:
        """Extract plain text from HTML for preview."""
        try:
            from bs4 import BeautifulSoup
            soup = BeautifulSoup(html, 'html.parser')
            text = soup.get_text(separator=' ', strip=True)
            # Clean up whitespace
            text = ' '.join(text.split())
            return text[:max_length] + ('...' if len(text) > max_length else '')
        except Exception as e:
            logger.error(f"Failed to extract text: {e}")
            return ""
    
    def _generate_keywords_rule_based(self, filename: str, content: str) -> List[str]:
        """Generate keywords using rule-based extraction (fallback)."""
        keywords = []
        
        # Extract from filename
        name_parts = Path(filename).stem.replace('.', ' ').replace('_', ' ').split()
        keywords.extend(name_parts)
        
        # Extract common educational terms from content
        educational_terms = [
            '年级', '单元', '课时', '教学', '学习', '练习', '作业',
            '导学案', '教案', '试卷', '测试', '复习', '预习',
            '数学', '语文', '英语', '科学', '物理', '化学', '生物'
        ]
        
        content_lower = content.lower()
        for term in educational_terms:
            if term in content_lower:
                keywords.append(term)
        
        # Remove duplicates and return
        return list(set(keywords))
    
    def _generate_keywords_with_llm(self, filename: str, content: str) -> List[str]:
        """Generate keywords using MiniMax LLM."""
        from src.libs.llm.base_llm import Message
        
        # Truncate content to avoid token limits
        content_sample = content[:1000] if len(content) > 1000 else content
        
        prompt = f"""请分析以下教学文档，提取5-10个关键词。关键词应该包括：
1. 学科（如：数学、语文、英语等）
2. 年级（如：六年级、五年级等）
3. 主题（如：方程、分数、单元等）
4. 文档类型（如：导学案、教案、试卷等）

文件名：{filename}
文档内容：
{content_sample}

请只返回关键词列表，用逗号分隔，不要其他解释。"""

        messages = [Message(role="user", content=prompt)]
        response = self._llm.chat(messages, temperature=0.3, max_tokens=200)
        
        # Parse keywords from response
        keywords_text = response.content.strip()
        keywords = [k.strip() for k in keywords_text.split(',') if k.strip()]
        
        logger.info(f"LLM generated {len(keywords)} keywords for {filename}")
        return keywords[:10]  # Limit to 10 keywords
    
    def _generate_metadata_with_llm(self, filename: str, content: str) -> tuple[str, List[str]]:
        """Generate both description and keywords using MiniMax LLM."""
        from src.libs.llm.base_llm import Message
        
        # Truncate content to avoid token limits
        content_sample = content[:1500] if len(content) > 1500 else content
        
        prompt = f"""请分析以下教学文档，生成：
1. 一句话描述（50字以内，说明这是什么文档、适用年级、主题）
2. 5-10个关键词（用逗号分隔）

文件名：{filename}
文档内容：
{content_sample}

请按以下格式返回：
描述：[一句话描述]
关键词：[关键词1,关键词2,关键词3,...]"""

        messages = [Message(role="user", content=prompt)]
        response = self._llm.chat(messages, temperature=0.3, max_tokens=300)
        
        # Parse response
        lines = response.content.strip().split('\n')
        desc = ""
        keywords = []
        
        for line in lines:
            if line.startswith('描述：') or line.startswith('描述:'):
                desc = line.split('：', 1)[-1].split(':', 1)[-1].strip()
            elif line.startswith('关键词：') or line.startswith('关键词:'):
                keywords_text = line.split('：', 1)[-1].split(':', 1)[-1].strip()
                keywords = [k.strip() for k in keywords_text.split(',') if k.strip()]
        
        # Fallback if parsing failed
        if not desc:
            desc = content[:100] + '...'
        if not keywords:
            keywords = self._generate_keywords_rule_based(filename, content)
        
        logger.info(f"LLM generated metadata for {filename}: desc={len(desc)} chars, keywords={len(keywords)}")
        return desc, keywords[:10]
    
    def index_file(self, file_path: Path, force: bool = False) -> Optional[TemplateMetadata]:
        """
        Index a single template file.
        
        Args:
            file_path: Path to the template file
            force: Force re-indexing even if already indexed
        
        Returns:
            TemplateMetadata or None if indexing failed
        """
        try:
            relative_path = file_path.relative_to(self.templates_dir)
            filename = str(relative_path)
            
            # Check if already indexed and file hasn't changed
            file_mtime = file_path.stat().st_mtime
            if not force and filename in self._index:
                existing = self._index[filename]
                if existing.file_modified_at == file_mtime:
                    logger.debug(f"Skipping unchanged file: {filename}")
                    return existing
            
            logger.info(f"Indexing file: {filename}")
            
            # Parse file content
            parse_result = self.parser.parse_file(file_path)
            
            # Extract text preview
            content_preview = self._extract_text_from_html(parse_result.html_content)
            
            # Generate description and keywords using LLM
            if self.use_llm and self._llm:
                try:
                    desc, keywords = self._generate_metadata_with_llm(filename, content_preview)
                except Exception as e:
                    logger.error(f"LLM metadata generation failed, using fallback: {e}")
                    desc = content_preview[:200] + ('...' if len(content_preview) > 200 else '')
                    keywords = self._generate_keywords_rule_based(filename, content_preview)
            else:
                # Fallback to simple extraction
                desc = content_preview[:200] + ('...' if len(content_preview) > 200 else '')
                keywords = self._generate_keywords_rule_based(filename, content_preview)
            
            # Create metadata
            metadata = TemplateMetadata(
                filename=filename,
                desc=desc,
                keywords=keywords,
                content_preview=content_preview,
                file_modified_at=file_mtime
            )
            
            # Update index
            self._index[filename] = metadata
            self._save_index()
            
            logger.info(f"Indexed: {filename}, keywords: {len(keywords)}")
            return metadata
        
        except Exception as e:
            logger.error(f"Failed to index {file_path}: {e}")
            return None
    
    def index_all(self, force: bool = False):
        """
        Index all template files in the directory.
        
        Args:
            force: Force re-indexing of all files
        """
        logger.info(f"Starting indexing of {self.templates_dir}")
        
        indexed_count = 0
        skipped_count = 0
        failed_count = 0
        
        for file_path in self.templates_dir.rglob('*'):
            if not file_path.is_file():
                continue
            
            # Skip hidden files and README
            if file_path.name.startswith('.') or file_path.name == 'README.md':
                continue
            
            # Only index supported formats
            if file_path.suffix.lower() not in ['.docx', '.doc', '.pdf']:
                continue
            
            result = self.index_file(file_path, force=force)
            if result:
                indexed_count += 1
            elif result is None:
                failed_count += 1
            else:
                skipped_count += 1
        
        logger.info(
            f"Indexing complete: {indexed_count} indexed, "
            f"{skipped_count} skipped, {failed_count} failed"
        )
    
    def get_metadata(self, filename: str) -> Optional[TemplateMetadata]:
        """Get metadata for a specific file."""
        return self._index.get(filename)
    
    def search(
        self,
        query: str,
        search_filename: bool = True,
        search_metadata: bool = True
    ) -> List[tuple[str, float]]:
        """
        Search templates by query.
        
        Args:
            query: Search query
            search_filename: Search in filename
            search_metadata: Search in desc and keywords
        
        Returns:
            List of (filename, relevance_score) tuples, sorted by score
        """
        if not query:
            return []
        
        query_lower = query.lower()
        query_terms = query_lower.split()
        
        results = []
        
        for filename, metadata in self._index.items():
            score = 0.0
            
            # Layer 1: Filename search
            if search_filename:
                filename_lower = filename.lower()
                for term in query_terms:
                    if term in filename_lower:
                        # Exact match
                        if term == filename_lower:
                            score += 10
                        # Starts with
                        elif filename_lower.startswith(term):
                            score += 5
                        # Contains
                        else:
                            score += 2
            
            # Layer 2: Metadata search
            if search_metadata:
                # Search in description
                desc_lower = metadata.desc.lower()
                for term in query_terms:
                    if term in desc_lower:
                        score += 3
                
                # Search in keywords (higher weight)
                for keyword in metadata.keywords:
                    keyword_lower = keyword.lower()
                    for term in query_terms:
                        if term in keyword_lower:
                            score += 4
            
            if score > 0:
                results.append((filename, score))
        
        # Sort by score (descending)
        results.sort(key=lambda x: x[1], reverse=True)
        
        return results
    
    def update_metadata(
        self,
        filename: str,
        desc: Optional[str] = None,
        keywords: Optional[List[str]] = None
    ):
        """
        Manually update metadata for a file.
        
        Args:
            filename: Template filename
            desc: New description (optional)
            keywords: New keywords (optional)
        """
        if filename not in self._index:
            logger.warning(f"File not in index: {filename}")
            return
        
        metadata = self._index[filename]
        
        if desc is not None:
            metadata.desc = desc
        
        if keywords is not None:
            metadata.keywords = keywords
        
        self._save_index()
        logger.info(f"Updated metadata for: {filename}")
