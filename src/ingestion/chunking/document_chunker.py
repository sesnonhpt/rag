"""Document chunking module - adapts libs.splitter for business layer.

This module serves as the adapter layer between libs.splitter (pure text splitting)
and Ingestion Pipeline (business object transformation). It transforms Document
objects into Chunk objects with proper ID generation, metadata inheritance, and
traceability.

Core Value-Add (vs libs.splitter):
1. Chunk ID Generation: Deterministic and unique IDs for each chunk
2. Metadata Inheritance: Propagates Document metadata to all chunks
3. chunk_index: Records sequential position within document
4. source_ref: Establishes parent-child traceability
5. Type Conversion: str → Chunk object (core.types contract)

Design Principles:
- Adapter Pattern: Bridges text splitter tool with business objects
- Config-Driven: Uses SplitterFactory for configuration-based strategy selection
- Deterministic: Same Document produces same Chunk IDs on repeat splits
- Type-Safe: Enforces core.types.Chunk contract
"""

from __future__ import annotations

import hashlib
import re
from typing import TYPE_CHECKING, Any, Dict, List, Optional, Tuple

from src.core.types import Chunk, Document
from src.libs.splitter.splitter_factory import SplitterFactory

if TYPE_CHECKING:
    from src.core.settings import Settings


class DocumentChunker:
    """Converts Documents into Chunks with business-level enrichment.
    
    This class wraps a text splitter (from libs) and adds business logic:
    - Generates stable chunk IDs
    - Inherits and extends metadata
    - Maintains document traceability
    
    Attributes:
        _splitter: The underlying text splitter from libs layer
        _settings: Configuration settings for chunking behavior
    
    Example:
        >>> from src.core.settings import load_settings
        >>> from src.core.types import Document
        >>> settings = load_settings("config/settings.yaml")
        >>> chunker = DocumentChunker(settings)
        >>> document = Document(
        ...     id="doc_123",
        ...     text="Long document content...",
        ...     metadata={"source_path": "data/report.pdf"}
        ... )
        >>> chunks = chunker.split_document(document)
        >>> print(f"Generated {len(chunks)} chunks")
        >>> print(f"First chunk ID: {chunks[0].id}")
        >>> print(f"First chunk index: {chunks[0].metadata['chunk_index']}")
    """
    
    def __init__(self, settings: Settings):
        """Initialize DocumentChunker with configuration.
        
        Args:
            settings: Configuration settings containing splitter configuration.
                     The splitter config is expected at settings.splitter.*
        
        Raises:
            ValueError: If splitter configuration is invalid or provider unknown
        """
        self._settings = settings
        self._splitter = SplitterFactory.create(settings)
    
    def split_document(self, document: Document) -> List[Chunk]:
        """Split a Document into Chunks with full business enrichment.
        
        This is the main entry point that orchestrates the transformation:
        1. Uses underlying splitter to get text fragments
        2. Generates deterministic IDs for each chunk
        3. Inherits and extends metadata from document
        4. Creates Chunk objects conforming to core.types contract
        
        Args:
            document: Source document to split into chunks
        
        Returns:
            List of Chunk objects with:
            - Unique, deterministic IDs
            - Inherited metadata + chunk_index + source_ref
            - Proper type contract (core.types.Chunk)
        
        Raises:
            ValueError: If document has no text or invalid structure
        
        Example:
            >>> doc = Document(
            ...     id="doc_abc",
            ...     text="Section 1 content.\\n\\nSection 2 content.",
            ...     metadata={"source_path": "file.pdf", "title": "Report"}
            ... )
            >>> chunker = DocumentChunker(settings)
            >>> chunks = chunker.split_document(doc)
            >>> len(chunks) >= 1
            True
            >>> chunks[0].metadata["source_path"]
            'file.pdf'
            >>> chunks[0].metadata["chunk_index"]
            0
            >>> chunks[0].metadata["source_ref"]
            'doc_abc'
        """
        if not document.text or not document.text.strip():
            raise ValueError(f"Document {document.id} has no text content to split")
        
        # Step 1: Choose chunking strategy
        fragment_payloads = self._split_text_fragments(document)
        text_fragments = [payload[0] for payload in fragment_payloads]
        
        if not text_fragments:
            raise ValueError(
                f"Splitter returned no chunks for document {document.id}. "
                f"Text length: {len(document.text)}"
            )
        
        # Step 2: Transform text fragments into Chunk objects with enrichment
        chunks: List[Chunk] = []
        for index, (text, extra_metadata) in enumerate(fragment_payloads):
            chunk_id = self._generate_chunk_id(document.id, index, text)
            chunk_metadata = self._inherit_metadata(
                document,
                index,
                text,
                extra_metadata=extra_metadata,
            )
            
            chunk = Chunk(
                id=chunk_id,
                text=text,
                metadata=chunk_metadata
            )
            chunks.append(chunk)
        
        return chunks

    def _split_text_fragments(self, document: Document) -> List[Tuple[str, Dict[str, Any]]]:
        """Split document text into fragments with optional structure metadata."""
        if self._should_use_long_document_strategy(document):
            structured_fragments = self._split_long_document(document)
            if structured_fragments:
                return structured_fragments

        return [(text, {}) for text in self._splitter.split_text(document.text)]

    def _should_use_long_document_strategy(self, document: Document) -> bool:
        """Enable textbook-style chunking for long paginated PDFs."""
        page_count = document.metadata.get("page_count")
        return (
            isinstance(page_count, int)
            and page_count > self.LONG_DOC_PAGE_THRESHOLD
            and bool(self.PAGE_MARKER_PATTERN.search(document.text))
        )

    def _split_long_document(self, document: Document) -> List[Tuple[str, Dict[str, Any]]]:
        """Split long paginated documents by chapter/section first, then by page/length."""
        pages = self._parse_page_blocks(document.text)
        if not pages:
            return []

        max_chars = max(getattr(self._settings.ingestion, "chunk_size", 1000), 1400)
        current_chapter: Optional[str] = None
        current_section: Optional[str] = None
        fragments: List[Tuple[str, Dict[str, Any]]] = []
        buffer_texts: List[str] = []
        buffer_meta: Optional[Dict[str, Any]] = None

        def flush_buffer() -> None:
            nonlocal buffer_texts, buffer_meta
            if not buffer_texts or buffer_meta is None:
                return
            chunk_text = "\n\n".join(part.strip() for part in buffer_texts if part.strip()).strip()
            if chunk_text:
                fragments.append((chunk_text, buffer_meta.copy()))
            buffer_texts = []
            buffer_meta = None

        for page_num, page_text in pages:
            page_lines = self._clean_page_lines(page_text)
            if self._looks_like_table_of_contents(page_lines):
                flush_buffer()
                fragments.append((
                    page_text.strip(),
                    {
                        "chapter": None,
                        "section": None,
                        "page_start": page_num,
                        "page_end": page_num,
                    },
                ))
                continue

            chapter = self._extract_chapter_title(page_text) or current_chapter
            if chapter and chapter != current_chapter:
                flush_buffer()
                current_chapter = chapter
                current_section = None

            section = None
            if current_chapter is not None or self._extract_chapter_title(page_text):
                section = self._extract_section_title(page_text)
            if section and section != current_section:
                flush_buffer()
                current_section = section

            page_fragments = self._split_page_content(page_text, max_chars=max_chars)
            for fragment_text in page_fragments:
                fragment_meta = {
                    "chapter": current_chapter,
                    "section": current_section,
                    "page_start": page_num,
                    "page_end": page_num,
                }

                if buffer_meta is None:
                    buffer_texts = [fragment_text]
                    buffer_meta = fragment_meta
                    continue

                same_structure = (
                    buffer_meta.get("chapter") == fragment_meta.get("chapter")
                    and buffer_meta.get("section") == fragment_meta.get("section")
                )
                projected_length = len("\n\n".join(buffer_texts + [fragment_text]))

                if same_structure and projected_length <= max_chars:
                    buffer_texts.append(fragment_text)
                    buffer_meta["page_end"] = page_num
                else:
                    flush_buffer()
                    buffer_texts = [fragment_text]
                    buffer_meta = fragment_meta

        flush_buffer()
        return fragments

    def _parse_page_blocks(self, text: str) -> List[Tuple[int, str]]:
        """Parse text rebuilt as `## Page N` blocks."""
        matches = list(self.PAGE_MARKER_PATTERN.finditer(text))
        if not matches:
            return []

        pages: List[Tuple[int, str]] = []
        for index, match in enumerate(matches):
            page_num = int(match.group(1))
            start = match.end()
            end = matches[index + 1].start() if index + 1 < len(matches) else len(text)
            page_text = text[start:end].strip()
            if page_text:
                pages.append((page_num, page_text))
        return pages

    def _extract_chapter_title(self, page_text: str) -> Optional[str]:
        """Extract chapter title from page text."""
        lines = self._clean_page_lines(page_text)
        for index, line in enumerate(lines):
            inline_match = self.CHAPTER_INLINE_PATTERN.match(line)
            if inline_match:
                title = self._normalize_heading(inline_match.group(2).strip())
                return f"{inline_match.group(1)} {title}".strip()

            if self.CHAPTER_LINE_PATTERN.match(line):
                next_line = lines[index + 1] if index + 1 < len(lines) else ""
                if next_line and self._is_heading_like(next_line):
                    return f"{line} {self._normalize_heading(next_line)}".strip()
                return line
        return None

    def _extract_section_title(self, page_text: str) -> Optional[str]:
        """Extract section title from page text using lightweight textbook heuristics."""
        lines = self._clean_page_lines(page_text)
        for index, line in enumerate(lines[:8]):
            section = self._match_section_title(lines, index)
            if section:
                return section
        return None

    def _match_section_title(self, lines: List[str], index: int) -> Optional[str]:
        line = lines[index]
        next_line = lines[index + 1] if index + 1 < len(lines) else ""
        prev_line = lines[index - 1] if index > 0 else ""

        inline_match = self.SECTION_INLINE_PATTERN.match(line)
        if inline_match:
            number = inline_match.group(1)
            title = self._normalize_heading(inline_match.group(2).strip())
            if (
                number.isdigit()
                and 1 <= int(number) <= 9
                and self._is_probable_section_title(title)
            ):
                return f"{inline_match.group(1)}. {title}"

        if (
            self._is_small_section_number(line)
            and next_line
            and self._is_probable_section_title(next_line)
        ):
            return f"{line}. {self._normalize_heading(next_line)}"

        if self._starts_major_section_block(line):
            if self._is_small_section_number(next_line):
                return f"{next_line}. {line}"
            return line

        if (
            self._is_small_section_number(next_line)
            and self._is_probable_section_title(line)
            and not self.CHAPTER_LINE_PATTERN.match(prev_line)
            and not self.CHAPTER_INLINE_PATTERN.match(prev_line)
        ):
            return f"{next_line}. {self._normalize_heading(line)}"

        return None

    def _split_page_content(self, page_text: str, max_chars: int) -> List[str]:
        """Split oversized page text while trying not to cut teaching blocks."""
        page_text = page_text.strip()
        if not page_text:
            return []
        if len(page_text) <= max_chars:
            return [page_text]

        blocks = self._build_semantic_blocks(page_text)
        fragments: List[str] = []
        current_parts: List[str] = []

        def flush_parts() -> None:
            nonlocal current_parts
            if not current_parts:
                return
            combined = "\n\n".join(current_parts).strip()
            if combined:
                fragments.append(combined)
            current_parts = []

        for block in blocks:
            if not current_parts:
                current_parts = [block]
                continue

            projected = "\n\n".join(current_parts + [block])
            if len(projected) <= max_chars:
                current_parts.append(block)
                continue

            flush_parts()
            if len(block) <= max_chars:
                current_parts = [block]
            else:
                for sub_block in self._splitter.split_text(block):
                    fragments.append(sub_block)
                current_parts = []

        flush_parts()
        return fragments

    def _build_semantic_blocks(self, page_text: str) -> List[str]:
        """Convert page text into larger semantic blocks before length splitting."""
        raw_lines = [line.rstrip() for line in page_text.splitlines()]
        blocks: List[str] = []
        current_lines: List[str] = []

        def flush_lines() -> None:
            nonlocal current_lines
            block = "\n".join(line for line in current_lines if line.strip()).strip()
            if block:
                blocks.append(block)
            current_lines = []

        for line in raw_lines:
            stripped = line.strip()
            if not stripped:
                flush_lines()
                continue

            if self._starts_special_block(stripped) and current_lines:
                flush_lines()
            current_lines.append(stripped)

        flush_lines()
        return blocks or [page_text]

    def _clean_page_lines(self, page_text: str) -> List[str]:
        """Normalize page lines and drop obvious running headers/footers."""
        cleaned: List[str] = []
        for raw_line in page_text.splitlines():
            line = " ".join(raw_line.strip().split())
            if not line:
                continue
            if self._is_running_header_or_footer(line):
                continue
            cleaned.append(line)
        return cleaned

    def _is_running_header_or_footer(self, line: str) -> bool:
        """Detect obvious non-content lines that should not drive structure parsing."""
        compact = line.replace(" ", "")
        if "高中物理必修第一册" in compact:
            return True
        return False

    def _starts_special_block(self, line: str) -> bool:
        return any(line.startswith(keyword) for keyword in self.SPECIAL_BLOCK_KEYWORDS)

    def _starts_major_section_block(self, line: str) -> bool:
        return line.startswith("实验：") or line == "实验"

    def _is_heading_like(self, text: str) -> bool:
        """Loose heuristic for short educational headings."""
        candidate = text.strip()
        if not candidate:
            return False
        if len(candidate) > 40:
            return False
        if candidate.endswith(("。", "？", "！", ".", "!", "?")):
            return False
        if " / " in candidate:
            return False
        return True

    def _contains_chinese(self, text: str) -> bool:
        return bool(re.search(r"[\u4e00-\u9fff]", text))

    def _is_small_section_number(self, text: str) -> bool:
        return bool(re.fullmatch(r"[1-9]", text))

    def _normalize_heading(self, text: str) -> str:
        """Normalize OCR'd heading text and strip trailing page-number noise."""
        cleaned = " ".join(text.strip().split())
        cleaned = re.sub(r"(?<=[\u4e00-\u9fff])(\d{2,3})$", "", cleaned).strip()
        cleaned = cleaned.strip(" _-")
        return cleaned

    def _is_probable_section_title(self, text: str) -> bool:
        """Heuristic for true textbook section titles, not body sentences or exercises."""
        candidate = self._normalize_heading(text)
        if not candidate:
            return False
        if not self._is_heading_like(candidate):
            return False
        if not self._contains_chinese(candidate):
            return False
        if len(candidate) > 20:
            return False
        if re.search(r"\d", candidate):
            return False
        if any(mark in candidate for mark in ("，", "。", "？", "！", ",")):
            return False
        if candidate in {"思考与讨论", "问题", "参考案例1", "参考案例2", "实 验"}:
            return False
        if candidate.startswith(("图", "例", "练习", "习题")):
            return False
        if candidate.startswith(("用", "将", "把", "在", "从", "如图", "根据", "设")):
            return False
        if any(token in candidate for token in ("下列", "说法", "观察", "通过", "公里", "km", "m·s", "加速度在")):
            return False
        if candidate.endswith(("的", "了", "是", "为", "在", "哪", "所", "均", "会")):
            return False
        return True

    def _looks_like_table_of_contents(self, lines: List[str]) -> bool:
        """Detect dense chapter/section listing pages such as textbook TOC."""
        if not lines:
            return False
        chapter_hits = sum(
            1
            for line in lines
            if self.CHAPTER_INLINE_PATTERN.match(line) or self.CHAPTER_LINE_PATTERN.match(line)
        )
        section_hits = sum(
            1
            for line in lines
            if self.SECTION_INLINE_PATTERN.match(line)
        )
        return chapter_hits >= 2 and section_hits >= 4
    
    def _generate_chunk_id(self, doc_id: str, index: int, text: str) -> str:
        """Generate unique and deterministic chunk ID.
        
        ID format: {doc_id}_{index:04d}_{content_hash}
        - doc_id: Parent document identifier
        - index: Sequential position (zero-padded to 4 digits)
        - content_hash: First 8 chars of text SHA256 hash
        
        This ensures:
        - Uniqueness: Combination of doc_id + index + content_hash
        - Determinism: Same input always produces same ID
        - Debuggability: Human-readable structure
        
        Args:
            doc_id: Parent document ID
            index: Sequential position of chunk (0-based)
            text: Chunk text content
        
        Returns:
            Unique chunk ID string
        
        Example:
            >>> chunker._generate_chunk_id("doc_123", 0, "Hello world")
            'doc_123_0000_c0535e4b'
        """
        # Compute content hash for uniqueness
        content_hash = hashlib.sha256(text.encode("utf-8")).hexdigest()[:8]
        
        # Format: {doc_id}_{index:04d}_{hash_8chars}
        return f"{doc_id}_{index:04d}_{content_hash}"
    
    def _inherit_metadata(
        self,
        document: Document,
        chunk_index: int,
        chunk_text: str = "",
        extra_metadata: Optional[Dict[str, Any]] = None,
    ) -> dict:
        """Inherit metadata from document and add chunk-specific fields.
        
        This creates a new metadata dict containing:
        - All fields from document.metadata (copied, not referenced)
        - chunk_index: Sequential position (0-based)
        - source_ref: Reference to parent document ID
        - image_refs: List of image IDs referenced in this chunk (extracted from placeholders)
        
        Note: The document-level 'images' field is intentionally excluded from chunk
        metadata as it would be redundant. Instead, chunk-specific 'image_refs' is
        populated based on [IMAGE: xxx] placeholders found in the chunk text.
        
        Args:
            document: Source document whose metadata to inherit
            chunk_index: Sequential position of this chunk
            chunk_text: The text content of this chunk (used to extract image_refs)
        
        Returns:
            Metadata dict with inherited and chunk-specific fields
        
        Example:
            >>> doc = Document(
            ...     id="doc_123",
            ...     text="Content",
            ...     metadata={"source_path": "file.pdf", "title": "Report"}
            ... )
            >>> metadata = chunker._inherit_metadata(doc, 2, "See [IMAGE: img_001]")
            >>> metadata["source_path"]
            'file.pdf'
            >>> metadata["chunk_index"]
            2
            >>> metadata["source_ref"]
            'doc_123'
            >>> metadata["image_refs"]
            ['img_001']
        """
        import re
        
        # Copy all document metadata (shallow copy is sufficient for primitives)
        chunk_metadata = document.metadata.copy()
        
        # Get document-level images for lookup
        doc_images = document.metadata.get("images", [])
        
        # Remove document-level 'images' field - we'll add chunk-specific images below
        chunk_metadata.pop("images", None)
        
        # Add chunk-specific fields
        chunk_metadata["chunk_index"] = chunk_index
        chunk_metadata["source_ref"] = document.id
        if extra_metadata:
            chunk_metadata.update({k: v for k, v in extra_metadata.items() if v is not None})
        
        # Extract image_refs from chunk text by finding [IMAGE: xxx] placeholders
        image_refs = []
        if chunk_text:
            # Pattern matches [IMAGE: image_id] placeholders
            pattern = r'\[IMAGE:\s*([^\]]+)\]'
            matches = re.findall(pattern, chunk_text)
            image_refs = [m.strip() for m in matches]
        
        chunk_metadata["image_refs"] = image_refs
        
        # Build chunk-specific 'images' list with full metadata for referenced images
        # This is needed by ImageCaptioner to access image paths for Vision API calls
        chunk_images = []
        if image_refs and doc_images:
            image_lookup = {img.get("id"): img for img in doc_images}
            for img_id in image_refs:
                if img_id in image_lookup:
                    chunk_images.append(image_lookup[img_id])
        
        if chunk_images:
            chunk_metadata["images"] = chunk_images
        
        # Try to determine page_num from the first referenced image
        if chunk_images:
            chunk_metadata["page_num"] = chunk_images[0].get("page")
        elif "page_start" in chunk_metadata:
            chunk_metadata["page_num"] = chunk_metadata["page_start"]

        return chunk_metadata
    LONG_DOC_PAGE_THRESHOLD = 50
    PAGE_MARKER_PATTERN = re.compile(r"^## Page (\d+)\s*$", re.MULTILINE)
    CHAPTER_INLINE_PATTERN = re.compile(
        r"^(第[一二三四五六七八九十百零0-9]+章)\s+(.+)$"
    )
    CHAPTER_LINE_PATTERN = re.compile(r"^第[一二三四五六七八九十百零0-9]+章$")
    SECTION_INLINE_PATTERN = re.compile(r"^([1-9])[.．、]\s*(.+)$")
    SPECIAL_BLOCK_KEYWORDS = (
        "实验",
        "定义",
        "概念",
        "规律",
        "定律",
        "例题",
        "问题",
        "小结",
        "练习与应用",
        "科学漫步",
        "思考与讨论",
        "做一做",
        "STEM",
    )
