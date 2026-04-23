"""Template export service for converting HTML to various formats."""

from __future__ import annotations

import base64
import tempfile
from io import BytesIO
from pathlib import Path
from typing import Optional, Callable

from app.services.docx_export_service import build_lesson_docx_bytes
from src.observability.logger import get_logger

logger = get_logger(__name__)


class TemplateExportService:
    """Export template content to various formats."""
    
    def __init__(self, image_storage_path: Optional[Path] = None):
        """
        Initialize export service.
        
        Args:
            image_storage_path: Path to image storage directory
        """
        self.image_storage_path = image_storage_path or Path("data/templates/images")
    
    def export_to_docx(
        self,
        content_html: str,
        resolve_image_path: Optional[Callable[[str], Optional[Path]]] = None,
    ) -> bytes:
        """
        Export HTML content to DOCX format.
        
        Args:
            content_html: HTML content to export
            resolve_image_path: Optional function to resolve image paths
        
        Returns:
            DOCX file bytes
        """
        if resolve_image_path is None:
            resolve_image_path = self._default_resolve_image_path
        
        try:
            docx_bytes = build_lesson_docx_bytes(
                content_html=content_html,
                resolve_image_path=resolve_image_path,
                resolve_image_bytes=self._resolve_image_bytes,
            )
            logger.info(f"Exported to DOCX, size: {len(docx_bytes)} bytes")
            return docx_bytes
        except Exception as e:
            logger.exception("Failed to export to DOCX")
            raise RuntimeError(f"DOCX 导出失败: {str(e)}")
    
    def export_to_pdf(self, content_html: str) -> bytes:
        """
        Export HTML content to PDF format using WeasyPrint.
        
        Args:
            content_html: HTML content to export
        
        Returns:
            PDF file bytes
        """
        try:
            from weasyprint import HTML, CSS
        except ImportError:
            raise RuntimeError(
                "WeasyPrint 未安装。请运行: pip install weasyprint"
            )
        
        try:
            # Add CSS styling for better PDF output
            css_style = CSS(string="""
                @page {
                    size: A4;
                    margin: 2cm;
                }
                body {
                    font-family: "SimSun", "Times New Roman", serif;
                    font-size: 12pt;
                    line-height: 1.5;
                    color: #000;
                }
                h1 {
                    font-family: "SimHei", "Arial", sans-serif;
                    font-size: 16pt;
                    font-weight: bold;
                    text-align: center;
                    margin-top: 0;
                    margin-bottom: 18pt;
                }
                h2 {
                    font-family: "SimHei", "Arial", sans-serif;
                    font-size: 14pt;
                    font-weight: bold;
                    margin-top: 12pt;
                    margin-bottom: 6pt;
                }
                h3 {
                    font-size: 12pt;
                    font-weight: bold;
                    margin-top: 10pt;
                    margin-bottom: 4pt;
                }
                p {
                    margin-top: 0;
                    margin-bottom: 6pt;
                }
                img {
                    max-width: 100%;
                    height: auto;
                    display: block;
                    margin: 10pt auto;
                }
                ul, ol {
                    margin-top: 6pt;
                    margin-bottom: 6pt;
                }
            """)
            
            # Create HTML document
            html_doc = HTML(string=content_html)
            
            # Generate PDF
            pdf_bytes = html_doc.write_pdf(stylesheets=[css_style])
            
            logger.info(f"Exported to PDF, size: {len(pdf_bytes)} bytes")
            return pdf_bytes
        
        except Exception as e:
            logger.exception("Failed to export to PDF")
            raise RuntimeError(f"PDF 导出失败: {str(e)}")
    
    def export_to_markdown(self, content_html: str) -> str:
        """
        Export HTML content to Markdown format.
        
        Args:
            content_html: HTML content to export
        
        Returns:
            Markdown text
        """
        try:
            import html2text
        except ImportError:
            raise RuntimeError(
                "html2text 未安装。请运行: pip install html2text"
            )
        
        try:
            # Configure html2text
            h = html2text.HTML2Text()
            h.ignore_links = False
            h.ignore_images = False
            h.ignore_emphasis = False
            h.body_width = 0  # Don't wrap lines
            h.unicode_snob = True  # Use unicode characters
            h.skip_internal_links = True
            
            # Convert HTML to Markdown
            markdown_text = h.handle(content_html)
            
            logger.info(f"Exported to Markdown, length: {len(markdown_text)} chars")
            return markdown_text
        
        except Exception as e:
            logger.exception("Failed to export to Markdown")
            raise RuntimeError(f"Markdown 导出失败: {str(e)}")
    
    def _default_resolve_image_path(self, src: str) -> Optional[Path]:
        """
        Default image path resolver.
        
        Args:
            src: Image source URL or path
        
        Returns:
            Path to image file if exists, None otherwise
        """
        # Handle absolute paths
        if src.startswith('/'):
            src = src.lstrip('/')
        
        # Try to resolve from image storage
        image_path = self.image_storage_path / src
        if image_path.exists():
            return image_path
        
        # Try to resolve as relative path
        relative_path = Path(src)
        if relative_path.exists():
            return relative_path
        
        logger.warning(f"Image not found: {src}")
        return None

    def _resolve_image_bytes(self, src: str) -> Optional[bytes]:
        """
        Resolve embedded data URLs to raw bytes for DOCX export.
        """
        if not src.startswith("data:"):
            return None

        try:
            _, encoded = src.split(",", 1)
        except ValueError:
            logger.warning("Invalid data URL image source")
            return None

        try:
            return base64.b64decode(encoded)
        except Exception as exc:
            logger.warning("Failed to decode embedded image: %r", exc)
            return None
