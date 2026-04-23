"""File parser service for converting doc/docx/pdf to HTML."""

from __future__ import annotations

import base64
import subprocess
import tempfile
from pathlib import Path
from typing import Dict, List, Optional, Any

from src.observability.logger import get_logger

logger = get_logger(__name__)


class ParseResult:
    """Result of file parsing."""
    
    def __init__(
        self,
        html_content: str,
        images: List[Dict[str, str]],
        metadata: Dict[str, Any]
    ):
        self.html_content = html_content
        self.images = images
        self.metadata = metadata


class FileParserService:
    """Parse template files to HTML."""
    
    def __init__(self, image_storage: Any = None):
        self.image_storage = image_storage
    
    def parse_file(self, file_path: Path) -> ParseResult:
        """Parse a file based on its extension."""
        suffix = file_path.suffix.lower()
        
        if suffix == '.docx':
            return self.parse_docx(file_path)
        elif suffix == '.pdf':
            return self.parse_pdf(file_path)
        elif suffix == '.doc':
            return self.parse_doc(file_path)
        else:
            raise ValueError(f"Unsupported file format: {suffix}")
    
    def parse_docx(self, file_path: Path) -> ParseResult:
        """
        Parse .docx file using mammoth with better formatting preservation.
        Extracts images and embeds them as base64 data URLs.
        
        Returns:
            ParseResult with html_content, images, metadata
        """
        try:
            import mammoth
        except ImportError:
            raise RuntimeError("mammoth library not installed. Run: pip install mammoth")
        
        try:
            # Custom style map for better Word document conversion
            style_map = """
                p[style-name='Heading 1'] => h1:fresh
                p[style-name='Heading 2'] => h2:fresh
                p[style-name='Heading 3'] => h3:fresh
                p[style-name='标题 1'] => h1:fresh
                p[style-name='标题 2'] => h2:fresh
                p[style-name='标题 3'] => h3:fresh
                p[style-name='Title'] => h1.title:fresh
                p[style-name='Subtitle'] => h2.subtitle:fresh
                r[style-name='Strong'] => strong
                r[style-name='Emphasis'] => em
                table => table.docx-table
                tr => tr
                td => td
            """
            
            # Image counter for tracking
            image_count = [0]
            
            def convert_image(image):
                """Convert image to base64 data URL."""
                try:
                    image_count[0] += 1
                    # Read image bytes using open() method
                    with image.open() as image_stream:
                        image_bytes = image_stream.read()
                    
                    # Determine content type
                    content_type = image.content_type or 'image/png'
                    
                    # Convert to base64
                    base64_data = base64.b64encode(image_bytes).decode('utf-8')
                    
                    # Create data URL
                    data_url = f"data:{content_type};base64,{base64_data}"
                    
                    logger.info(f"Converted image {image_count[0]}: {content_type}, size: {len(image_bytes)} bytes")
                    
                    return {"src": data_url}
                except Exception as e:
                    logger.error(f"Failed to convert image: {e}")
                    return {"src": ""}
            
            with open(file_path, "rb") as docx_file:
                result = mammoth.convert_to_html(
                    docx_file,
                    style_map=style_map,
                    include_default_style_map=True,
                    convert_image=mammoth.images.img_element(convert_image)
                )
                html = result.value
                messages = result.messages
            
            # Post-process HTML to improve formatting
            html = self._improve_html_formatting(html)
            
            # Log any warnings
            for msg in messages:
                logger.warning(f"mammoth warning: {msg}")
            
            # Extract metadata
            metadata = {
                "parser": "mammoth",
                "warnings": len(messages),
                "images": image_count[0]
            }
            
            # Images are embedded in HTML as base64
            images = []
            
            logger.info(f"Parsed DOCX: {file_path.name}, HTML length: {len(html)}, images: {image_count[0]}")
            
            return ParseResult(
                html_content=html,
                images=images,
                metadata=metadata
            )
        
        except Exception as e:
            logger.exception(f"Failed to parse DOCX: {file_path}")
            raise RuntimeError(f"DOCX parsing failed: {str(e)}")
    
    def _improve_html_formatting(self, html: str) -> str:
        """
        Improve HTML formatting to better match Word document appearance.
        """
        from bs4 import BeautifulSoup
        
        soup = BeautifulSoup(html, 'html.parser')
        
        # Add classes to tables for better styling
        for table in soup.find_all('table'):
            table['class'] = table.get('class', []) + ['docx-table']
            table['border'] = '1'
            table['cellpadding'] = '8'
            table['cellspacing'] = '0'
        
        # Preserve bold text
        for strong in soup.find_all('strong'):
            strong['style'] = 'font-weight: bold;'
        
        # Preserve underlines
        for u in soup.find_all('u'):
            u['style'] = 'text-decoration: underline;'
        
        # Style images for better display
        for img in soup.find_all('img'):
            # Add responsive image styling
            img['style'] = 'max-width: 100%; height: auto; display: block; margin: 16px 0;'
            # Add alt text if missing
            if not img.get('alt'):
                img['alt'] = '文档图片'
        
        return str(soup)
    
    def parse_pdf(self, file_path: Path) -> ParseResult:
        """
        Parse PDF using pdfplumber.
        
        Returns:
            ParseResult with html_content, images, metadata
        """
        try:
            import pdfplumber
        except ImportError:
            raise RuntimeError("pdfplumber library not installed. Run: pip install pdfplumber")
        
        try:
            text_parts = []
            page_count = 0
            
            with pdfplumber.open(file_path) as pdf:
                page_count = len(pdf.pages)
                
                for page in pdf.pages:
                    text = page.extract_text()
                    if text:
                        text_parts.append(text)
            
            # Convert text to simple HTML
            html_parts = []
            for i, text in enumerate(text_parts):
                if i == 0:
                    # First page might be title
                    lines = text.split('\n')
                    if lines:
                        html_parts.append(f'<h1>{lines[0]}</h1>')
                        html_parts.append('<p>' + '<br>'.join(lines[1:]) + '</p>')
                else:
                    html_parts.append('<p>' + text.replace('\n', '<br>') + '</p>')
            
            html = '\n'.join(html_parts)
            
            metadata = {
                "parser": "pdfplumber",
                "page_count": page_count
            }
            
            # TODO: Extract images from PDF
            images = []
            
            logger.info(f"Parsed PDF: {file_path.name}, pages: {page_count}, HTML length: {len(html)}")
            
            return ParseResult(
                html_content=html,
                images=images,
                metadata=metadata
            )
        
        except Exception as e:
            logger.exception(f"Failed to parse PDF: {file_path}")
            raise RuntimeError(f"PDF parsing failed: {str(e)}")
    
    def parse_doc(self, file_path: Path) -> ParseResult:
        """
        Parse legacy .doc format by converting to .docx first.
        
        Requires LibreOffice to be installed.
        
        Returns:
            ParseResult with html_content, images, metadata
        """
        try:
            # Create temp directory for conversion
            with tempfile.TemporaryDirectory() as temp_dir:
                temp_path = Path(temp_dir)
                
                # Convert .doc to .docx using LibreOffice
                result = subprocess.run([
                    "libreoffice",
                    "--headless",
                    "--convert-to", "docx",
                    str(file_path),
                    "--outdir", str(temp_path)
                ], capture_output=True, text=True, timeout=30)
                
                if result.returncode != 0:
                    raise RuntimeError(f"LibreOffice conversion failed: {result.stderr}")
                
                # Find converted file
                docx_path = temp_path / f"{file_path.stem}.docx"
                if not docx_path.exists():
                    raise RuntimeError("Converted DOCX file not found")
                
                # Parse the converted DOCX
                parse_result = self.parse_docx(docx_path)
                parse_result.metadata["parser"] = "libreoffice + mammoth"
                
                logger.info(f"Parsed DOC (via conversion): {file_path.name}")
                
                return parse_result
        
        except subprocess.TimeoutExpired:
            raise RuntimeError("LibreOffice conversion timed out")
        except FileNotFoundError:
            raise RuntimeError("LibreOffice not found. Please install: apt-get install libreoffice-writer")
        except Exception as e:
            logger.exception(f"Failed to parse DOC: {file_path}")
            raise RuntimeError(f"DOC parsing failed: {str(e)}")
