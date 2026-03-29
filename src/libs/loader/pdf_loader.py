"""PDF Loader implementation using MarkItDown.

This module implements PDF parsing with image extraction support,
converting PDFs to standardized Markdown format with image placeholders.

Features:
- Text extraction and Markdown conversion via MarkItDown
- Image extraction and storage
- Image placeholder insertion with metadata tracking
- Graceful degradation if image extraction fails
"""

from __future__ import annotations

import hashlib
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional

try:
    from markitdown import MarkItDown
    MARKITDOWN_AVAILABLE = True
except ImportError:
    MARKITDOWN_AVAILABLE = False

try:
    import fitz  # PyMuPDF
    PYMUPDF_AVAILABLE = True
except ImportError:
    PYMUPDF_AVAILABLE = False

from PIL import Image
import io

from src.core.types import Document
from src.libs.loader.base_loader import BaseLoader

logger = logging.getLogger(__name__)


class PdfLoader(BaseLoader):
    """PDF Loader using MarkItDown for text extraction and Markdown conversion.
    
    This loader:
    1. Extracts text from PDF and converts to Markdown
    2. Extracts images and saves to data/images/{doc_hash}/
    3. Inserts image placeholders in the format [IMAGE: {image_id}]
    4. Records image metadata in Document.metadata.images
    
    Configuration:
        extract_images: Enable/disable image extraction (default: True)
        image_storage_dir: Base directory for image storage (default: data/images)
    
    Graceful Degradation:
        If image extraction fails, logs warning and continues with text-only parsing.
    """
    
    def __init__(
        self,
        extract_images: bool = True,
        image_storage_dir: str | Path = "data/images"
    ):
        """Initialize PDF Loader.
        
        Args:
            extract_images: Whether to extract images from PDFs.
            image_storage_dir: Base directory for storing extracted images.
        """
        if not MARKITDOWN_AVAILABLE:
            raise ImportError(
                "MarkItDown is required for PdfLoader. "
                "Install with: pip install markitdown"
            )
        
        self.extract_images = extract_images
        self.image_storage_dir = Path(image_storage_dir)
        self._markitdown = MarkItDown()
    
    def load(self, file_path: str | Path) -> Document:
        """Load and parse a PDF file.
        
        Args:
            file_path: Path to the PDF file.
            
        Returns:
            Document with Markdown text and metadata.
            
        Raises:
            FileNotFoundError: If the PDF file doesn't exist.
            ValueError: If the file is not a valid PDF.
            RuntimeError: If parsing fails critically.
        """
        # Validate file
        path = self._validate_file(file_path)
        if path.suffix.lower() != '.pdf':
            raise ValueError(f"File is not a PDF: {path}")
        
        # Compute document hash for unique ID and image directory
        doc_hash = self._compute_file_hash(path)
        doc_id = f"doc_{doc_hash[:16]}"
        
        # Parse PDF with MarkItDown
        try:
            result = self._markitdown.convert(str(path))
            text_content = result.text_content if hasattr(result, 'text_content') else str(result)
        except Exception as e:
            logger.error(f"Failed to parse PDF {path}: {e}")
            raise RuntimeError(f"PDF parsing failed: {e}") from e
        
        # Initialize metadata
        metadata: Dict[str, Any] = {
            "source_path": str(path),
            "doc_type": "pdf",
            "doc_hash": doc_hash,
        }
        
        # Extract title from first heading if available
        title = self._extract_title(text_content)
        if title:
            metadata["title"] = title
        
        # Handle image extraction (with graceful degradation)
        if self.extract_images:
            try:
                text_content, images_metadata = self._extract_and_process_images(
                    path, text_content, doc_hash
                )
                if images_metadata:
                    metadata["images"] = images_metadata
            except Exception as e:
                logger.warning(
                    f"Image extraction failed for {path}, continuing with text-only: {e}"
                )
        
        return Document(
            id=doc_id,
            text=text_content,
            metadata=metadata
        )
    
    def _compute_file_hash(self, file_path: Path) -> str:
        """Compute SHA256 hash of file content.
        
        Args:
            file_path: Path to file.
            
        Returns:
            Hex string of SHA256 hash.
        """
        sha256 = hashlib.sha256()
        with open(file_path, 'rb') as f:
            for chunk in iter(lambda: f.read(8192), b''):
                sha256.update(chunk)
        return sha256.hexdigest()
    
    def _extract_title(self, text: str) -> Optional[str]:
        """Extract title from first Markdown heading or first non-empty line.
        
        Args:
            text: Markdown text content.
            
        Returns:
            Title string if found, None otherwise.
        """
        lines = text.split('\n')
        
        # First try to find a markdown heading
        for line in lines[:20]:  # Check first 20 lines
            line = line.strip()
            if line.startswith('# '):
                return line[2:].strip()
        
        # Fallback: use first non-empty line as title
        for line in lines[:10]:
            line = line.strip()
            if line and len(line) > 0:
                return line
        
        return None
    
    def _extract_and_process_images(
        self,
        pdf_path: Path,
        text_content: str,
        doc_hash: str
    ) -> tuple[str, List[Dict[str, Any]]]:
        """Extract images from PDF and insert placeholders.
        
        Uses PyMuPDF to extract images, save them to disk, and insert
        placeholders in the text content.
        
        Args:
            pdf_path: Path to PDF file.
            text_content: Extracted text content.
            doc_hash: Document hash for image directory.
            
        Returns:
            Tuple of (modified_text, images_metadata_list)
        """
        if not self.extract_images:
            logger.debug(f"Image extraction disabled for {pdf_path}")
            return text_content, []
        
        if not PYMUPDF_AVAILABLE:
            logger.warning(f"PyMuPDF not available, skipping image extraction for {pdf_path}")
            return text_content, []
        
        images_metadata = []
        modified_text = text_content
        
        try:
            # Create image storage directory
            image_dir = self.image_storage_dir / doc_hash
            image_dir.mkdir(parents=True, exist_ok=True)
            
            # Open PDF with PyMuPDF and rebuild text page-by-page so that
            # image placeholders stay close to the originating page content.
            doc = fitz.open(pdf_path)
            page_blocks: List[str] = []
            running_offset = 0
            zoom = 2.0  # render scale for snapshots/crops (vision readability)
            
            for page_num in range(len(doc)):
                page = doc[page_num]
                image_list = page.get_images(full=True)
                page_text = (page.get_text("text") or "").strip()
                
                page_lines: List[str] = [f"## Page {page_num + 1}"]
                if page_text:
                    page_lines.append(page_text)
                page_has_extracted_image = False
                
                for img_index, img_info in enumerate(image_list):
                    try:
                        # Extract image
                        xref = img_info[0]
                        base_image = doc.extract_image(xref)
                        image_bytes = base_image["image"]
                        image_ext = base_image.get("ext", "png")
                        
                        # Generate image ID and filename
                        image_id = self._generate_image_id(doc_hash, page_num + 1, img_index + 1)
                        image_filename = f"{image_id}.{image_ext}"
                        image_path = image_dir / image_filename
                        
                        # Save image
                        with open(image_path, "wb") as img_file:
                            img_file.write(image_bytes)
                        
                        # Get image dimensions
                        try:
                            img = Image.open(io.BytesIO(image_bytes))
                            width, height = img.size
                        except Exception:
                            width, height = 0, 0
                        
                        # Create placeholder and append to current page block
                        placeholder = f"[IMAGE: {image_id}]"
                        page_lines.append(placeholder)
                        
                        # Compute offset in the final rebuilt text
                        page_block_preview = "\n".join(page_lines)
                        insert_position = running_offset + page_block_preview.rfind(placeholder)
                        
                        # Convert path to be relative to project root or absolute
                        try:
                            relative_path = image_path.relative_to(Path.cwd())
                        except ValueError:
                            # If not in cwd, use absolute path
                            relative_path = image_path.absolute()
                        
                        # Record metadata
                        image_metadata = {
                            "id": image_id,
                            "path": str(relative_path),
                            "page": page_num + 1,
                            "text_offset": insert_position,
                            "text_length": len(placeholder),
                            "position": {
                                "width": width,
                                "height": height,
                                "page": page_num + 1,
                                "index": img_index
                            }
                        }
                        images_metadata.append(image_metadata)
                        page_has_extracted_image = True
                        
                        logger.debug(f"Extracted image {image_id} from page {page_num + 1}")
                        
                    except Exception as e:
                        logger.warning(f"Failed to extract image {img_index} from page {page_num + 1}: {e}")
                        continue

                # Fallback for PPT-converted PDFs:
                # many slides are vector drawings and cannot be extracted via get_images().
                # Prefer cropping "visual blocks" (drawings) for better retrieval.
                # If none found, render the full page as a snapshot so vision captioning can still work.
                if not page_has_extracted_image:
                    try:
                        # Render page once; reuse for crops and snapshot.
                        pix = page.get_pixmap(matrix=fitz.Matrix(zoom, zoom), alpha=False)

                        # 1) Try to crop drawing regions (vector shapes / charts)
                        crop_count = 0
                        max_crops_per_page = 6
                        min_area_ratio = 0.04  # skip tiny icons/logos

                        drawings = []
                        try:
                            drawings = page.get_drawings() or []
                        except Exception:
                            drawings = []

                        rects: List[fitz.Rect] = []
                        for d in drawings:
                            r = d.get("rect")
                            if r is None:
                                continue
                            try:
                                rect = fitz.Rect(r)
                            except Exception:
                                continue
                            if rect.is_empty or rect.is_infinite:
                                continue
                            rects.append(rect)

                        if rects:
                            merged = self._merge_rects(rects)
                            page_area = float(page.rect.get_area()) if hasattr(page.rect, "get_area") else float(page.rect.width * page.rect.height)

                            for i, rect in enumerate(merged):
                                if crop_count >= max_crops_per_page:
                                    break
                                # Filter by relative area
                                rect_area = float(rect.get_area()) if hasattr(rect, "get_area") else float(rect.width * rect.height)
                                if page_area > 0 and (rect_area / page_area) < min_area_ratio:
                                    continue

                                # Expand a bit to include labels around charts
                                margin = 8.0  # PDF points, small pad
                                expanded = fitz.Rect(
                                    rect.x0 - margin,
                                    rect.y0 - margin,
                                    rect.x1 + margin,
                                    rect.y1 + margin,
                                ) & page.rect  # clamp to page

                                # Convert to pixels (pixmap space)
                                x0 = max(0, int(expanded.x0 * zoom))
                                y0 = max(0, int(expanded.y0 * zoom))
                                x1 = min(pix.width, int(expanded.x1 * zoom))
                                y1 = min(pix.height, int(expanded.y1 * zoom))
                                if x1 <= x0 or y1 <= y0:
                                    continue

                                # PyMuPDF expects an *integer* rectangle for Pixmap cropping.
                                # Using IRect avoids internal type-check errors on some builds.
                                crop_pix = fitz.Pixmap(pix, fitz.IRect(x0, y0, x1, y1))

                                crop_seq = 800 + crop_count  # reserved for drawing crops
                                crop_id = self._generate_image_id(doc_hash, page_num + 1, crop_seq)
                                crop_path = image_dir / f"{crop_id}.png"
                                crop_pix.save(str(crop_path))

                                placeholder = f"[IMAGE: {crop_id}]"
                                page_lines.append(placeholder)
                                page_block_preview = "\n".join(page_lines)
                                insert_position = running_offset + page_block_preview.rfind(placeholder)

                                try:
                                    relative_path = crop_path.relative_to(Path.cwd())
                                except ValueError:
                                    relative_path = crop_path.absolute()

                                images_metadata.append(
                                    {
                                        "id": crop_id,
                                        "path": str(relative_path),
                                        "page": page_num + 1,
                                        "text_offset": insert_position,
                                        "text_length": len(placeholder),
                                        "position": {
                                            "width": crop_pix.width,
                                            "height": crop_pix.height,
                                            "page": page_num + 1,
                                            "index": crop_seq,
                                            "type": "page_crop",
                                            "bbox_pdf": [expanded.x0, expanded.y0, expanded.x1, expanded.y1],
                                        },
                                    }
                                )
                                crop_count += 1

                        # 2) If no crops were produced, snapshot whole page
                        if crop_count == 0:
                            snapshot_seq = 999  # reserved sequence for page snapshot
                            snapshot_id = self._generate_image_id(doc_hash, page_num + 1, snapshot_seq)
                            snapshot_path = image_dir / f"{snapshot_id}.png"
                            pix.save(str(snapshot_path))

                            placeholder = f"[IMAGE: {snapshot_id}]"
                            page_lines.append(placeholder)

                            page_block_preview = "\n".join(page_lines)
                            insert_position = running_offset + page_block_preview.rfind(placeholder)

                            try:
                                relative_path = snapshot_path.relative_to(Path.cwd())
                            except ValueError:
                                relative_path = snapshot_path.absolute()

                            images_metadata.append(
                                {
                                    "id": snapshot_id,
                                    "path": str(relative_path),
                                    "page": page_num + 1,
                                    "text_offset": insert_position,
                                    "text_length": len(placeholder),
                                    "position": {
                                        "width": pix.width,
                                        "height": pix.height,
                                        "page": page_num + 1,
                                        "index": snapshot_seq,
                                        "type": "page_snapshot",
                                    },
                                }
                            )
                            logger.debug(f"Generated page snapshot {snapshot_id} for page {page_num + 1}")
                    except Exception as e:
                        logger.warning(
                            f"Failed to generate page snapshot for page {page_num + 1}: {e}"
                        )
                
                page_block = "\n".join(page_lines).strip()
                if page_block:
                    page_blocks.append(page_block)
                    running_offset += len(page_block) + 2  # account for \n\n join
            
            doc.close()
            if page_blocks:
                modified_text = "\n\n".join(page_blocks)
            
            if images_metadata:
                logger.info(f"Extracted {len(images_metadata)} images from {pdf_path}")
            else:
                logger.debug(f"No images found in {pdf_path}")
            
            return modified_text, images_metadata
            
        except Exception as e:
            logger.warning(f"Image extraction failed for {pdf_path}: {e}")
            # Graceful degradation: return original text without images
            return text_content, []

    @staticmethod
    def _merge_rects(rects: List["fitz.Rect"]) -> List["fitz.Rect"]:
        """Merge overlapping/nearby rectangles to reduce duplicate crops.

        This is a simple O(n^2) merge suitable for a small number of drawing rects per page.
        """
        if not rects:
            return []

        # Sort for deterministic output
        rects = sorted(rects, key=lambda r: (r.y0, r.x0, r.y1, r.x1))
        merged: List[fitz.Rect] = []

        def overlaps(a: fitz.Rect, b: fitz.Rect, pad: float = 4.0) -> bool:
            ap = fitz.Rect(a.x0 - pad, a.y0 - pad, a.x1 + pad, a.y1 + pad)
            bp = fitz.Rect(b.x0 - pad, b.y0 - pad, b.x1 + pad, b.y1 + pad)
            return not (ap.x1 < bp.x0 or bp.x1 < ap.x0 or ap.y1 < bp.y0 or bp.y1 < ap.y0)

        for r in rects:
            placed = False
            for i in range(len(merged)):
                if overlaps(merged[i], r):
                    merged[i] = merged[i] | r  # union
                    placed = True
                    break
            if not placed:
                merged.append(r)

        # One more pass to ensure transitive merges
        changed = True
        while changed:
            changed = False
            out: List[fitz.Rect] = []
            for r in merged:
                merged_into_existing = False
                for j in range(len(out)):
                    if overlaps(out[j], r):
                        out[j] = out[j] | r
                        merged_into_existing = True
                        changed = True
                        break
                if not merged_into_existing:
                    out.append(r)
            merged = out

        return merged
    
    @staticmethod
    def _generate_image_id(doc_hash: str, page: int, sequence: int) -> str:
        """Generate unique image ID.
        
        Args:
            doc_hash: Document hash.
            page: Page number (0-based).
            sequence: Image sequence on page (0-based).
            
        Returns:
            Unique image ID string.
        """
        return f"{doc_hash[:8]}_{page}_{sequence}"
