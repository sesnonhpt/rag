"""Processing history management service."""

from __future__ import annotations

import json
import threading
from datetime import datetime, timezone
from pathlib import Path
from typing import List, Optional

from app.core.paths import PROCESSING_HISTORY_FILE
from app.schemas.document_models import ProcessingHistoryItem
from src.observability.logger import get_logger

logger = get_logger(__name__)


class ProcessingHistoryService:
    """Service for managing document processing history."""
    
    def __init__(self, storage_path: Optional[Path] = None, max_history_items: int = 10):
        """
        Initialize processing history service.
        
        Args:
            storage_path: Path to history storage file (default: PROCESSING_HISTORY_FILE)
            max_history_items: Maximum number of history items to keep (default: 10)
        """
        self.storage_path = storage_path or PROCESSING_HISTORY_FILE
        self.max_history_items = max_history_items
        self._lock = threading.Lock()
        
        # Ensure storage file exists
        self._ensure_storage_file()
    
    def _ensure_storage_file(self):
        """Ensure storage file exists with valid JSON structure."""
        if not self.storage_path.exists():
            self.storage_path.parent.mkdir(parents=True, exist_ok=True)
            self._write_history({'items': []})
            logger.info(f"Created processing history file: {self.storage_path}")
    
    def _read_history(self) -> dict:
        """
        Read history from storage file.
        
        Returns:
            Dictionary with 'items' list
        """
        try:
            with open(self.storage_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
                if not isinstance(data, dict) or 'items' not in data:
                    logger.warning("Invalid history file format, resetting")
                    return {'items': []}
                return data
        except (json.JSONDecodeError, FileNotFoundError) as e:
            logger.warning(f"Failed to read history file: {e}, resetting")
            return {'items': []}
    
    def _write_history(self, data: dict):
        """
        Write history to storage file.
        
        Args:
            data: Dictionary with 'items' list
        """
        try:
            with open(self.storage_path, 'w', encoding='utf-8') as f:
                json.dump(data, f, ensure_ascii=False, indent=2)
        except Exception as e:
            logger.exception(f"Failed to write history file: {e}")
            raise
    
    def save_history(
        self,
        processing_id: str,
        document_filename: str,
        processing_option: str,
        result: str,
        custom_prompt: Optional[str] = None
    ) -> None:
        """
        Save processing history item.
        
        Args:
            processing_id: Unique processing ID
            document_filename: Original document filename
            processing_option: Processing option used
            result: Full processing result
            custom_prompt: Custom prompt (if used)
        """
        with self._lock:
            try:
                # Read current history
                data = self._read_history()
                items = data.get('items', [])
                
                # Create result preview (first 200 characters)
                result_preview = result[:200] if len(result) > 200 else result
                
                # Create new history item
                new_item = {
                    'processing_id': processing_id,
                    'document_filename': document_filename,
                    'processing_option': processing_option,
                    'custom_prompt': custom_prompt,
                    'result': result,  # Store full result
                    'result_preview': result_preview,
                    'processed_at': datetime.now(timezone.utc).isoformat()
                }
                
                # Add to beginning of list (newest first)
                items.insert(0, new_item)
                
                # Cleanup old items if exceeds max
                if len(items) > self.max_history_items:
                    items = items[:self.max_history_items]
                    logger.info(f"Cleaned up old history items, kept {self.max_history_items} most recent")
                
                # Write back to file
                data['items'] = items
                self._write_history(data)
                
                logger.info(
                    f"Saved processing history: processing_id={processing_id}, "
                    f"option={processing_option}, "
                    f"filename={document_filename}"
                )
            
            except Exception as e:
                logger.exception("Failed to save processing history")
                raise
    
    def get_history(
        self,
        limit: int = 10,
        offset: int = 0
    ) -> List[ProcessingHistoryItem]:
        """
        Get processing history list with pagination.
        
        Args:
            limit: Maximum number of items to return (default: 10, max: 50)
            offset: Number of items to skip (default: 0)
        
        Returns:
            List of ProcessingHistoryItem objects
        """
        with self._lock:
            try:
                # Validate and cap limit
                limit = min(max(1, limit), 50)
                offset = max(0, offset)
                
                # Read history
                data = self._read_history()
                items = data.get('items', [])
                
                # Apply pagination
                paginated_items = items[offset:offset + limit]
                
                # Convert to ProcessingHistoryItem objects
                history_items = []
                for item in paginated_items:
                    try:
                        history_items.append(ProcessingHistoryItem(
                            processing_id=item['processing_id'],
                            document_filename=item['document_filename'],
                            processing_option=item['processing_option'],
                            custom_prompt=item.get('custom_prompt'),
                            result_preview=item['result_preview'],
                            processed_at=item['processed_at']
                        ))
                    except (KeyError, TypeError) as e:
                        logger.warning(f"Skipping invalid history item: {e}")
                        continue
                
                logger.info(f"Retrieved {len(history_items)} history items (limit={limit}, offset={offset})")
                
                return history_items
            
            except Exception as e:
                logger.exception("Failed to get processing history")
                raise
    
    def get_full_result(self, processing_id: str) -> Optional[str]:
        """
        Get full processing result by processing ID.
        
        Args:
            processing_id: Processing ID
        
        Returns:
            Full result string, or None if not found
        """
        with self._lock:
            try:
                data = self._read_history()
                items = data.get('items', [])
                
                for item in items:
                    if item.get('processing_id') == processing_id:
                        return item.get('result')
                
                return None
            
            except Exception as e:
                logger.exception(f"Failed to get full result for processing_id={processing_id}")
                return None
    
    def get_total_count(self) -> int:
        """
        Get total number of history items.
        
        Returns:
            Total count
        """
        with self._lock:
            try:
                data = self._read_history()
                return len(data.get('items', []))
            except Exception as e:
                logger.exception("Failed to get history count")
                return 0
    
    def clear_history(self) -> int:
        """
        Clear all processing history.
        
        Returns:
            Number of items deleted
        """
        with self._lock:
            try:
                data = self._read_history()
                deleted_count = len(data.get('items', []))
                
                # Reset to empty
                self._write_history({'items': []})
                
                logger.info(f"Cleared processing history: deleted {deleted_count} items")
                
                return deleted_count
            
            except Exception as e:
                logger.exception("Failed to clear processing history")
                raise
