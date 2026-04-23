"""Template database for storing template metadata and versions."""

from __future__ import annotations

import json
import sqlite3
import uuid
from contextlib import contextmanager
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Any

from src.observability.logger import get_logger

logger = get_logger(__name__)


class TemplateDatabase:
    """SQLite database for template metadata and version management."""
    
    def __init__(self, db_path: str):
        self.db_path = db_path
        self._init_db()
    
    def _init_db(self):
        """Initialize database schema."""
        Path(self.db_path).parent.mkdir(parents=True, exist_ok=True)
        
        with self.get_connection() as conn:
            # Enable WAL mode for better concurrency
            conn.execute("PRAGMA journal_mode=WAL")
            conn.execute("PRAGMA cache_size=-64000")  # 64MB cache
            conn.execute("PRAGMA foreign_keys=ON")
            
            # Main templates table
            conn.execute('''
                CREATE TABLE IF NOT EXISTS templates (
                    template_id TEXT PRIMARY KEY,
                    filename TEXT NOT NULL,
                    file_size INTEGER NOT NULL,
                    file_format TEXT NOT NULL,
                    file_path TEXT NOT NULL,
                    collection TEXT NOT NULL DEFAULT 'default',
                    subject TEXT,
                    grade TEXT,
                    tags TEXT,
                    created_at TEXT NOT NULL,
                    updated_at TEXT NOT NULL,
                    version_count INTEGER DEFAULT 1,
                    is_deleted INTEGER DEFAULT 0,
                    deleted_at TEXT
                )
            ''')
            
            # Version history table
            conn.execute('''
                CREATE TABLE IF NOT EXISTS template_versions (
                    version_id TEXT PRIMARY KEY,
                    template_id TEXT NOT NULL,
                    content_html TEXT NOT NULL,
                    created_at TEXT NOT NULL,
                    change_summary TEXT,
                    FOREIGN KEY (template_id) REFERENCES templates(template_id)
                )
            ''')
            
            # Indexes
            conn.execute('CREATE INDEX IF NOT EXISTS idx_templates_collection ON templates(collection)')
            conn.execute('CREATE INDEX IF NOT EXISTS idx_templates_updated_at ON templates(updated_at DESC)')
            conn.execute('CREATE INDEX IF NOT EXISTS idx_templates_is_deleted ON templates(is_deleted)')
            conn.execute('CREATE INDEX IF NOT EXISTS idx_versions_template_id ON template_versions(template_id, created_at DESC)')
            
            conn.commit()
            logger.info(f"Template database initialized at {self.db_path}")
    
    @contextmanager
    def get_connection(self):
        """Get database connection with row factory."""
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        try:
            yield conn
        finally:
            conn.close()
    
    def create_template(
        self,
        filename: str,
        file_size: int,
        file_format: str,
        file_path: str,
        collection: str = "default",
        subject: Optional[str] = None,
        grade: Optional[str] = None,
        tags: Optional[List[str]] = None,
    ) -> str:
        """Create a new template record."""
        template_id = str(uuid.uuid4())
        now = datetime.now().isoformat()
        tags_json = json.dumps(tags or [])
        
        with self.get_connection() as conn:
            conn.execute('''
                INSERT INTO templates (
                    template_id, filename, file_size, file_format, file_path,
                    collection, subject, grade, tags, created_at, updated_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            ''', (
                template_id, filename, file_size, file_format, file_path,
                collection, subject, grade, tags_json, now, now
            ))
            conn.commit()
        
        logger.info(f"Created template: {template_id} ({filename})")
        return template_id
    
    def get_template(self, template_id: str) -> Optional[Dict[str, Any]]:
        """Get template by ID."""
        with self.get_connection() as conn:
            cursor = conn.execute('''
                SELECT * FROM templates WHERE template_id = ? AND is_deleted = 0
            ''', (template_id,))
            row = cursor.fetchone()
            
            if row:
                result = dict(row)
                result['tags'] = json.loads(result['tags'] or '[]')
                return result
            return None
    
    def list_templates(
        self,
        collection: Optional[str] = None,
        limit: int = 20,
        offset: int = 0,
        sort_by: str = "updated_at",
        order: str = "desc"
    ) -> List[Dict[str, Any]]:
        """List templates with pagination and sorting."""
        with self.get_connection() as conn:
            query = 'SELECT * FROM templates WHERE is_deleted = 0'
            params = []
            
            if collection:
                query += ' AND collection = ?'
                params.append(collection)
            
            # Validate sort_by to prevent SQL injection
            valid_sort_fields = ['filename', 'created_at', 'updated_at', 'file_size']
            if sort_by not in valid_sort_fields:
                sort_by = 'updated_at'
            
            order_clause = 'DESC' if order.lower() == 'desc' else 'ASC'
            query += f' ORDER BY {sort_by} {order_clause} LIMIT ? OFFSET ?'
            params.extend([limit, offset])
            
            cursor = conn.execute(query, params)
            rows = cursor.fetchall()
            
            results = []
            for row in rows:
                result = dict(row)
                result['tags'] = json.loads(result['tags'] or '[]')
                results.append(result)
            
            return results
    
    def update_template(
        self,
        template_id: str,
        **kwargs
    ) -> bool:
        """Update template metadata."""
        allowed_fields = ['filename', 'subject', 'grade', 'tags']
        updates = []
        params = []
        
        for key, value in kwargs.items():
            if key in allowed_fields:
                if key == 'tags':
                    value = json.dumps(value)
                updates.append(f'{key} = ?')
                params.append(value)
        
        if not updates:
            return False
        
        params.append(datetime.now().isoformat())
        params.append(template_id)
        
        with self.get_connection() as conn:
            conn.execute(f'''
                UPDATE templates 
                SET {', '.join(updates)}, updated_at = ?
                WHERE template_id = ?
            ''', params)
            conn.commit()
        
        logger.info(f"Updated template: {template_id}")
        return True
    
    def delete_template(self, template_id: str) -> bool:
        """Soft delete a template."""
        now = datetime.now().isoformat()
        
        with self.get_connection() as conn:
            conn.execute('''
                UPDATE templates 
                SET is_deleted = 1, deleted_at = ?
                WHERE template_id = ?
            ''', (now, template_id))
            conn.commit()
        
        logger.info(f"Deleted template: {template_id}")
        return True
    
    def create_version(
        self,
        template_id: str,
        content_html: str,
        change_summary: Optional[str] = None
    ) -> str:
        """Create a new version for a template."""
        version_id = str(uuid.uuid4())
        now = datetime.now().isoformat()
        
        with self.get_connection() as conn:
            # Insert new version
            conn.execute('''
                INSERT INTO template_versions (
                    version_id, template_id, content_html, created_at, change_summary
                ) VALUES (?, ?, ?, ?, ?)
            ''', (version_id, template_id, content_html, now, change_summary))
            
            # Update version count and updated_at
            conn.execute('''
                UPDATE templates 
                SET version_count = version_count + 1, updated_at = ?
                WHERE template_id = ?
            ''', (now, template_id))
            
            # Keep only last 10 versions
            conn.execute('''
                DELETE FROM template_versions
                WHERE template_id = ? AND version_id NOT IN (
                    SELECT version_id FROM template_versions
                    WHERE template_id = ?
                    ORDER BY created_at DESC
                    LIMIT 10
                )
            ''', (template_id, template_id))
            
            conn.commit()
        
        logger.info(f"Created version: {version_id} for template {template_id}")
        return version_id
    
    def get_version(self, version_id: str) -> Optional[Dict[str, Any]]:
        """Get a specific version."""
        with self.get_connection() as conn:
            cursor = conn.execute('''
                SELECT * FROM template_versions WHERE version_id = ?
            ''', (version_id,))
            row = cursor.fetchone()
            return dict(row) if row else None
    
    def list_versions(self, template_id: str) -> List[Dict[str, Any]]:
        """List all versions for a template."""
        with self.get_connection() as conn:
            cursor = conn.execute('''
                SELECT * FROM template_versions 
                WHERE template_id = ?
                ORDER BY created_at DESC
                LIMIT 10
            ''', (template_id,))
            rows = cursor.fetchall()
            return [dict(row) for row in rows]
    
    def get_latest_version(self, template_id: str) -> Optional[Dict[str, Any]]:
        """Get the latest version for a template."""
        with self.get_connection() as conn:
            cursor = conn.execute('''
                SELECT * FROM template_versions 
                WHERE template_id = ?
                ORDER BY created_at DESC
                LIMIT 1
            ''', (template_id,))
            row = cursor.fetchone()
            return dict(row) if row else None
