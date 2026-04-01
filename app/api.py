"""
API Entry Point - Re-export FastAPI app from main
"""
from app.main import app

__all__ = ["app"]
