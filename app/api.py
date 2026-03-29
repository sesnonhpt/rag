"""
API Entry Point - Re-export FastAPI app from chat_api
"""
from app.chat_api import app

__all__ = ["app"]
