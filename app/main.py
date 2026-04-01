"""Primary FastAPI app assembly with modular routers."""

from __future__ import annotations

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles

from app.core.app_runtime import global_exception_handler, lifespan
from app.core.paths import STATIC_DIR
from app.routers.assets import router as assets_router
from app.routers.chat import router as chat_router
from app.routers.lesson import router as lesson_router


app = FastAPI(title="RAG Chat API", lifespan=lifespan)

app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost", "http://127.0.0.1"],
    allow_origin_regex=r"http://(localhost|127\.0\.0\.1)(:\d+)?",
    allow_methods=["*"],
    allow_headers=["*"],
)

app.add_exception_handler(Exception, global_exception_handler)

app.include_router(assets_router)
app.include_router(chat_router)
app.include_router(lesson_router)
