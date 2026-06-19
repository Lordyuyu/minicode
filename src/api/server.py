"""
FastAPI application factory for the MiniCode REST + SSE API.

Usage::

    uvicorn src.api.server:app --host 0.0.0.0 --port 8000 --reload
"""

from __future__ import annotations

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from src.api.routes import router
from src.utils.logger import get_logger

logger = get_logger(__name__)


def create_app() -> FastAPI:
    """Build and configure the FastAPI application."""
    app = FastAPI(
        title="MiniCode API",
        description="REST + SSE streaming API for the MiniCode AI code-repair agent",
        version="0.1.0",
        docs_url="/docs",
        redoc_url="/redoc",
    )

    # CORS — open for development, lock down in production
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    app.include_router(router)

    @app.on_event("startup")
    async def _startup() -> None:
        logger.info("MiniCode API server starting")

    @app.on_event("shutdown")
    async def _shutdown() -> None:
        logger.info("MiniCode API server shutting down")

    @app.get("/health")
    async def health() -> dict:
        return {"status": "ok"}

    return app


# Module-level app instance (used by uvicorn)
app = create_app()
