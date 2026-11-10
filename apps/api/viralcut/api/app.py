"""FastAPI application factory."""

from __future__ import annotations

import logging
from contextlib import asynccontextmanager

from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from .. import __version__
from ..config import get_settings
from ..db import get_engine
from .routes import router

log = logging.getLogger("viralcut.api")


@asynccontextmanager
async def lifespan(app: FastAPI):
    get_engine()  # create tables
    worker = None
    if get_settings().embedded_worker:
        from ..worker.main import start_embedded

        worker = start_embedded()
        log.info("embedded worker started")
    yield
    if worker is not None:
        worker.stop()


def create_app() -> FastAPI:
    settings = get_settings()
    app = FastAPI(
        title="ViralCut AI API",
        version=__version__,
        description="Upload a long video, get ranked, captioned vertical clips.",
        lifespan=lifespan,
    )
    app.add_middleware(
        CORSMiddleware,
        allow_origins=settings.cors_origin_list,
        allow_methods=["GET", "POST", "PATCH", "DELETE", "OPTIONS"],
        allow_headers=["Content-Type"],
        expose_headers=["Content-Disposition"],
        max_age=600,
    )

    @app.middleware("http")
    async def security_headers(request: Request, call_next):  # noqa: ANN001
        response = await call_next(request)
        response.headers.setdefault("X-Content-Type-Options", "nosniff")
        response.headers.setdefault(
            "Referrer-Policy", "no-referrer"
        )  # job ids are capabilities: never leak them
        response.headers.setdefault("Cross-Origin-Resource-Policy", "cross-origin")
        if response.headers.get("content-type", "").startswith("application/json"):
            response.headers.setdefault("Cache-Control", "no-store")
        return response

    @app.exception_handler(Exception)
    async def unhandled(request: Request, exc: Exception):  # noqa: ANN001
        log.exception("unhandled error on %s %s", request.method, request.url.path)
        return JSONResponse({"detail": "Internal server error."}, status_code=500)

    app.include_router(router)
    return app


app = create_app()
