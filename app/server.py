from __future__ import annotations

import asyncio
import contextlib

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse, Response, StreamingResponse

from app.config import get_settings
from app.models import ChatRequest, SpeechRequest
from app.services import AppServices


def create_app() -> FastAPI:
    settings = get_settings()
    services = AppServices(settings)

    app = FastAPI(title="Prox — Vulcan OmniPro 220 Assistant", version="0.2.0")

    app.add_middleware(
        CORSMiddleware,
        allow_origins=settings.cors_allowed_origins,
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    @app.on_event("startup")
    async def startup() -> None:
        with contextlib.suppress(Exception):
            services.ensure_cache()
        try:
            await services.validate_anthropic_models()
        except Exception as exc:
            services.startup_status["model_validation"] = {
                "validated": False,
                "reason": "validation_failed",
                "message": str(exc),
            }
            if settings.strict_startup_validation:
                raise
        services.ensure_cache()

    # on_event is deperecated, fix: 

    @app.get("/health")
    async def health() -> JSONResponse:
        return JSONResponse({
            "status": "ok",
            "cache_ready": settings.cache_ready_file.exists(),
            "anthropic_enabled": settings.anthropic_enabled,
            "claude_sdk_installed": settings.claude_sdk_installed,
            "sentence_transformers_installed": settings.sentence_transformers_installed,
            "local_tts_enabled": settings.local_tts_enabled,
            "local_tts_ready": settings.local_tts_enabled and services.tts.enabled,
            "hosted_demo": settings.hosted_demo,
            "frontend_origin": settings.frontend_origin,
            "semantic_embed_model": settings.semantic_embed_model,
            "semantic_embed_model_path": str(settings.semantic_embed_model_path),
            "semantic_embed_model_cached": settings.semantic_embed_model_path.exists(),
            "startup": services.startup_status,
        })

    @app.post("/preprocess")
    async def preprocess() -> JSONResponse:
        manifest = services.preprocessor.run()
        return JSONResponse({"status": "ok", "manifest": manifest})

    @app.post("/chat")
    async def chat(request: ChatRequest) -> StreamingResponse:
        services.ensure_cache()
        return StreamingResponse(
            services.stream_answer(request),
            media_type="text/event-stream",
            headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
        )

    @app.get("/pages/{doc}/{page}")
    async def page_image(doc: str, page: int) -> FileResponse:
        services.ensure_cache()
        path = services.page_store.get_page_image_path_by_parts(doc, page)
        if not path.exists():
            return JSONResponse({"status": "error", "message": "Page image not found"}, status_code=404)
        return FileResponse(path, media_type="image/png")

    @app.get("/artifacts/{artifact_id}")
    async def artifact(artifact_id: str) -> Response:
        artifact_payload = services.get_artifact(artifact_id)
        if not artifact_payload:
            return JSONResponse({"status": "error", "message": "Artifact not found"}, status_code=404)
        artifact_type = artifact_payload["artifact_type"]
        media_type = "text/plain"
        if artifact_type == "svg":
            media_type = "image/svg+xml"
        elif artifact_type == "html":
            media_type = "text/html"
        elif artifact_type == "code":
            media_type = "application/vnd.ant.code"
        elif artifact_type == "markdown":
            media_type = "text/markdown"
        elif artifact_type == "mermaid":
            media_type = "application/vnd.ant.mermaid"
        elif artifact_type == "json":
            media_type = "application/json"
        return Response(content=artifact_payload["content"], media_type=media_type)

    @app.post("/speech")
    async def synthesize_speech(request: SpeechRequest) -> JSONResponse:
        payload = await asyncio.to_thread(services.synthesize_speech, request.text)
        if not payload:
            return JSONResponse({"status": "error", "message": "Local TTS is unavailable"}, status_code=503)
        return JSONResponse({"status": "ok", "audio_id": payload["audio_id"], "url": payload["url"]})

    @app.get("/speech/{audio_id}")
    async def speech_file(audio_id: str) -> Response:
        path = services.get_speech_path(audio_id)
        if not path:
            return JSONResponse({"status": "error", "message": "Speech not found"}, status_code=404)
        return FileResponse(path, media_type="audio/wav", filename=f"{audio_id}.wav")

    return app
