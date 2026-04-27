from __future__ import annotations

import asyncio
from collections import deque
import contextlib
import json
from pathlib import Path
import secrets
import time

import httpx
import websockets
from anthropic import AsyncAnthropic, AuthenticationError
from fastapi import FastAPI, Request, WebSocket
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse, Response, StreamingResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from app.auth_store import AuthStore, DuplicateUserError, UserRecord
from app.config import get_settings
from app.models import ChatRequest, SpeechRequest
from app.services import AppServices


class ValidateKeyRequest(BaseModel):
    anthropic_api_key: str


class ValidateDeepgramKeyRequest(BaseModel):
    deepgram_api_key: str


class SaveCredentialRequest(BaseModel):
    api_key: str


class AuthRequest(BaseModel):
    username: str | None = None
    email: str | None = None
    password: str


class InMemoryRateLimiter:
    def __init__(self, max_attempts: int, window_seconds: int) -> None:
        self.max_attempts = max_attempts
        self.window_seconds = window_seconds
        self._events: dict[str, deque[float]] = {}

    def hit(self, key: str) -> int | None:
        now = time.time()
        bucket = self._events.setdefault(key, deque())
        cutoff = now - self.window_seconds
        while bucket and bucket[0] <= cutoff:
            bucket.popleft()
        if len(bucket) >= self.max_attempts:
            retry_after = int(max(1, self.window_seconds - (now - bucket[0])))
            return retry_after
        bucket.append(now)
        if not bucket:
            self._events.pop(key, None)
        return None


def _deepgram_error_response(exc: Exception) -> tuple[int, str]:
    if isinstance(exc, ValueError):
        return 400, str(exc)
    if isinstance(exc, httpx.HTTPStatusError):
        status = exc.response.status_code
        if status in (401, 403):
            return 401, "Invalid Deepgram API key"
        if status == 402:
            return 402, "Deepgram API key has no available balance"
        if status == 429:
            return 429, "Deepgram API key is rate limited or out of quota"
        return 502, f"Deepgram request failed with HTTP {status}"

    message = str(exc)
    lowered = message.lower()
    if "401" in message or "403" in message or "unauthorized" in lowered:
        return 401, "Invalid Deepgram API key"
    if "402" in message or "payment" in lowered or "balance" in lowered:
        return 402, "Deepgram API key has no available balance"
    if "429" in message or "quota" in lowered or "rate" in lowered:
        return 429, "Deepgram API key is rate limited or out of quota"
    return 502, message


def _normalize_username(username: str) -> str:
    return username.strip().lower()


def _auth_username(payload: AuthRequest) -> str:
    return _normalize_username(payload.username or payload.email or "")


def _validate_auth_payload(username: str, password: str) -> tuple[str, str | None]:
    normalized = _normalize_username(username)
    if len(normalized) < 3:
        return normalized, "Username must be at least 3 characters."
    if len(normalized) > 32:
        return normalized, "Username must be 32 characters or fewer."
    if not all(ch.isalnum() or ch in {"_", "-", "."} for ch in normalized):
        return normalized, "Use letters, numbers, dots, dashes, or underscores in your username."
    if len(password) < 8:
        return normalized, "Password must be at least 8 characters."
    return normalized, None


def create_app() -> FastAPI:
    settings = get_settings()
    if settings.deployed and not settings.app_secret_key_looks_secure:
        raise RuntimeError("APP_SECRET_KEY must be set to a strong, unique value before deploying.")
    auth_store = AuthStore(settings.database_url, settings.database_file, settings.app_secret_key)
    services = AppServices(settings)
    session_cookie_name = settings.session_cookie_name
    csrf_cookie_name = settings.csrf_cookie_name
    csrf_header_name = settings.csrf_header_name
    auth_rate_limiter = InMemoryRateLimiter(
        max_attempts=settings.auth_rate_limit_max_attempts,
        window_seconds=settings.auth_rate_limit_window_seconds,
    )

    def _client_ip(request: Request) -> str:
        forwarded_for = (request.headers.get("x-forwarded-for") or "").split(",")[0].strip()
        if forwarded_for:
            return forwarded_for
        return request.client.host if request.client else "unknown"

    def _set_cookie(response: Response, name: str, value: str, max_age: int) -> None:
        response.set_cookie(
            name,
            value,
            httponly=name == session_cookie_name,
            samesite=settings.session_cookie_samesite,
            secure=settings.deployed,
            max_age=max_age,
            path="/",
        )

    def _delete_cookie(response: Response, name: str) -> None:
        response.delete_cookie(
            name,
            path="/",
            secure=settings.deployed,
            httponly=name == session_cookie_name,
            samesite=settings.session_cookie_samesite,
        )

    def _set_session_cookies(response: Response, session_token: str, csrf_token: str) -> None:
        max_age = settings.session_ttl_days * 24 * 60 * 60
        _set_cookie(response, session_cookie_name, session_token, max_age=max_age)
        _set_cookie(response, csrf_cookie_name, csrf_token, max_age=max_age)

    def _ensure_anonymous_csrf_cookie(response: Response, request: Request | None = None) -> str:
        existing = (request.cookies.get(csrf_cookie_name) if request is not None else "") or ""
        csrf_token = existing.strip() or secrets.token_urlsafe(32)
        _set_cookie(response, csrf_cookie_name, csrf_token, max_age=settings.session_ttl_days * 24 * 60 * 60)
        return csrf_token

    def _csrf_error(message: str = "CSRF validation failed.") -> JSONResponse:
        return JSONResponse({"error": message}, status_code=403)

    def _require_csrf(request: Request, *, bind_to_session: bool) -> JSONResponse | None:
        csrf_cookie = (request.cookies.get(csrf_cookie_name) or "").strip()
        csrf_header = (request.headers.get(csrf_header_name) or "").strip()
        if not csrf_cookie or not csrf_header:
            return _csrf_error("Missing CSRF token.")
        if not secrets.compare_digest(csrf_cookie, csrf_header):
            return _csrf_error()
        if bind_to_session and not auth_store.validate_session_csrf(
            request.cookies.get(session_cookie_name),
            csrf_header,
        ):
            return _csrf_error()
        return None

    def _auth_rate_limit_response(retry_after: int) -> JSONResponse:
        return JSONResponse(
            {"error": "Too many authentication attempts. Please try again later."},
            status_code=429,
            headers={"Retry-After": str(retry_after)},
        )

    def _check_auth_rate_limit(request: Request, username: str) -> JSONResponse | None:
        normalized = _normalize_username(username)
        keys = [
            f"auth:ip:{_client_ip(request)}",
            f"auth:user:{normalized}",
        ]
        retry_after = max((auth_rate_limiter.hit(key) or 0) for key in keys)
        if retry_after:
            return _auth_rate_limit_response(retry_after)
        return None

    def current_user_from_request(request: Request) -> UserRecord | None:
        return auth_store.get_user_by_session(request.cookies.get(session_cookie_name))

    def current_user_from_websocket(websocket: WebSocket) -> UserRecord | None:
        return auth_store.get_user_by_session(websocket.cookies.get(session_cookie_name))

    def anthropic_key_for_user(user: UserRecord | None) -> str:
        if settings.api_key:
            return settings.api_key
        if user is None:
            return ""
        return auth_store.get_api_key(user.id, "anthropic")

    def deepgram_key_for_user(user: UserRecord | None) -> str:
        if settings.deepgram_api_key:
            return settings.deepgram_api_key
        if user is None:
            return ""
        return auth_store.get_api_key(user.id, "deepgram")

    app = FastAPI(title="Vulcan OmniPro 220 Assistant", version="0.2.0")

    app.add_middleware(
        CORSMiddleware,
        allow_origins=settings.cors_allowed_origins,
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    frontend_dist_dir = settings.frontend_dist_dir
    frontend_assets_dir = frontend_dist_dir / "assets"

    def frontend_available() -> bool:
        return frontend_dist_dir.exists() and (frontend_dist_dir / "index.html").exists()

    def frontend_index_response() -> FileResponse:
        return FileResponse(frontend_dist_dir / "index.html")

    @app.on_event("startup")
    async def startup() -> None:
        auth_store.delete_expired_sessions()
        if not settings.deployed:
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
        if not settings.deployed:
            services.ensure_cache()
        # Build vision cache in background after startup (no-op if already cached)
        asyncio.ensure_future(services.build_vision_cache())

    # on_event is deperecated, fix: 

    @app.get("/health")
    async def health(request: Request) -> JSONResponse:
        response = JSONResponse({
            "status": "ok",
            "cache_ready": settings.cache_ready_file.exists(),
            "anthropic_enabled": bool(settings.api_key),
            "claude_sdk_installed": settings.claude_sdk_installed,
            "sentence_transformers_installed": settings.sentence_transformers_installed,
            "local_tts_enabled": settings.local_tts_enabled,
            "local_tts_ready": settings.local_tts_enabled and services.tts.enabled,
            "deepgram_enabled": bool(settings.deepgram_api_key),
            "deployment_env": settings.deployment_env,
            "frontend_origin": settings.frontend_origin,
            "semantic_embed_model": settings.semantic_embed_model,
            "semantic_embed_model_path": str(settings.semantic_embed_model_path),
            "semantic_embed_model_cached": settings.semantic_embed_model_path.exists(),
            "startup": services.startup_status,
        })
        _ensure_anonymous_csrf_cookie(response, request)
        return response

    @app.post("/auth/signup")
    async def signup(request: Request, payload: AuthRequest) -> JSONResponse:
        csrf_error = _require_csrf(request, bind_to_session=False)
        if csrf_error is not None:
            return csrf_error
        username = _auth_username(payload)
        password = payload.password
        _, validation_error = _validate_auth_payload(username, password)
        if validation_error:
            return JSONResponse({"error": validation_error}, status_code=400)
        limited = _check_auth_rate_limit(request, username)
        if limited is not None:
            return limited
        try:
            user = auth_store.create_user(username, password)
        except DuplicateUserError:
            return JSONResponse({"error": "That username is already taken."}, status_code=409)

        session = auth_store.create_session(user.id, ttl_days=settings.session_ttl_days)
        response = JSONResponse({"authenticated": True, "user": {"username": user.username}})
        _set_session_cookies(response, session.token, session.csrf_token)
        return response

    @app.post("/auth/login")
    async def login(request: Request, payload: AuthRequest) -> JSONResponse:
        csrf_error = _require_csrf(request, bind_to_session=False)
        if csrf_error is not None:
            return csrf_error
        username = _auth_username(payload)
        limited = _check_auth_rate_limit(request, username)
        if limited is not None:
            return limited
        user = auth_store.authenticate_user(username, payload.password)
        if user is None:
            return JSONResponse({"error": "Incorrect username or password."}, status_code=401)
        session = auth_store.create_session(user.id, ttl_days=settings.session_ttl_days)
        response = JSONResponse({"authenticated": True, "user": {"username": user.username}})
        _set_session_cookies(response, session.token, session.csrf_token)
        return response

    @app.post("/auth/logout")
    async def logout(request: Request) -> JSONResponse:
        csrf_error = _require_csrf(request, bind_to_session=True)
        if csrf_error is not None:
            return csrf_error
        auth_store.delete_session(request.cookies.get(session_cookie_name))
        response = JSONResponse({"ok": True})
        _delete_cookie(response, session_cookie_name)
        _delete_cookie(response, csrf_cookie_name)
        return response

    @app.get("/auth/me")
    async def auth_me(request: Request) -> JSONResponse:
        user = current_user_from_request(request)
        if user is None:
            response = JSONResponse({"authenticated": False}, status_code=401)
            _ensure_anonymous_csrf_cookie(response, request)
            return response
        response = JSONResponse({"authenticated": True, "user": {"username": user.username}})
        _ensure_anonymous_csrf_cookie(response, request)
        return response

    @app.post("/validate-key")
    async def validate_key(request: ValidateKeyRequest) -> JSONResponse:
        key = request.anthropic_api_key.strip()
        if not key:
            return JSONResponse({"valid": False, "error": "No key provided"}, status_code=400)
        try:
            client = AsyncAnthropic(api_key=key)
            await asyncio.wait_for(client.models.list(), timeout=15.0)
            return JSONResponse({"valid": True})
        except asyncio.TimeoutError:
            return JSONResponse({"valid": False, "error": "Validation timed out"}, status_code=504)
        except AuthenticationError:
            return JSONResponse({"valid": False, "error": "Invalid API key"}, status_code=401)
        except Exception as exc:
            return JSONResponse({"valid": False, "error": str(exc)}, status_code=502)

    @app.post("/validate-deepgram-key")
    async def validate_deepgram_key(request: ValidateDeepgramKeyRequest) -> JSONResponse:
        key = request.deepgram_api_key.strip()
        if not key:
            return JSONResponse({"valid": False, "error": "No key provided"}, status_code=400)
        try:
            client = services.runtime_deepgram_tts(key)
            await client.validate()
            return JSONResponse({"valid": True})
        except Exception as exc:
            status, message = _deepgram_error_response(exc)
            return JSONResponse({"valid": False, "error": message}, status_code=status)

    @app.get("/credentials/status")
    async def credential_status(request: Request) -> JSONResponse:
        user = current_user_from_request(request)
        if user is None:
            return JSONResponse({"error": "Authentication required"}, status_code=401)
        anthropic_stored = auth_store.has_api_key(user.id, "anthropic")
        deepgram_stored = auth_store.has_api_key(user.id, "deepgram")
        return JSONResponse({
            "anthropic_configured": bool(settings.api_key) or anthropic_stored,
            "deepgram_configured": bool(settings.deepgram_api_key) or deepgram_stored,
            "anthropic_source": "env" if settings.api_key else ("stored" if anthropic_stored else None),
            "deepgram_source": "env" if settings.deepgram_api_key else ("stored" if deepgram_stored else None),
        })

    @app.post("/credentials/anthropic")
    async def save_anthropic_credential(request: Request, payload: SaveCredentialRequest) -> JSONResponse:
        csrf_error = _require_csrf(request, bind_to_session=True)
        if csrf_error is not None:
            return csrf_error
        user = current_user_from_request(request)
        if user is None:
            return JSONResponse({"saved": False, "error": "Authentication required"}, status_code=401)
        key = payload.api_key.strip()
        if not key:
            return JSONResponse({"saved": False, "error": "No key provided"}, status_code=400)
        try:
            client = AsyncAnthropic(api_key=key)
            await asyncio.wait_for(client.models.list(), timeout=15.0)
        except asyncio.TimeoutError:
            return JSONResponse({"saved": False, "error": "Validation timed out"}, status_code=504)
        except AuthenticationError:
            return JSONResponse({"saved": False, "error": "Invalid API key"}, status_code=401)
        except Exception as exc:
            return JSONResponse({"saved": False, "error": str(exc)}, status_code=502)

        auth_store.upsert_api_key(user.id, "anthropic", key)
        return JSONResponse({"saved": True, "configured": True, "source": "stored"})

    @app.delete("/credentials/anthropic")
    async def delete_anthropic_credential(request: Request) -> JSONResponse:
        csrf_error = _require_csrf(request, bind_to_session=True)
        if csrf_error is not None:
            return csrf_error
        user = current_user_from_request(request)
        if user is None:
            return JSONResponse({"error": "Authentication required"}, status_code=401)
        auth_store.delete_api_key(user.id, "anthropic")
        return JSONResponse({
            "deleted": True,
            "configured": bool(settings.api_key),
            "source": "env" if settings.api_key else None,
        })

    @app.post("/credentials/deepgram")
    async def save_deepgram_credential(request: Request, payload: SaveCredentialRequest) -> JSONResponse:
        csrf_error = _require_csrf(request, bind_to_session=True)
        if csrf_error is not None:
            return csrf_error
        user = current_user_from_request(request)
        if user is None:
            return JSONResponse({"saved": False, "error": "Authentication required"}, status_code=401)
        key = payload.api_key.strip()
        if not key:
            return JSONResponse({"saved": False, "error": "No key provided"}, status_code=400)
        try:
            client = services.runtime_deepgram_tts(key)
            await client.validate()
        except Exception as exc:
            status, message = _deepgram_error_response(exc)
            return JSONResponse({"saved": False, "error": message}, status_code=status)

        auth_store.upsert_api_key(user.id, "deepgram", key)
        return JSONResponse({"saved": True, "configured": True, "source": "stored"})

    @app.delete("/credentials/deepgram")
    async def delete_deepgram_credential(request: Request) -> JSONResponse:
        csrf_error = _require_csrf(request, bind_to_session=True)
        if csrf_error is not None:
            return csrf_error
        user = current_user_from_request(request)
        if user is None:
            return JSONResponse({"error": "Authentication required"}, status_code=401)
        auth_store.delete_api_key(user.id, "deepgram")
        return JSONResponse({
            "deleted": True,
            "configured": bool(settings.deepgram_api_key),
            "source": "env" if settings.deepgram_api_key else None,
        })

    @app.post("/preprocess")
    async def preprocess(request: Request) -> JSONResponse:
        if settings.deployed:
            return JSONResponse({"error": "Preprocess is disabled in deployed environments."}, status_code=403)
        csrf_error = _require_csrf(request, bind_to_session=False)
        if csrf_error is not None:
            return csrf_error
        manifest = services.preprocessor.run()
        return JSONResponse({"status": "ok", "manifest": manifest})

    @app.post("/chat")
    async def chat(request: Request, payload: ChatRequest) -> StreamingResponse:
        csrf_error = _require_csrf(request, bind_to_session=True)
        if csrf_error is not None:
            return csrf_error
        user = current_user_from_request(request)
        if user is None:
            return JSONResponse({"error": "Authentication required"}, status_code=401)
        services.ensure_cache()
        resolved_payload = payload.model_copy(update={
            "anthropic_api_key": payload.anthropic_api_key or anthropic_key_for_user(user) or None,
        })
        return StreamingResponse(
            services.stream_answer(resolved_payload),
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
    async def synthesize_speech(request: Request, payload: SpeechRequest) -> JSONResponse:
        csrf_error = _require_csrf(request, bind_to_session=True)
        if csrf_error is not None:
            return csrf_error
        user = current_user_from_request(request)
        if user is None:
            return JSONResponse({"error": "Authentication required"}, status_code=401)
        payload = await asyncio.to_thread(services.synthesize_speech, payload.text)
        if not payload:
            return JSONResponse({"status": "error", "message": "Local TTS is unavailable"}, status_code=503)
        return JSONResponse({"status": "ok", "audio_id": payload["audio_id"], "url": payload["url"]})

    @app.get("/speech/{audio_id}")
    async def speech_file(audio_id: str) -> Response:
        path = services.get_speech_path(audio_id)
        if not path:
            return JSONResponse({"status": "error", "message": "Speech not found"}, status_code=404)
        return FileResponse(path, media_type="audio/wav", filename=f"{audio_id}.wav")

    @app.post("/speech/stream")
    async def stream_speech_deepgram(request: Request, payload: SpeechRequest) -> Response:
        csrf_error = _require_csrf(request, bind_to_session=True)
        if csrf_error is not None:
            return csrf_error
        user = current_user_from_request(request)
        if user is None:
            return JSONResponse({"error": "Authentication required"}, status_code=401)
        key = (payload.deepgram_api_key or "").strip() or deepgram_key_for_user(user)
        tts = services.runtime_deepgram_tts(key or None)
        if not tts.enabled:
            return JSONResponse({"error": "Deepgram TTS not configured"}, status_code=503)

        try:
            audio = await tts.synthesize(payload.text)
        except Exception as exc:
            status, message = _deepgram_error_response(exc)
            return JSONResponse({"error": message}, status_code=status)

        return Response(
            content=audio,
            media_type="audio/mpeg",
            headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
        )

    @app.websocket("/ws/transcribe")
    async def ws_transcribe(websocket: WebSocket) -> None:
        await websocket.accept()
        user = current_user_from_websocket(websocket)
        if user is None:
            await websocket.send_json({"error": "Authentication required"})
            await websocket.close()
            return
        session_token = websocket.cookies.get(session_cookie_name)
        csrf_token = (websocket.query_params.get("csrf_token") or "").strip()
        if not auth_store.validate_session_csrf(session_token, csrf_token):
            await websocket.send_json({"error": "CSRF validation failed"})
            await websocket.close()
            return
        deepgram_api_key = (websocket.query_params.get("deepgram_api_key") or "").strip() or deepgram_key_for_user(user)
        if not deepgram_api_key:
            await websocket.send_json({"error": "Deepgram API key not configured"})
            await websocket.close()
            return

        dg_url = (
            "wss://api.deepgram.com/v1/listen"
            "?model=nova-2"
            "&language=en-US"
            "&encoding=linear16"
            "&sample_rate=16000"
            "&channels=1"
            "&interim_results=true"
            "&smart_format=true"
            "&endpointing=300"
            "&utterance_end_ms=1000"
        )

        try:
            async with websockets.connect(
                dg_url,
                additional_headers={"Authorization": f"Token {deepgram_api_key}"},
            ) as dg_ws:

                async def from_browser() -> None:
                    try:
                        while True:
                            msg = await websocket.receive()
                            if msg["type"] == "websocket.disconnect":
                                break
                            if msg.get("bytes"):
                                await dg_ws.send(msg["bytes"])
                            elif msg.get("text") == "EOS":
                                await dg_ws.send(json.dumps({"type": "CloseStream"}))
                                break
                    except Exception:
                        pass

                async def from_deepgram() -> None:
                    try:
                        async for raw in dg_ws:
                            if isinstance(raw, bytes):
                                continue
                            data = json.loads(raw)
                            msg_type = data.get("type")
                            if msg_type == "Results":
                                alts = data.get("channel", {}).get("alternatives", [])
                                transcript = alts[0].get("transcript", "") if alts else ""
                                is_final = data.get("is_final", False)
                                speech_final = data.get("speech_final", False)
                                if transcript or speech_final:
                                    await websocket.send_json({
                                        "transcript": transcript,
                                        "is_final": is_final,
                                        "speech_final": speech_final,
                                    })
                            elif msg_type == "UtteranceEnd":
                                await websocket.send_json({
                                    "transcript": "",
                                    "is_final": True,
                                    "speech_final": True,
                                })
                    except Exception:
                        pass

                await asyncio.gather(from_browser(), from_deepgram(), return_exceptions=True)

        except Exception as exc:
            _, message = _deepgram_error_response(exc)
            with contextlib.suppress(Exception):
                await websocket.send_json({"error": message})
        finally:
            with contextlib.suppress(Exception):
                await websocket.close()

    if frontend_assets_dir.exists():
        app.mount("/assets", StaticFiles(directory=frontend_assets_dir), name="frontend-assets")

    @app.get("/")
    async def frontend_root() -> Response:
        if frontend_available():
            return frontend_index_response()
        return JSONResponse(
            {
                "status": "ok",
                "message": "Frontend build not found. Run `cd frontend && npm run build` for same-origin serving.",
            },
            status_code=503,
        )

    @app.get("/{full_path:path}")
    async def frontend_spa_fallback(full_path: str) -> Response:
        if not frontend_available():
            return JSONResponse({"error": "Not found"}, status_code=404)

        requested_path = Path(full_path)
        if requested_path.name and "." in requested_path.name:
            candidate = (frontend_dist_dir / requested_path).resolve()
            frontend_root = frontend_dist_dir.resolve()
            if candidate.is_file() and candidate.is_relative_to(frontend_root):
                return FileResponse(candidate)
            return JSONResponse({"error": "Not found"}, status_code=404)

        return frontend_index_response()

    return app
