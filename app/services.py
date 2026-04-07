from __future__ import annotations

import copy
import json
import traceback
from typing import AsyncIterator

from anthropic import AsyncAnthropic

from agents import ArtifactAgent, DiagnosticAgent, OrchestratorAgent, RetrievalAgent, VisionAgent
from app.config import Settings
from app.models import ChatRequest, PageRef
from app.response_cache import ResponseCache
from preprocessing import Preprocessor
from tools import LocalEmbeddingClient, LocalTTS, ManualSearchEngine, PageStore


class AppServices:
    def __init__(self, settings: Settings) -> None:
        self.settings = settings
        self.startup_status: dict = {
            "model_validation": {
                "validated": False,
                "reason": "not_run",
            }
        }
        self.preprocessor = Preprocessor(settings)
        self.embedding_client = LocalEmbeddingClient(settings)
        self.tts = LocalTTS(settings)
        self.search_engine = ManualSearchEngine(settings, self.embedding_client)
        self.page_store = PageStore(settings)
        self.retrieval_agent = RetrievalAgent(self.search_engine)
        self.vision_agent = VisionAgent(settings, self.page_store)
        self.diagnostic_agent = DiagnosticAgent(settings)
        self.artifact_agent = ArtifactAgent(settings)
        self.artifact_store: dict[str, dict] = {}
        self._response_cache: ResponseCache | None = (
            ResponseCache(settings.response_cache_max_size) if settings.response_cache_enabled else None
        )
        self.orchestrator = OrchestratorAgent(
            settings=self.settings,
            retrieval_agent=self.retrieval_agent,
            vision_agent=self.vision_agent,
            diagnostic_agent=self.diagnostic_agent,
            artifact_agent=self.artifact_agent,
            page_store=self.page_store,
        )

    async def validate_anthropic_models(self) -> dict:
        if not self.settings.anthropic_enabled:
            status = {"validated": False, "reason": "anthropic_disabled"}
            self.startup_status["model_validation"] = status
            return status

        client = AsyncAnthropic(api_key=self.settings.api_key)
        response = await client.models.list()
        available = {model.id for model in response.data}
        configured = self.settings.configured_anthropic_models
        missing = {role: model for role, model in configured.items() if model not in available}

        if missing:
            available_sorted = sorted(available)
            missing_lines = ", ".join(f"{role}={model}" for role, model in missing.items())
            raise RuntimeError(
                "Configured Anthropic models are not available for this API key: "
                f"{missing_lines}. Available models: {available_sorted}"
            )

        status = {
            "validated": True,
            "configured": configured,
            "available_count": len(available),
        }
        self.startup_status["model_validation"] = status
        return status

    def ensure_cache(self) -> dict:
        manifest = {}
        if self._cache_refresh_required():
            manifest = self.preprocessor.run()
            self.search_engine = ManualSearchEngine(self.settings, self.embedding_client)
            self.retrieval_agent = RetrievalAgent(self.search_engine)
            self.orchestrator = OrchestratorAgent(
                settings=self.settings,
                retrieval_agent=self.retrieval_agent,
                vision_agent=self.vision_agent,
                diagnostic_agent=self.diagnostic_agent,
                artifact_agent=self.artifact_agent,
                page_store=self.page_store,
            )
        return manifest

    def _cache_refresh_required(self) -> bool:
        if not self.settings.cache_ready_file.exists():
            return True

        manifest = self._read_manifest()
        if manifest.get("cache_version") != Preprocessor.CACHE_VERSION:
            return True

        page_index = self.search_engine.page_index
        return any("full_text" not in entry for entry in page_index[:3])

    def _read_manifest(self) -> dict:
        path = self.settings.structured_dir / "manifest.json"
        if not path.exists():
            return {}
        try:
            return json.loads(path.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            return {}

    async def stream_answer(self, request: ChatRequest) -> AsyncIterator[str]:
        # Only cache when no per-request API key is supplied (shared key = deterministic context)
        use_cache = self._response_cache is not None and not (request.anthropic_api_key or "").strip()
        cache_key: str | None = None

        if use_cache:
            page_keys = [e.get("doc", "") + ":" + str(e.get("page", "")) for e in self.search_engine.page_index]
            cache_key = self._response_cache.make_key(request.message, page_keys)
            cached = self._response_cache.get(cache_key)
            if cached is not None:
                for event_str in cached:
                    yield event_str
                return

        collected: list[str] = []

        try:
            orchestrator = self._runtime_orchestrator(request.anthropic_api_key)
            async for event in orchestrator.stream(request.message, request.history, request.voice_mode):
                event_type = event.get("type")

                if event_type == "text_delta":
                    evt = self._event("text_delta", {"content": event["content"]})

                elif event_type == "image":
                    payload = {
                        "doc": event["doc"],
                        "page": event["page"],
                        "caption": event["caption"],
                        "url": event["url"],
                    }
                    if request.include_image_data:
                        ref = PageRef(doc=event["doc"], page=event["page"])
                        encoded = self.page_store.get_page_image_base64(ref)
                        if encoded:
                            payload["src"] = f"data:image/png;base64,{encoded}"
                    evt = self._event("image", payload)

                elif event_type == "artifact":
                    artifact = event["artifact"]
                    self.artifact_store[artifact.artifact_id] = {
                        "artifact_type": artifact.artifact_type,
                        "title": artifact.title,
                        "content": artifact.content,
                    }
                    evt = self._event("artifact", {
                        "artifact": {
                            "artifact_id": artifact.artifact_id,
                            "artifact_type": artifact.artifact_type,
                            "title": artifact.title,
                            "content": artifact.content,
                        }
                    })

                elif event_type == "done":
                    evt = self._event("done", {
                        "citations": event.get("citations", []),
                        "debug": event.get("debug", {}),
                    })

                else:
                    evt = self._event(event_type or "unknown", {k: v for k, v in event.items() if k != "type"})

                if use_cache:
                    collected.append(evt)
                yield evt

            if use_cache and cache_key and collected:
                self._response_cache.put(cache_key, collected)

        except Exception as exc:
            yield self._event("error", {
                "message": str(exc),
                "traceback": traceback.format_exc(),
            })
            yield self._event("done", {"citations": [], "debug": {"error": True}})

    def _event(self, event_type: str, payload: dict) -> str:
        return f"data: {json.dumps({'type': event_type, **payload})}\n\n"

    def _runtime_orchestrator(self, api_key: str | None) -> OrchestratorAgent:
        cleaned_key = (api_key or "").strip()
        if not cleaned_key:
            return self.orchestrator

        runtime_settings = copy.copy(self.settings)
        runtime_settings.api_key = cleaned_key
        diagnostic_agent = DiagnosticAgent(runtime_settings)
        artifact_agent = ArtifactAgent(runtime_settings)
        vision_agent = VisionAgent(runtime_settings, self.page_store)
        return OrchestratorAgent(
            settings=runtime_settings,
            retrieval_agent=self.retrieval_agent,
            vision_agent=vision_agent,
            diagnostic_agent=diagnostic_agent,
            artifact_agent=artifact_agent,
            page_store=self.page_store,
        )

    async def build_vision_cache(self) -> int:
        """Run VisionAgent on every visual page not yet in the cache. Returns count of pages processed."""
        if not self.settings.vision_cache_enabled or not self.vision_agent.enabled:
            return 0
        processed = 0
        for entry in self.search_engine.page_index:
            page = PageRef(doc=entry["doc"], page=entry["page"])
            if self.vision_agent._cache_key(page) in self.vision_agent._vision_cache:
                continue
            topics = entry.get("topics", [])
            # Only process visually-rich pages
            if not any(t in topics for t in ("polarity", "troubleshooting", "wire_settings", "duty_cycle", "visual")):
                continue
            try:
                await self.vision_agent.run_single_page(page)
                processed += 1
            except Exception:
                pass
        if processed:
            self.vision_agent._save_vision_cache()
        return processed

    def get_artifact(self, artifact_id: str) -> dict | None:
        return self.artifact_store.get(artifact_id)

    def synthesize_speech(self, text: str) -> dict | None:
        if not self.settings.local_tts_enabled:
            return None
        path = self.tts.synthesize(text)
        if not path:
            return None
        return {
            "audio_id": path.stem,
            "url": f"/speech/{path.stem}",
            "path": path,
        }

    def get_speech_path(self, audio_id: str):
        path = self.settings.audio_dir / f"{audio_id}.wav"
        if path.exists():
            return path
        return None

    def _artifact_block(self, artifact) -> str:
        media_type = self._artifact_media_type(artifact.artifact_type)
        return (
            f'<antArtifact identifier="{artifact.artifact_id}" '
            f'type="{media_type}" title="{artifact.title}">\n'
            f"{artifact.content}\n"
            f"</antArtifact>"
        )

    def _artifact_media_type(self, artifact_type: str) -> str:
        mapping = {
            "svg": "image/svg+xml",
            "react": "application/vnd.ant.react",
            "html": "text/html",
            "code": "application/vnd.ant.code",
            "markdown": "text/markdown",
            "mermaid": "application/vnd.ant.mermaid",
            "json": "application/json",
        }
        return mapping.get(artifact_type, "text/plain")
