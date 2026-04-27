from functools import lru_cache
import importlib.util
from pathlib import Path
import os
import secrets

from dotenv import load_dotenv


load_dotenv(override=True)


class Settings:
    ORCHESTRATOR_MODEL = "claude-opus-4-6"
    VISION_MODEL = "claude-sonnet-4-6"
    DIAGNOSTIC_MODEL = "claude-opus-4-6"
    ARTIFACT_MODEL = "claude-haiku-4-5-20251001"
    SEMANTIC_EMBED_MODEL = "BAAI/bge-small-en-v1.5"

    def __init__(self) -> None:
        root = Path(__file__).resolve().parent.parent
        self.project_root = root
        self.frontend_dir = root / "frontend"
        self.frontend_dist_dir = self.frontend_dir / "dist"
        self.files_dir = root / "files"
        self.cache_dir = root / "cache"
        self.pages_dir = self.cache_dir / "pages"
        self.audio_dir = self.cache_dir / "audio"
        self.text_dir = self.cache_dir / "text"
        self.structured_dir = self.cache_dir / "structured"
        self.credentials_file = self.cache_dir / "credentials.json"
        self.database_file = self.cache_dir / "app.db"
        self.database_url = os.getenv("DATABASE_URL", "").strip()
        self.api_key = os.getenv("ANTHROPIC_API_KEY", "").strip()
        self.deepgram_api_key = (
            os.getenv("DEEPGRAM_API_KEY", "").strip()
            or os.getenv("DEEPGRAM_KEY", "").strip()
        )
        self.deepgram_voice_model = os.getenv("DEEPGRAM_VOICE_MODEL", "aura-asteria-en").strip()
        self.orchestrator_model = self.ORCHESTRATOR_MODEL
        self.vision_model = self.VISION_MODEL
        self.diagnostic_model = self.DIAGNOSTIC_MODEL
        self.artifact_model = self.ARTIFACT_MODEL
        self.semantic_embed_model = self.SEMANTIC_EMBED_MODEL
        self.models_dir = root / "models"
        default_embed_path = self.models_dir / "semantic" / self.semantic_embed_model.replace("/", "__")
        self.semantic_embed_model_path = default_embed_path
        self.max_search_results = 6
        self.retrieval_candidate_pool = 12
        self.frontend_origin = os.getenv("FRONTEND_ORIGIN", "http://localhost:5173").strip()
        extra_origins_raw = os.getenv("CORS_ALLOWED_ORIGINS", "").strip()
        self.cors_allowed_origins = [
            origin.strip()
            for origin in ([self.frontend_origin, "http://localhost:3000", "http://127.0.0.1:5173"] + extra_origins_raw.split(","))
            if origin.strip()
        ]
        self.strict_startup_validation = os.getenv("STRICT_STARTUP_VALIDATION", "false").strip().lower() == "true"
        self.deployment_env = os.getenv("DEPLOYMENT_ENV", "local").strip().lower() or "local"
        self.deployed = self.deployment_env != "local"
        self.local_tts_enabled = os.getenv(
            "ENABLE_LOCAL_TTS",
            "false" if self.deployed else "true",
        ).strip().lower() == "true"
        self.semantic_search_enabled = os.getenv(
            "ENABLE_SEMANTIC_SEARCH",
            "false" if self.deployed else "true",
        ).strip().lower() == "true"
        self.render_dpi = int(
            os.getenv(
                "PAGE_RENDER_DPI",
                "120" if self.deployed else "180",
            ).strip()
        )
        # Hybrid RAG: dense (embeddings) vs sparse (BM25 over page index)
        self.semantic_weight = float(os.getenv("RAG_SEMANTIC_WEIGHT", "0.45"))
        self.sparse_weight = float(os.getenv("RAG_SPARSE_WEIGHT", os.getenv("RAG_LEXICAL_WEIGHT", "0.55")))
        # Cross-encoder reranking
        self.cross_encoder_enabled = os.getenv("ENABLE_CROSS_ENCODER", "true").strip().lower() == "true"
        self.cross_encoder_model = "cross-encoder/ms-marco-MiniLM-L-6-v2"
        self.cross_encoder_model_path = self.models_dir / "cross-encoder" / self.cross_encoder_model.replace("/", "__")
        # Contextual compression
        self.contextual_compression_enabled = os.getenv("ENABLE_CONTEXTUAL_COMPRESSION", "true").strip().lower() == "true"
        self.compression_max_sentences = int(os.getenv("COMPRESSION_MAX_SENTENCES", "6"))
        # Vision caching
        self.vision_cache_enabled = os.getenv("ENABLE_VISION_CACHE", "true").strip().lower() == "true"
        # Response caching
        self.response_cache_enabled = os.getenv("ENABLE_RESPONSE_CACHE", "true").strip().lower() == "true"
        self.response_cache_max_size = int(os.getenv("RESPONSE_CACHE_MAX_SIZE", "200"))
        self.claude_sdk_enabled = os.getenv("ENABLE_CLAUDE_AGENT_SDK", "false").strip().lower() == "true"
        self.app_secret_key = os.getenv("APP_SECRET_KEY", "local-dev-only-secret-change-me").strip()
        self.session_ttl_days = int(os.getenv("SESSION_TTL_DAYS", "30").strip())
        self.session_cookie_name = "__Host-vulcan_auth_session" if self.deployed else "vulcan_auth_session"
        self.csrf_cookie_name = "__Host-vulcan_csrf_token" if self.deployed else "vulcan_csrf_token"
        self.csrf_header_name = "X-CSRF-Token"
        self.session_cookie_samesite = self._normalize_samesite(
            os.getenv("SESSION_COOKIE_SAMESITE", "lax").strip()
        )
        self.auth_rate_limit_window_seconds = int(os.getenv("AUTH_RATE_LIMIT_WINDOW_SECONDS", "900").strip())
        self.auth_rate_limit_max_attempts = int(os.getenv("AUTH_RATE_LIMIT_MAX_ATTEMPTS", "8").strip())

    @property
    def cache_ready_file(self) -> Path:
        return self.structured_dir / "page_index.json"

    @property
    def anthropic_enabled(self) -> bool:
        return bool(self.api_key)

    @property
    def deepgram_enabled(self) -> bool:
        return bool(self.deepgram_api_key)

    @property
    def claude_sdk_installed(self) -> bool:
        return importlib.util.find_spec("claude_agent_sdk") is not None

    @property
    def sentence_transformers_installed(self) -> bool:
        return importlib.util.find_spec("sentence_transformers") is not None

    @property
    def configured_anthropic_models(self) -> dict[str, str]:
        return {
            "orchestrator": self.orchestrator_model,
            "vision": self.vision_model,
            "diagnostic": self.diagnostic_model,
            "artifact": self.artifact_model,
        }

    @property
    def app_secret_key_looks_secure(self) -> bool:
        if not self.app_secret_key or self.app_secret_key == "local-dev-only-secret-change-me":
            return False
        if len(self.app_secret_key) < 32:
            return False
        return secrets.compare_digest(self.app_secret_key.strip(), self.app_secret_key) and bool(self.app_secret_key)

    def _normalize_samesite(self, value: str) -> str:
        lowered = value.lower()
        if lowered not in {"lax", "strict", "none"}:
            return "lax"
        if lowered == "none" and not self.deployed:
            return "lax"
        return lowered


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    return Settings()
