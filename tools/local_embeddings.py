from __future__ import annotations

from pathlib import Path
from typing import Any

from app.config import Settings

try:
    from sentence_transformers import SentenceTransformer
except Exception:  # pragma: no cover - optional dependency until installed
    SentenceTransformer = None

SEMANTIC_EMBED_MODEL = "BAAI/bge-small-en-v1.5"


class LocalEmbeddingClient:
    def __init__(self, settings: Settings) -> None:
        self.settings = settings
        self.model_name = SEMANTIC_EMBED_MODEL
        self.model_path = settings.models_dir / "semantic" / self.model_name.replace("/", "__")
        self._model: SentenceTransformer | None = None
        self.disabled_reason: str | None = None

    @property
    def enabled(self) -> bool:
        return (
            self.settings.semantic_search_enabled
            and SentenceTransformer is not None
            and self.disabled_reason is None
        )

    def embed_texts(self, texts: list[str]) -> list[list[float]]:
        if not self.enabled or not texts:
            return []
        model = self._load_model()
        if not model:
            return []
        try:
            vectors = model.encode(
                texts,
                normalize_embeddings=True,
                show_progress_bar=False,
                batch_size=16,
            )
        except Exception as exc:  # pragma: no cover - model/runtime failures
            self.disabled_reason = str(exc)
            return []
        return [[float(value) for value in vector] for vector in vectors]

    def embed_query(self, query: str) -> list[float]:
        vectors = self.embed_texts([query])
        return vectors[0] if vectors else []

    def _load_model(self) -> SentenceTransformer | None:
        if self._model is not None:
            return self._model
        if SentenceTransformer is None:
            return None
        try:
            source = self._model_source()
            self._model = SentenceTransformer(source)
        except Exception as exc:  # pragma: no cover - model download/load failures
            self.disabled_reason = str(exc)
            return None
        return self._model

    def _model_source(self) -> str:
        local_path = Path(self.model_path)
        if local_path.exists():
            return str(local_path)
        return self.model_name
