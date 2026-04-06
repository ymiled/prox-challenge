from __future__ import annotations

from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from app.config import get_settings

try:
    from sentence_transformers import SentenceTransformer
except Exception as exc:  # pragma: no cover
    raise SystemExit(f"sentence-transformers is not installed: {exc}")


def main() -> None:
    settings = get_settings()
    target = Path(settings.semantic_embed_model_path)
    target.parent.mkdir(parents=True, exist_ok=True)

    if target.exists():
        print(f"Model cache already exists at {target}")
        return

    print(f"Downloading {settings.semantic_embed_model} to {target} ...")
    model = SentenceTransformer(settings.semantic_embed_model)
    model.save(str(target))
    print(f"Saved embedding model to {target}")


if __name__ == "__main__":
    main()
