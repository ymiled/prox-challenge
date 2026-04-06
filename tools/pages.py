from __future__ import annotations

from pathlib import Path

from app.config import Settings
from app.models import PageRef
from app.utils import encode_file_base64


class PageStore:
    def __init__(self, settings: Settings) -> None:
        self.settings = settings

    def get_page_image_path(self, ref: PageRef) -> Path:
        return self.settings.pages_dir / ref.doc / f"{ref.page:03d}.png"

    def get_page_image_path_by_parts(self, doc: str, page: int) -> Path:
        return self.settings.pages_dir / doc / f"{page:03d}.png"

    def get_page_image_base64(self, ref: PageRef) -> str | None:
        path = self.get_page_image_path(ref)
        if not path.exists():
            return None
        return encode_file_base64(path)
