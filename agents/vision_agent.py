from __future__ import annotations

import json
from datetime import datetime, timezone

from app.config import Settings
from app.models import PageRef, VisionResult
from app.prompts import VISION_PROMPT
from app.utils import read_json, safe_excerpt, write_json
from tools import PageStore

from .base import ClaudeAgent

VISION_MODEL = "claude-sonnet-4-6"
GENERIC_VISION_PROMPT = (
    "Describe this welding manual page in detail. Identify all labels, diagrams, "
    "settings tables, socket labels, polarity markings, control positions, and any "
    "visual information a user might ask about. "
    "Return valid JSON with keys: summary (string), extracted (object with all found labels/values)."
)


class VisionAgent(ClaudeAgent):
    CACHE_FILENAME = "vision_cache.json"

    def __init__(self, settings: Settings, page_store: PageStore) -> None:
        super().__init__(settings, VISION_MODEL, VISION_PROMPT)
        self.page_store = page_store
        self._vision_cache: dict[str, dict] = {}
        self._vision_cache_path = settings.structured_dir / self.CACHE_FILENAME
        if settings.vision_cache_enabled:
            self._load_vision_cache()

    def _load_vision_cache(self) -> None:
        try:
            self._vision_cache = read_json(self._vision_cache_path, {})
        except Exception:
            self._vision_cache = {}

    def _cache_key(self, page: PageRef) -> str:
        return f"{page.doc}:{page.page}"

    def _result_from_cache(self, question: str, pages: list[PageRef]) -> VisionResult:
        summary_parts: list[str] = []
        extracted: dict = {}
        resolved: list[PageRef] = []
        for p in pages[:3]:
            entry = self._vision_cache.get(self._cache_key(p), {})
            if entry.get("summary"):
                summary_parts.append(entry["summary"])
            extracted.update(entry.get("extracted", {}))
            resolved.append(p)
        return VisionResult(
            summary=" ".join(summary_parts) or "Cached visual analysis.",
            relevant_pages=resolved,
            extracted=extracted,
        )

    async def run_single_page(self, page: PageRef) -> VisionResult:
        """Run generic vision extraction on one page (used for preprocessing cache build)."""
        # Return from cache if already processed
        if self.settings.vision_cache_enabled and self._cache_key(page) in self._vision_cache:
            entry = self._vision_cache[self._cache_key(page)]
            return VisionResult(
                summary=entry.get("summary", ""),
                relevant_pages=[page],
                extracted=entry.get("extracted", {}),
            )
        content: list[dict] = [{"type": "text", "text": GENERIC_VISION_PROMPT}]
        encoded = self.page_store.get_page_image_base64(page)
        if encoded:
            content.append({"type": "image", "source": {"type": "base64", "media_type": "image/png", "data": encoded}})
        parsed = await self.complete_json(content)
        if not parsed:
            return VisionResult(summary="", relevant_pages=[page], extracted={})
        result = VisionResult(
            summary=parsed.get("summary", ""),
            relevant_pages=[page],
            extracted=parsed.get("extracted", {}),
        )
        if self.settings.vision_cache_enabled:
            self._update_cache_entry(page, result)
        return result

    def _save_vision_cache(self) -> None:
        try:
            write_json(self._vision_cache_path, self._vision_cache)
        except Exception:
            pass

    def _update_cache_entry(self, page: PageRef, result: VisionResult) -> None:
        key = self._cache_key(page)
        self._vision_cache[key] = {
            "summary": result.summary,
            "extracted": result.extracted,
        }

    async def run(self, question: str, pages: list[PageRef]) -> VisionResult:
        if not pages:
            return VisionResult(summary="No visual pages were available.", relevant_pages=[], extracted={})

        # Return from cache if all requested pages are already cached
        if self.settings.vision_cache_enabled and self._vision_cache:
            target_pages = pages[:3]
            if all(self._cache_key(p) in self._vision_cache for p in target_pages):
                return self._result_from_cache(question, pages)

        if not self.enabled:
            return VisionResult(
                summary="Visual analysis is configured but Anthropic is unavailable, so returning page references only.",
                relevant_pages=pages[:3],
                extracted={"fallback": True},
            )

        content: list[dict] = [
            {
                "type": "text",
                "text": (
                    "Analyze the manual pages for this user question.\n"
                    f"Question: {question}\n"
                    f"Pages: {json.dumps([page.model_dump() for page in pages], indent=2)}\n"
                    "Return valid JSON."
                ),
            }
        ]
        for page in pages[:3]:
            encoded = self.page_store.get_page_image_base64(page)
            if not encoded:
                continue
            content.append(
                {
                    "type": "image",
                    "source": {
                        "type": "base64",
                        "media_type": "image/png",
                        "data": encoded,
                    },
                }
            )
        parsed = await self.complete_json(content)
        if not parsed:
            return VisionResult(
                summary="Vision agent did not return a response.",
                relevant_pages=pages[:3],
                extracted={},
            )
        raw_pages = parsed.get("relevant_pages") or []
        resolved_pages = []
        for item in raw_pages:
            if isinstance(item, dict) and "doc" in item and "page" in item:
                resolved_pages.append(PageRef(**item))
            elif isinstance(item, int):
                matched = next((page for page in pages if page.page == item), None)
                if matched:
                    resolved_pages.append(matched)

        def dedupe(refs: list[PageRef]) -> list[PageRef]:
            seen: set[tuple[str, int]] = set()
            out: list[PageRef] = []
            for p in refs:
                k = (p.doc, p.page)
                if k in seen:
                    continue
                seen.add(k)
                out.append(p)
            return out

        final_pages = dedupe(resolved_pages) or dedupe(pages[:3])
        return VisionResult(
            summary=parsed.get("summary", safe_excerpt(str(parsed))),
            relevant_pages=final_pages,
            extracted=parsed.get("extracted", parsed),
        )
