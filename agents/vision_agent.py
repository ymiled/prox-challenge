from __future__ import annotations

import json

from app.config import Settings
from app.models import PageRef, VisionResult
from app.prompts import VISION_PROMPT
from app.utils import safe_excerpt
from tools import PageStore

from .base import ClaudeAgent

VISION_MODEL = "claude-sonnet-4-6"


class VisionAgent(ClaudeAgent):
    def __init__(self, settings: Settings, page_store: PageStore) -> None:
        super().__init__(settings, VISION_MODEL, VISION_PROMPT)
        self.page_store = page_store

    async def run(self, question: str, pages: list[PageRef]) -> VisionResult:
        if not pages:
            return VisionResult(summary="No visual pages were available.", relevant_pages=[], extracted={})

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
