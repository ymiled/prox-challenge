from __future__ import annotations

from app.models import RetrievalResult
from tools import ManualSearchEngine


class RetrievalAgent:
    def __init__(self, search_engine: ManualSearchEngine) -> None:
        self.search_engine = search_engine

    async def run(self, query: str) -> RetrievalResult:
        return self.search_engine.search(query)
