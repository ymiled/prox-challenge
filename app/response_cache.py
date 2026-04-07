from __future__ import annotations

import hashlib
import json
from collections import OrderedDict


class ResponseCache:
    """
    LRU in-memory cache for complete SSE response streams.

    Keys are SHA-256 hashes of (message, sorted page keys).
    Values are the list of SSE event strings exactly as yielded by
    AppServices.stream_answer, so replays are identical to live responses.
    """

    def __init__(self, max_size: int = 200) -> None:
        self.max_size = max_size
        self._store: OrderedDict[str, list[str]] = OrderedDict()

    def make_key(self, message: str, page_keys: list[str]) -> str:
        payload = json.dumps({"message": message, "pages": sorted(page_keys)}, sort_keys=True)
        return hashlib.sha256(payload.encode()).hexdigest()

    def get(self, key: str) -> list[str] | None:
        if key not in self._store:
            return None
        self._store.move_to_end(key)
        return self._store[key]

    def put(self, key: str, events: list[str]) -> None:
        if key in self._store:
            self._store.move_to_end(key)
        self._store[key] = events
        if len(self._store) > self.max_size:
            self._store.popitem(last=False)

    def __len__(self) -> int:
        return len(self._store)
