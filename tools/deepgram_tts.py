from __future__ import annotations

from typing import AsyncIterator

import httpx


class DeepgramTTS:
    BASE_URL = "https://api.deepgram.com/v1/speak"

    def __init__(self, api_key: str, voice_model: str = "aura-asteria-en") -> None:
        self.api_key = api_key
        self.voice_model = voice_model

    @property
    def enabled(self) -> bool:
        return bool(self.api_key)

    async def validate(self, text: str = "test") -> None:
        if not self.enabled:
            raise ValueError("Deepgram API key not configured")

        async with httpx.AsyncClient(timeout=30.0) as client:
            response = await client.post(
                f"{self.BASE_URL}?model={self.voice_model}&encoding=mp3",
                headers={
                    "Authorization": f"Token {self.api_key}",
                    "Content-Type": "application/json",
                },
                json={"text": text},
            )
            response.raise_for_status()

    async def synthesize(self, text: str) -> bytes:
        if not self.enabled:
            raise ValueError("Deepgram API key not configured")

        async with httpx.AsyncClient(timeout=30.0) as client:
            response = await client.post(
                f"{self.BASE_URL}?model={self.voice_model}&encoding=mp3",
                headers={
                    "Authorization": f"Token {self.api_key}",
                    "Content-Type": "application/json",
                },
                json={"text": text},
            )
            response.raise_for_status()
            return response.content

    async def stream(self, text: str) -> AsyncIterator[bytes]:
        if not self.enabled:
            return

        async with httpx.AsyncClient(timeout=30.0) as client:
            async with client.stream(
                "POST",
                f"{self.BASE_URL}?model={self.voice_model}&encoding=mp3",
                headers={
                    "Authorization": f"Token {self.api_key}",
                    "Content-Type": "application/json",
                },
                json={"text": text},
            ) as response:
                response.raise_for_status()
                async for chunk in response.aiter_bytes(chunk_size=4096):
                    yield chunk
