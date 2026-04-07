from __future__ import annotations

import json
import re
from typing import Any, AsyncGenerator

from anthropic import AsyncAnthropic

from app.config import Settings

try:
    from claude_agent_sdk import ClaudeAgentOptions, query
    from claude_agent_sdk import AssistantMessage as SDKAssistantMessage
    from claude_agent_sdk import TextBlock as SDKTextBlock
except Exception:  # pragma: no cover - dependency may be absent before install
    ClaudeAgentOptions = None
    SDKAssistantMessage = None
    SDKTextBlock = None
    query = None


class ClaudeAgent:
    def __init__(self, settings: Settings, model: str, system_prompt: str) -> None:
        self.settings = settings
        self.model = model
        self.system_prompt = system_prompt
        self.client = AsyncAnthropic(api_key=settings.api_key) if settings.anthropic_enabled else None

    @property
    def enabled(self) -> bool:
        return self.client is not None

    async def stream_text(
        self,
        user_content: list[dict[str, Any]] | str,
        history: list[dict] | None = None,
        max_tokens: int = 1200,
    ) -> AsyncGenerator[str, None]:
        ''''
        Stream (yielding as they arrive) the text content of the agent's response,
        given the user content and optional conversation history. 
        If the Anthropic API is not configured, yields a single message indicating
        that. If the content is a simple string and the Claude Agent SDK is available,
        uses the SDK for a richer experience; otherwise falls back to direct API calls.
        '''

        if not self.enabled:
            yield "[Anthropic API not configured - set ANTHROPIC_API_KEY in .env]"
            return

        async for text in self._anthropic_stream_text(user_content, history, max_tokens):
            yield text

    async def complete_json(
        self,
        user_content: list[dict[str, Any]] | str,
        max_tokens: int = 1400,
    ) -> dict[str, Any] | None:
        if not self.enabled:
            return None

        raw = await self.complete_text(user_content, max_tokens=max_tokens)
        if not raw:
            return None

        parsed = self._parse_json_text(raw)
        if parsed is not None:
            return parsed
        return {"raw": raw}

    async def complete_text(
        self,
        user_content: list[dict[str, Any]] | str,
        max_tokens: int = 2000,
    ) -> str | None:
        if not self.enabled:
            return None

        return await self._anthropic_complete_text(user_content, max_tokens=max_tokens)

    async def _sdk_complete_text(self, prompt: str, max_tokens: int) -> str | None:
        chunks: list[str] = []
        async for chunk in self._sdk_stream(prompt, max_tokens=max_tokens):
            chunks.append(chunk)
        raw = "".join(chunks).strip()
        return raw or None

    async def _sdk_stream(self, prompt: str, max_tokens: int) -> AsyncGenerator[str, None]:
        options = ClaudeAgentOptions(
            system_prompt=self.system_prompt,
            model=self.model,
            cwd=str(self.settings.project_root),
            max_turns=max(1, min(8, max_tokens // 250)),
        )
        async for message in query(prompt=prompt, options=options):
            if not isinstance(message, SDKAssistantMessage):
                continue
            for block in message.content:
                if isinstance(block, SDKTextBlock) and block.text:
                    yield block.text

    async def _anthropic_stream_text(
        self,
        user_content: list[dict[str, Any]] | str,
        history: list[dict] | None,
        max_tokens: int,
    ) -> AsyncGenerator[str, None]:
        content = user_content if isinstance(user_content, list) else [{"type": "text", "text": user_content}]
        messages = list(history or []) + [{"role": "user", "content": content}]
        async with self.client.messages.stream(
            model=self.model,
            max_tokens=max_tokens,
            system=self.system_prompt,
            messages=messages,
        ) as stream:
            async for text in stream.text_stream:
                yield text

    async def _anthropic_complete_text(
        self,
        user_content: list[dict[str, Any]] | str,
        max_tokens: int,
    ) -> str | None:
        content = user_content if isinstance(user_content, list) else [{"type": "text", "text": user_content}]
        response = await self.client.messages.create(
            model=self.model,
            max_tokens=max_tokens,
            system=self.system_prompt,
            messages=[{"role": "user", "content": content}],
        )
        text_parts = [block.text for block in response.content if getattr(block, "type", "") == "text"]
        raw = "\n".join(text_parts).strip()
        return raw or None

    def _sdk_prompt(self, user_content: str, history: list[dict] | None = None) -> str:
        if not history:
            return user_content

        lines = ["Conversation so far:"]
        for msg in history:
            role = str(msg.get("role", "user")).upper()
            content = msg.get("content", "")
            if isinstance(content, list):
                text_parts = [
                    part.get("text", "")
                    for part in content
                    if isinstance(part, dict) and part.get("type") == "text"
                ]
                content = "\n".join(part for part in text_parts if part)
            lines.append(f"{role}: {str(content).strip()}")
        lines.append("")
        lines.append("Current user request:")
        lines.append(user_content)
        return "\n".join(lines)

    def _parse_json_text(self, raw: str) -> dict[str, Any] | None:
        try:
            return json.loads(raw)
        except json.JSONDecodeError:
            pass
        fenced = re.search(r"```(?:json)?\s*(\{.*\})\s*```", raw, re.S)
        if fenced:
            try:
                return json.loads(fenced.group(1))
            except json.JSONDecodeError:
                return None
        inline = re.search(r"(\{.*\})", raw, re.S)
        if inline:
            try:
                return json.loads(inline.group(1))
            except json.JSONDecodeError:
                return None
        return None
