from typing import Any, Literal

from pydantic import BaseModel, Field


class ChatMessage(BaseModel):
    role: Literal["user", "assistant", "system"]
    content: str


class ChatRequest(BaseModel):
    message: str = Field(min_length=1)
    conversation_id: str | None = None
    history: list[ChatMessage] = Field(default_factory=list)
    include_image_data: bool = False
    voice_mode: bool = False
    anthropic_api_key: str | None = Field(default=None, min_length=1)


class SpeechRequest(BaseModel):
    text: str = Field(min_length=1)
    deepgram_api_key: str | None = Field(default=None, min_length=1)


class PageRef(BaseModel):
    doc: str
    page: int
    score: float | None = None
    section: str | None = None
    excerpt: str | None = None


class RetrievalResult(BaseModel):
    query: str
    answerable: bool = True
    pages: list[PageRef] = Field(default_factory=list)
    excerpts: list[str] = Field(default_factory=list)
    structured_hits: dict[str, Any] = Field(default_factory=dict)


class VisionResult(BaseModel):
    summary: str
    relevant_pages: list[PageRef] = Field(default_factory=list)
    extracted: dict[str, Any] = Field(default_factory=dict)


class DiagnosticResult(BaseModel):
    summary: str
    likely_causes: list[str] = Field(default_factory=list)
    steps: list[str] = Field(default_factory=list)
    flowchart_spec: dict[str, Any] = Field(default_factory=dict)


class ArtifactResult(BaseModel):
    artifact_id: str
    artifact_type: Literal["react", "svg", "html", "json", "code", "markdown", "mermaid"]
    title: str
    content: str


class AgentAnswer(BaseModel):
    text: str
    citations: list[PageRef] = Field(default_factory=list)
    surfaced_images: list[PageRef] = Field(default_factory=list)
    artifacts: list[ArtifactResult] = Field(default_factory=list)
    debug: dict[str, Any] = Field(default_factory=dict)
