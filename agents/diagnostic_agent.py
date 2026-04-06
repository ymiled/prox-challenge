from __future__ import annotations

import json

from app.config import Settings
from app.models import DiagnosticResult, RetrievalResult
from app.prompts import DIAGNOSTIC_PROMPT
from app.utils import safe_excerpt

from .base import ClaudeAgent

DIAGNOSTIC_MODEL = "claude-opus-4-6"


class DiagnosticAgent(ClaudeAgent):
    def __init__(self, settings: Settings) -> None:
        super().__init__(settings, DIAGNOSTIC_MODEL, DIAGNOSTIC_PROMPT)

    async def run(self, symptoms: str, retrieval: RetrievalResult) -> DiagnosticResult:
        if not self.enabled:
            steps = [
                "Check the troubleshooting pages surfaced in the citations.",
                "Verify the machine setup, polarity, and consumables before changing settings.",
                "Inspect shielding gas, work clamp connection, and material cleanliness.",
            ]
            return DiagnosticResult(
                summary="Diagnostic agent fallback mode is active because Anthropic is unavailable.",
                likely_causes=["Insufficient grounding or incorrect setup", "Consumable or gas issue"],
                steps=steps,
                flowchart_spec={"nodes": [], "edges": []},
            )

        prompt = {
            "symptoms": symptoms,
            "retrieval": retrieval.model_dump(),
            "instructions": "Return valid JSON with summary, likely_causes, steps, and flowchart_spec.",
        }
        parsed = await self.complete_json(json.dumps(prompt, indent=2))
        if not parsed:
            return DiagnosticResult(
                summary="Diagnostic agent did not return a response.",
                likely_causes=[],
                steps=[],
                flowchart_spec={},
            )
        return DiagnosticResult(
            summary=parsed.get("summary", safe_excerpt(str(parsed))),
            likely_causes=parsed.get("likely_causes", []),
            steps=parsed.get("steps", []),
            flowchart_spec=parsed.get("flowchart_spec", {}),
        )
