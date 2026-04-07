from __future__ import annotations

import asyncio
import json
import re
from typing import Any, AsyncGenerator

from app.config import Settings
from app.models import ArtifactResult, ChatMessage, PageRef, RetrievalResult
from app.prompts import ORCHESTRATOR_PROMPT
from tools import PageStore

from .artifact_agent import ArtifactAgent
from .base import ClaudeAgent
from .diagnostic_agent import DiagnosticAgent
from .retrieval_agent import RetrievalAgent
from .vision_agent import VisionAgent

ORCHESTRATOR_MODEL = "claude-opus-4-6"


class OrchestratorAgent(ClaudeAgent):
    def __init__(
        self,
        settings: Settings,
        retrieval_agent: RetrievalAgent,
        vision_agent: VisionAgent,
        diagnostic_agent: DiagnosticAgent,
        artifact_agent: ArtifactAgent,
        page_store: PageStore,
    ) -> None:
        super().__init__(settings, ORCHESTRATOR_MODEL, ORCHESTRATOR_PROMPT)
        self.retrieval_agent = retrieval_agent
        self.vision_agent = vision_agent
        self.diagnostic_agent = diagnostic_agent
        self.artifact_agent = artifact_agent
        self.page_store = page_store

    @staticmethod
    def _dedupe_pages(pages: list[PageRef]) -> list[PageRef]:
        seen: set[tuple[str, int]] = set()
        out: list[PageRef] = []
        for p in pages:
            key = (p.doc, p.page)
            if key in seen:
                continue
            seen.add(key)
            out.append(p)
        return out

    async def stream(
        self, message: str, history: list[ChatMessage], voice_mode: bool = False
    ) -> AsyncGenerator[dict[str, Any], None]:
        """
        Async generator that yields SSE-style event dicts.
        Events: text_delta, image, artifact, done
        """
        if self._is_simple_greeting(message, history):
            greeting = (
                "Hi! I can help with setup, polarity, settings, duty cycle, and troubleshooting for the Vulcan OmniPro 220."
            )
            yield {"type": "text_delta", "content": greeting}
            yield {
                "type": "done",
                "citations": [],
                "debug": {
                    "mode": {"query_type": "greeting", "fast_path": True},
                    "voice_mode": voice_mode,
                    "retrieval_pages": 0,
                    "vision": None,
                    "diagnostic": None,
                },
            }
            return

        mode = self._classify(message)
        follow_up = self._follow_up_profile(message, history, mode)
        mode["follow_up"] = follow_up

        # Always run retrieval first
        retrieval_query = self._query_for_retrieval(message, history, mode, follow_up)
        retrieval = await self.retrieval_agent.run(retrieval_query)
        retrieval = self._suppress_low_relevance_retrieval(message, retrieval)
        pages = self._dedupe_pages(retrieval.pages)[:4]

        clarity = self._assess_query_clarity(message, mode, retrieval, history)
        mode["clarification_first"] = clarity["level"] == "high"
        mode["clarification_level"] = clarity["level"]

        vision_result = None
        diagnostic_result = None

        if not mode["clarification_first"]:
            if mode["needs_vision"] and mode["needs_diagnostic"]:
                results = await asyncio.gather(
                    self.vision_agent.run(message, pages),
                    self.diagnostic_agent.run(message, retrieval),
                )
                vision_result, diagnostic_result = results
            elif mode["needs_vision"]:
                vision_result = await self.vision_agent.run(message, pages)
            elif mode["needs_diagnostic"]:
                diagnostic_result = await self.diagnostic_agent.run(message, retrieval)

        # Build conversation history for the LLM (user/assistant turns only)
        llm_history = [
            {"role": msg.role, "content": msg.content}
            for msg in history
            if msg.role in ("user", "assistant")
        ]

        user_prompt = self._build_user_prompt(message, retrieval, mode, vision_result, diagnostic_result, clarity, voice_mode)

        max_tokens = 700 if mode["clarification_first"] else 1200
        explicit_artifact_request = mode.get("requested_artifact_type") in {"html", "markdown", "mermaid", "svg", "code"}
        if not explicit_artifact_request:
            async for chunk in self.stream_text(user_prompt, history=llm_history, max_tokens=max_tokens):
                yield {"type": "text_delta", "content": chunk}
        elif mode.get("requested_artifact_type") == "html":
            yield {"type": "text_delta", "content": "Rendered result below."}

        if not mode["clarification_first"]:
            surfaced = vision_result.relevant_pages if vision_result else pages[:2]
            surfaced = self._dedupe_pages(surfaced) or pages[:2]
            if not explicit_artifact_request:
                for page in surfaced:
                    yield {
                        "type": "image",
                        "doc": page.doc,
                        "page": page.page,
                        "caption": f"{page.doc} page {page.page}",
                        "url": f"/pages/{page.doc}/{page.page}",
                    }

            artifact_spec = self._build_artifact_spec(message, retrieval, mode, vision_result, diagnostic_result)
            if artifact_spec:
                artifact = await self.artifact_agent.run(artifact_spec)
                if artifact:
                    yield {"type": "artifact", "artifact": artifact}

        # Done — include citations and debug info
        yield {
            "type": "done",
            "citations": [] if explicit_artifact_request else [page.model_dump() for page in pages],
            "debug": {
                "mode": mode,
                "clarification": clarity,
                "voice_mode": voice_mode,
                "retrieval_pages": len(pages),
                "vision": vision_result.model_dump() if vision_result else None,
                "diagnostic": diagnostic_result.model_dump() if diagnostic_result else None,
            },
        }

    def _is_simple_greeting(self, message: str, history: list[ChatMessage]) -> bool:
        normalized = re.sub(r"[^\w\s]", "", message.lower()).strip()
        if not normalized:
            return False
        greetings = {
            "hi",
            "hello",
            "hey",
            "yo",
            "good morning",
            "good afternoon",
            "good evening",
        }
        if normalized not in greetings:
            return False
        prior_user_turns = sum(1 for msg in history if msg.role == "user")
        return prior_user_turns <= 1

    def _follow_up_profile(self, message: str, history: list[ChatMessage], mode: dict[str, Any]) -> dict[str, Any]:
        lowered = message.lower().strip()
        tokens = lowered.split()
        previous_user_messages = [
            msg.content.strip()
            for msg in history
            if msg.role == "user" and msg.content.strip()
        ]
        previous_user_messages = [msg for msg in previous_user_messages if msg != message.strip()]

        if not previous_user_messages:
            return {
                "is_follow_up": False,
                "anchor_message": None,
                "score": 0,
                "reasons": [],
            }

        referential_phrases = (
            "it", "that", "this", "those", "these", "same",
            "again", "above", "previous", "earlier", "last one",
            "that answer", "that setup", "that question", "that one",
            "this answer", "this setup", "this question", "the same thing",
        )
        format_request = mode.get("requested_artifact_type") in {"html", "markdown", "mermaid", "svg", "code"}
        transformation_verbs = (
            "generate", "make", "create", "turn", "convert", "show", "render",
            "put", "give", "format", "explain", "rewrite",
        )
        welding_terms = (
            "tig", "mig", "stick", "flux", "fcaw", "gmaw", "gtaw", "smaw",
            "polarity", "socket", "ground clamp", "torch", "wire", "duty cycle",
            "porosity", "spatter", "welder", "omnipro", "vulcan",
        )

        score = 0
        reasons: list[str] = []

        if any(phrase in lowered for phrase in referential_phrases):
            score += 3
            reasons.append("referential_language")
        if format_request:
            score += 2
            reasons.append("format_request")
        if len(tokens) <= 12:
            score += 1
            reasons.append("short_message")
        if any(lowered.startswith(verb) for verb in transformation_verbs):
            score += 1
            reasons.append("transformation_verb")
        if not any(term in lowered for term in welding_terms):
            score += 2
            reasons.append("missing_domain_terms")

        recent_assistant_text = ""
        for msg in reversed(history):
            if msg.role == "assistant" and msg.content.strip():
                recent_assistant_text = msg.content.lower()
                break
        if recent_assistant_text and any(term in recent_assistant_text for term in welding_terms):
            score += 1
            reasons.append("recent_assistant_domain_context")

        return {
            "is_follow_up": score >= 4,
            "anchor_message": previous_user_messages[-1],
            "score": score,
            "reasons": reasons,
        }

    def _query_for_retrieval(
        self,
        message: str,
        history: list[ChatMessage],
        mode: dict[str, Any],
        follow_up: dict[str, Any],
    ) -> str:
        requested_artifact_type = mode.get("requested_artifact_type")
        if requested_artifact_type not in {"html", "markdown", "mermaid", "svg", "code"} and not follow_up.get("is_follow_up"):
            return message

        lowered = message.lower().strip()
        welding_terms = (
            "tig", "mig", "stick", "flux", "fcaw", "gmaw", "gtaw", "smaw",
            "polarity", "socket", "ground clamp", "torch", "wire", "duty cycle",
            "porosity", "spatter", "welder", "omnipro", "vulcan",
        )
        has_domain_terms = any(term in lowered for term in welding_terms)
        short_follow_up = len(lowered.split()) <= 10

        if has_domain_terms and not short_follow_up and not follow_up.get("is_follow_up"):
            return message

        anchor = follow_up.get("anchor_message")
        if not anchor:
            return message

        return f"{anchor}\n\nRequested output format: {message}"

    def _suppress_low_relevance_retrieval(self, message: str, retrieval: RetrievalResult) -> RetrievalResult:
        lowered = message.lower().strip()
        # Use the max score across all pages rather than pages[0].score — pages may
        # be reranked by topic (e.g. process-selection) so the first page's score
        # is not necessarily the best relevance signal.
        top_score = max((p.score for p in retrieval.pages if p.score is not None), default=None)
        casual_messages = {
            "how are you",
            "hows it going",
            "how's it going",
            "who are you",
            "what can you do",
            "thanks",
            "thank you",
            "ok",
            "okay",
            "cool",
            "nice",
            "great",
            "awesome",
            "lol",
        }
        casual_prefixes = (
            "how are you",
            "who are you",
            "what can you do",
            "tell me about yourself",
        )
        is_casual = lowered in casual_messages or any(lowered.startswith(prefix) for prefix in casual_prefixes)
        weak_retrieval = top_score is None or top_score < 0.12

        if not (is_casual or weak_retrieval):
            return retrieval

        return retrieval.model_copy(update={
            "answerable": False,
            "pages": [],
            "excerpts": [],
            "structured_hits": {},
        })

    # ─── Ambiguity / clarification ───────────────────────────────────────────────

    def _clarity_level_from_reasons(self, reasons: list[str]) -> str:
        if not reasons:
            return "clear"
        high_tags = frozenset({
            "no_manual_hits",
            "duty_missing_process_voltage_or_amps",
            "settings_missing_process_or_thickness",
            "polarity_needs_process",
            "troubleshoot_needs_symptom_detail",
            "short_vague_query",
        })
        if any(r in high_tags for r in reasons):
            return "high"
        if "weak_retrieval_top_score" in reasons:
            return "low"
        return "clear"

    def _assess_query_clarity(
        self,
        message: str,
        mode: dict[str, Any],
        retrieval: RetrievalResult,
        history: list[ChatMessage],
    ) -> dict[str, Any]:
        """
        Returns level: clear | low | high.
        high → clarification-only turn: no vision/diagnostic/artifacts; model asks questions.
        """
        t = message.strip()
        lowered = t.lower()
        words = t.split()
        wn = len(words)

        reasons: list[str] = []

        prior_user_turns = sum(1 for m in history if m.role == "user")

        if not retrieval.pages:
            reasons.append("no_manual_hits")

        top_score = retrieval.pages[0].score if retrieval.pages else None
        if top_score is not None and top_score < 0.09:
            reasons.append("weak_retrieval_top_score")

        vague_cue = any(
            x in lowered
            for x in [
                "help",
                "how do i",
                "how to",
                "what should i",
                "something wrong",
                "not working",
                "won't work",
                "doesn't work",
                "any tips",
                "question",
            ]
        )
        if wn <= 5 and len(t) < 45 and (vague_cue or "?" in t):
            reasons.append("short_vague_query")

        if mode["query_type"] == "duty_cycle":
            # Conceptual questions ("what happens if I exceed…") and explicit
            # artifact requests ("make a calculator") don't need specific values.
            is_conceptual_dc = any(
                phrase in lowered
                for phrase in [
                    "what happens", "what is", "explain", "define", "exceed",
                    "too often", "consequence", "damage", "why", "how long",
                    "what does it mean", "calculator", "if i go over", "if i exceed",
                ]
            )
            has_artifact_planned = bool(mode.get("artifact_type"))
            if not is_conceptual_dc and not has_artifact_planned:
                has_process = any(p in lowered for p in ["mig", "tig", "stick", "flux", "fcaw", "gmaw", "gtaw", "smaw"])
                has_voltage = "120" in lowered or "240" in lowered
                # Fix: "200A" has no \b between digit and letter, so detect directly
                has_amp = (
                    bool(re.search(r"\b\d{2,3}\s*[aA]\b", t))   # "200A", "200 A"
                    or bool(re.search(r"\bat\s+\d{2,3}\b", lowered))  # "at 200"
                    or "amp" in lowered
                )
                if not (has_process and has_voltage and has_amp):
                    reasons.append("duty_missing_process_voltage_or_amps")

        if mode["query_type"] == "settings":
            has_process = any(p in lowered for p in ["mig", "tig", "stick", "flux", "fcaw"])
            has_thickness = bool(re.search(r"\d\s*/\s*\d+", t)) or any(
                x in lowered for x in ["inch", "mm", "gauge", "thick", "sheet", "plate"]
            )
            if not has_process or not has_thickness:
                reasons.append("settings_missing_process_or_thickness")

        if any(
            k in lowered
            for k in ["polarity", "ground clamp", "which socket", "positive socket", "negative socket", "dinse"]
        ):
            has_process = any(p in lowered for p in ["mig", "tig", "stick", "flux", "fcaw"])
            if not has_process:
                reasons.append("polarity_needs_process")

        if mode["query_type"] == "diagnostic":
            has_symptom = any(
                x in lowered
                for x in [
                    "porosity",
                    "spatter",
                    "undercut",
                    "crack",
                    "sticking",
                    "no arc",
                    "arc",
                    "burn",
                    "fusion",
                    "cold",
                    "bead",
                    "wire",
                    "bird",
                    "worm",
                ]
            )
            if not has_symptom and wn < 12:
                reasons.append("troubleshoot_needs_symptom_detail")

        # Later turns: user may have supplied detail in a follow-up message
        if prior_user_turns >= 1:
            if reasons == ["short_vague_query"]:
                reasons = []
            if "duty_missing_process_voltage_or_amps" in reasons and len(t) > 55:
                reasons = [r for r in reasons if r != "duty_missing_process_voltage_or_amps"]
            if "settings_missing_process_or_thickness" in reasons and len(t) > 45:
                reasons = [r for r in reasons if r != "settings_missing_process_or_thickness"]

        level = self._clarity_level_from_reasons(reasons)

        suggestions: list[str] = []
        if "duty_missing_process_voltage_or_amps" in reasons:
            suggestions.append("Which process (MIG, TIG, Stick, flux-cored) and are you on 120V or 240V, at roughly what amperage?")
        if "settings_missing_process_or_thickness" in reasons:
            suggestions.append("Which process and what material thickness are you welding?")
        if "polarity_needs_process" in reasons:
            suggestions.append("Which welding process are you setting up (MIG, TIG, Stick, flux-cored)?")
        if "troubleshoot_needs_symptom_detail" in reasons:
            suggestions.append("What does the weld or arc look like, and which process and wire/gas are you using?")
        if "no_manual_hits" in reasons:
            suggestions.append("What are you trying to do with the OmniPro 220 right now?")

        return {
            "level": level,
            "reasons": reasons,
            "suggested_questions": suggestions,
        }

    # Classification

    def _classify(self, message: str) -> dict[str, Any]:
        lowered = message.lower()
        requested_artifact_type = self._detect_requested_artifact_type(lowered)
        requested_language = self._detect_requested_language(lowered, requested_artifact_type)
        explicit_artifact_intent = self._detect_explicit_artifact_intent(lowered)
        is_duty_cycle = "duty cycle" in lowered
        wants_draw = any(
            t in lowered
            for t in [
                "draw",
                "drawing",
                "sketch",
                "diagram",
                "schematic",
                "chart",
                "visualize",
                "visualise",
                "illustrate",
                "walk me through",
                "walkthrough",
                "walk-through",
                "step-by-step",
                "step by step",
                "show me how",
                "animated",
                "interactive schematic",
                "layout",
                "what does it look like",
                "hard to picture",
                "hard to explain",
                "easier to see",
                "see how",
                "picture of",
            ]
        )
        if re.search(r"\bvisual\w*\b", lowered):
            wants_draw = True
        if re.search(r"\b(image|images|photo|photos|screenshot|screenshots)\b", lowered):
            wants_draw = True
        needs_vision = any(
            t in lowered
            for t in ["diagram", "show", "socket", "where", "inside", "panel",
                      "polarity", "wiring", "picture", "schematic", "connect", "cable",
                      "knob", "dial", "display", "lcd", "controls", "front panel",
                      "torch", "gun", "wire feed", "drive roll", "exploded",
                      "image", "photo", "screenshot"]
        )
        needs_vision = needs_vision or wants_draw
        needs_diagnostic = any(
            t in lowered
            for t in ["problem", "issue", "wrong", "porosity", "spatter", "crack",
                      "burn", "undercut", "diagnose", "troubleshoot", "not working",
                      "bad weld", "bead"]
        )
        is_settings = any(
            t in lowered
            for t in ["setting", "wire feed", "recommend", "what should i set",
                      "mild steel", "stainless", "aluminum", "thickness"]
        )

        is_setup_checklist = any(
            k in lowered
            for k in [
                "first time", "first-time", "initial setup", "unboxing", "getting started",
                "before first weld", "setup checklist", "commission", "first use", "out of the box",
                "brand new welder", "new welder setup",
            ]
        )
        is_process_selector = any(
            k in lowered
            for k in [
                "which process", "what process",
                "which welding process", "what welding process",
                "mig or tig", "tig or mig",
                "choose a process", "selection chart", "pick a process", "what mode should",
                "flux or mig", "stick or", "which welding mode",
                "best process", "what should i use", "which should i use",
            ]
        )
        is_maintenance = any(
            k in lowered
            for k in [
                "maintenance", "clean the welder", "service the", "inspect the machine",
                "fan cleaning", "dust inside", "take care of the welder",
            ]
        )
        is_safety = any(
            k in lowered
            for k in [
                "safety", "ppe", "warning", "electric shock", "safe to weld",
                "ventilation", "fumes",
            ]
        )
        is_voltage_amperage_hint = any(
            t in lowered for t in ["amperage", "240v", "120v", "220v", "input current", "breaker"]
        )

        # Wiring / polarity diagrams → SVG (not React)
        wiring_svg = needs_vision and any(
            t in lowered
            for t in ["polarity", "socket", "wiring", "cable", "connect", "ground clamp", "dinse", "electrode holder"]
        )
        wants_guided_ui = explicit_artifact_intent or any(
            phrase in lowered
            for phrase in [
                "walk me through",
                "walkthrough",
                "walk-through",
                "step by step",
                "step-by-step",
                "interactive",
                "wizard",
                "selector",
                "checklist",
                "calculator",
                "configurator",
                "flowchart",
                "decision tree",
                "guide me through",
            ]
        )
        wants_visual_artifact = wants_draw or any(
            phrase in lowered
            for phrase in ["show me", "diagram", "schematic", "visual", "draw"]
        )
        if wiring_svg:
            wants_visual_artifact = True

        artifact_type: str | None = None
        artifact_style: str | None = None

        if requested_artifact_type:
            artifact_type = requested_artifact_type
            if wiring_svg and requested_artifact_type in {"svg", "mermaid"}:
                artifact_style = "visual_reference"
            elif needs_diagnostic:
                artifact_style = "flowchart"
            elif is_setup_checklist:
                artifact_style = "setup_checklist"
            elif is_process_selector:
                artifact_style = "process_selector"
            elif is_duty_cycle or is_voltage_amperage_hint:
                artifact_style = "duty_cycle"
            elif is_settings:
                artifact_style = "settings"
            elif is_maintenance:
                artifact_style = "maintenance"
            elif is_safety:
                artifact_style = "safety"
            elif needs_vision and wants_visual_artifact:
                artifact_style = "visual_walkthrough" if wants_draw else "visual_reference"
        elif wiring_svg and wants_visual_artifact:
            artifact_type = "svg"
        else:
            # Priority: diagnostic flowchart → guided wizards → duty/settings → maintenance/safety → visual cards
            if needs_diagnostic and wants_guided_ui:
                artifact_style = "flowchart"
                artifact_type = "react"
            elif is_setup_checklist and wants_guided_ui:
                artifact_style = "setup_checklist"
                artifact_type = "react"
            elif is_process_selector and wants_guided_ui:
                artifact_style = "process_selector"
                artifact_type = "react"
            elif (is_duty_cycle or is_voltage_amperage_hint) and wants_guided_ui:
                artifact_style = "duty_cycle"
                artifact_type = "react"
            elif is_settings and wants_guided_ui:
                artifact_style = "settings"
                artifact_type = "react"
            elif is_maintenance and wants_guided_ui:
                artifact_style = "maintenance"
                artifact_type = "react"
            elif is_safety and wants_guided_ui:
                artifact_style = "safety"
                artifact_type = "react"
            elif needs_vision and wants_visual_artifact:
                artifact_style = "visual_walkthrough" if wants_draw else "visual_reference"
                artifact_type = "react"

        query_type = "general"
        if wiring_svg:
            query_type = "visual"
        elif needs_diagnostic:
            query_type = "diagnostic"
        elif is_setup_checklist:
            query_type = "setup"
        elif is_process_selector:
            query_type = "process"
        elif is_duty_cycle or is_voltage_amperage_hint:
            query_type = "duty_cycle"
        elif is_settings:
            query_type = "settings"
        elif is_maintenance:
            query_type = "maintenance"
        elif is_safety:
            query_type = "safety"
        elif needs_vision:
            query_type = "visual"

        return {
            "query_type": query_type,
            "needs_vision": needs_vision,
            "needs_diagnostic": needs_diagnostic,
            "is_settings": is_settings,
            "artifact_type": artifact_type,
            "artifact_style": artifact_style,
            "requested_artifact_type": requested_artifact_type,
            "requested_language": requested_language,
            "explicit_artifact_intent": explicit_artifact_intent,
            "wants_draw": wants_draw,
        }

    def _detect_explicit_artifact_intent(self, lowered: str) -> bool:
        return any(
            phrase in lowered
            for phrase in [
                "make a",
                "build a",
                "create a",
                "give me a checklist",
                "make a checklist",
                "create a flowchart",
                "make a flowchart",
                "decision tree",
                "wizard",
                "calculator",
                "configurator",
                "process selector",
                "interactive",
                "visual walkthrough",
                "visual guide",
            ]
        )

    def _detect_requested_artifact_type(self, lowered: str) -> str | None:
        if "mermaid" in lowered:
            return "mermaid"
        if re.search(r"\bsvg\b", lowered):
            return "svg"
        if "markdown" in lowered or re.search(r"\bmd\b", lowered):
            return "markdown"
        if "single-file html" in lowered or "single file html" in lowered or "one-file html" in lowered or "one file html" in lowered:
            return "html"
        if "html" in lowered:
            return "html"
        if any(
            term in lowered
            for term in [
                "code snippet",
                "python script",
                "typescript function",
                "javascript function",
                "react component",
                "json schema",
                "write code",
                "generate code",
            ]
        ):
            return "code"
        return None

    def _detect_requested_language(self, lowered: str, requested_artifact_type: str | None) -> str | None:
        if requested_artifact_type != "code":
            return None
        if "python" in lowered:
            return "python"
        if "typescript" in lowered:
            return "typescript"
        if "javascript" in lowered:
            return "javascript"
        if "tsx" in lowered or "react component" in lowered:
            return "tsx"
        if "json schema" in lowered or re.search(r"\bjson\b", lowered):
            return "json"
        if re.search(r"\bhtml\b", lowered):
            return "html"
        return "text"

    # ─── LLM context prompt ────────────────────────────────────────────────────

    def _build_user_prompt(
        self,
        message: str,
        retrieval: RetrievalResult,
        mode: dict[str, Any],
        vision_result: Any,
        diagnostic_result: Any,
        clarity: dict[str, Any],
        voice_mode: bool,
    ) -> str:
        if mode.get("clarification_first"):
            sug_lines = clarity.get("suggested_questions") or []
            sug = "\n".join(f"  - {s}" for s in sug_lines[:4])
            if not sug:
                sug = "  - Which process (MIG / TIG / Stick / flux-cored)?\n  - 120V or 240V input?\n  - Rough amperage or material thickness?"
            flags = ", ".join(clarity.get("reasons", [])) or "underspecified"
            return (
                f"User question: {message}\n\n"
                f"[[ CLARIFICATION REQUIRED ]]\n"
                f"The question is too vague to quote this machine's manual safely. "
                f"Your entire reply must be clarifying questions only: at most one short opening line, "
                f"then 1–3 bullet questions. Do NOT give duty cycles, polarity, wire speeds, settings, "
                f"or troubleshooting causes in this turn.\n\n"
                f"Automated flags: {flags}.\n"
                f"You may use these as inspiration (rephrase naturally):\n{sug}\n"
            )

        blocks: list[str] = []

        if clarity.get("level") == "low":
            blocks.append(
                "[[ CONTEXT MAY BE THIN ]]\n"
                "Manual retrieval is borderline for this query. Answer from the excerpts below if they "
                "are sufficient; if not, ask one focused follow-up question instead of inventing specs."
            )

        if mode.get("wants_draw") or mode.get("artifact_type") == "svg":
            blocks.append(
                "[[ VISUAL FIRST ]]\n"
                "An interactive diagram or schematic is generated after your text. For layouts, cable paths, "
                "controls, polarity, or sequences: keep prose short (key facts + page refs). Do not try to "
                "replace the diagram with a wall of text. You may say the interactive visual below shows "
                "the layout or steps — never say 'artifact' or 'tool'."
            )

        if mode.get("requested_artifact_type") in {"html", "markdown", "mermaid", "svg", "code"}:
            blocks.append(
                "[[ EXPLICIT ARTIFACT REQUEST ]]\n"
                "The user explicitly requested a specific output format. Keep your text to at most one short sentence, "
                "or no sentence if the format fully answers the request. Do not mention files, saving, write permission, "
                "downloads, project folders, or approval prompts."
            )

        if voice_mode:
            blocks.append(
                "[[ VOICE MODE ]]\n"
                "This reply may be spoken aloud. Keep the opening answer short, direct, and easy to say. "
                "Lead with the answer in 1-3 short sentences before any extra detail."
            )

        # Manual text excerpts with page citations
        if retrieval.excerpts:
            pairs = list(zip(retrieval.pages, retrieval.excerpts))
            lines = "\n".join(
                f"  [{p.doc} p.{p.page}]: {exc.strip()}"
                for p, exc in pairs[:4]
            )
            blocks.append(f"MANUAL CONTEXT:\n{lines}")

        # Structured duty-cycle table
        if mode["query_type"] == "duty_cycle":
            entries = retrieval.structured_hits.get("duty_cycles", [])
            if entries:
                blocks.append(f"DUTY CYCLE TABLE:\n{json.dumps(entries[:10], indent=2)}")

        # Structured wire settings
        if mode["query_type"] == "settings":
            wire = retrieval.structured_hits.get("wire_settings", {})
            if wire:
                blocks.append(f"WIRE SETTINGS:\n{json.dumps(wire, indent=2)}")

        if mode["query_type"] in ("diagnostic", "maintenance", "safety", "setup"):
            trouble = retrieval.structured_hits.get("troubleshooting", [])
            if trouble:
                blocks.append(f"TROUBLESHOOTING / MAINTENANCE SNIPPETS:\n{json.dumps(trouble[:8], indent=2)}")

        # Vision agent output
        if vision_result and vision_result.summary:
            extracted = json.dumps(vision_result.extracted, indent=2) if vision_result.extracted else ""
            blocks.append(f"VISUAL ANALYSIS (from manual diagrams):\n{vision_result.summary}\n{extracted}")

        if re.search(
            r"\b(image|images|photo|photos|picture|pictures|screenshot|show me)\b",
            message.lower(),
        ) and (vision_result or retrieval.excerpts):
            blocks.append(
                "[[ PAGE PREVIEWS IN UI ]]\n"
                "This chat app shows clickable manual page thumbnails directly beneath your reply. "
                "Describe what appears on those pages (socket labels, +/- markings, panel layout) using "
                "VISUAL ANALYSIS and MANUAL CONTEXT. Do not say that no image is available, that the "
                "retrieved text lacks a photo, or that you cannot display one—the user can open the previews."
            )

        # Diagnostic agent output
        if diagnostic_result and diagnostic_result.summary:
            causes = "\n".join(f"  - {c}" for c in diagnostic_result.likely_causes)
            steps = "\n".join(f"  {i + 1}. {s}" for i, s in enumerate(diagnostic_result.steps))
            blocks.append(
                f"DIAGNOSTIC ANALYSIS:\nMost likely: {diagnostic_result.summary}\n"
                f"Causes:\n{causes}\nSteps:\n{steps}"
            )

        if not blocks:
            blocks.append("No manual context was retrieved for this question.")

        context = "\n\n".join(blocks)
        return f"User question: {message}\n\n{context}\n\nAnswer based on the context above."

    # ─── Artifact spec builder ─────────────────────────────────────────────────

    def _build_artifact_spec(
        self,
        message: str,
        retrieval: RetrievalResult,
        mode: dict[str, Any],
        vision_result: Any,
        diagnostic_result: Any,
    ) -> dict[str, Any] | None:
        if mode.get("clarification_first"):
            return None

        artifact_type = mode.get("artifact_type")
        if not artifact_type:
            return None

        title = "Welder Reference"
        if artifact_type == "svg":
            title = "Polarity / Wiring Diagram"
        elif mode.get("artifact_style") == "flowchart":
            title = "Troubleshooting Flowchart"
        elif mode["query_type"] == "setup":
            title = "First-Use Setup Checklist"
        elif mode["query_type"] == "process":
            title = "Process Selection Wizard"
        elif mode["query_type"] == "maintenance":
            title = "Maintenance Checklist"
        elif mode["query_type"] == "safety":
            title = "Safety Quick Reference"
        elif mode["query_type"] == "diagnostic":
            title = "Troubleshooting Guide"
        elif mode["query_type"] == "duty_cycle":
            title = "Duty Cycle Calculator"
        elif mode["query_type"] == "settings":
            title = "Settings Configurator"
        elif mode.get("artifact_style") == "visual_reference":
            title = "Control & Diagram Reference"
        elif mode.get("artifact_style") == "visual_walkthrough":
            title = "Interactive Visual Walkthrough"

        payload: dict[str, Any] = {
            "query": retrieval.query,
            "pages": [p.model_dump() for p in retrieval.pages[:2]],
            "excerpts": retrieval.excerpts[:6],
        }
        if mode["query_type"] == "duty_cycle":
            payload["duty_cycles"] = retrieval.structured_hits.get("duty_cycles", [])
        elif mode["query_type"] == "settings":
            payload["wire_settings"] = retrieval.structured_hits.get("wire_settings", {})
        if mode["query_type"] in ("diagnostic", "maintenance", "safety", "setup", "process"):
            payload["troubleshooting_snippets"] = retrieval.structured_hits.get("troubleshooting", [])[:12]

        return {
            "title": title,
            "artifact_type": artifact_type,
            "artifact_style": mode.get("artifact_style"),
            "requested_language": mode.get("requested_language"),
            "wants_draw": mode.get("wants_draw", False),
            "question": message,
            "retrieval": payload,
            "vision": vision_result.model_dump() if vision_result else None,
            "diagnostic": diagnostic_result.model_dump() if diagnostic_result else None,
        }
