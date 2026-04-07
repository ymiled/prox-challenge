from __future__ import annotations

import json
import re
import uuid

from app.config import Settings
from app.models import ArtifactResult
from app.prompts import ARTIFACT_PROMPT

from .base import ClaudeAgent

ARTIFACT_MODEL = "claude-haiku-4-5-20251001"


class ArtifactAgent(ClaudeAgent):
    def __init__(self, settings: Settings) -> None:
        super().__init__(settings, ARTIFACT_MODEL, ARTIFACT_PROMPT)

    async def run(self, spec: dict) -> ArtifactResult | None:
        artifact_id = spec.get("artifact_id") or f"artifact-{uuid.uuid4().hex[:8]}"
        title = spec.get("title", "Artifact")
        artifact_type = spec.get("artifact_type", "react")

        if artifact_type == "svg":
            return await self._generate_svg(artifact_id, title, spec)

        if artifact_type == "react":
            return await self._generate_react(artifact_id, title, spec)

        if artifact_type == "html":
            return await self._generate_html(artifact_id, title, spec)

        if artifact_type == "markdown":
            return await self._generate_markdown(artifact_id, title, spec)

        if artifact_type == "mermaid":
            return await self._generate_mermaid(artifact_id, title, spec)

        if artifact_type == "code":
            return await self._generate_code(artifact_id, title, spec)

        # Fallback: compact JSON summary
        return ArtifactResult(
            artifact_id=artifact_id,
            artifact_type="json",
            title=title,
            content=json.dumps({"title": title, "question": spec.get("question"), "data": spec.get("retrieval")}, indent=2),
        )

    # React component generation

    async def _generate_react(self, artifact_id: str, title: str, spec: dict) -> ArtifactResult | None:
        if not self.enabled:
            return self._react_fallback(artifact_id, title, spec)

        prompt = self._build_react_prompt(spec)
        style = spec.get("artifact_style") or ""
        large_styles = {"flowchart", "process_selector", "setup_checklist", "visual_reference", "visual_walkthrough"}
        max_tokens = 2800 if style in large_styles else 2200
        raw = await self.complete_text(prompt, max_tokens=max_tokens)
        if not raw:
            return self._react_fallback(artifact_id, title, spec)

        code = self._extract_jsx(raw)
        return ArtifactResult(
            artifact_id=artifact_id,
            artifact_type="react",
            title=title,
            content=code,
        )

    async def _generate_html(self, artifact_id: str, title: str, spec: dict) -> ArtifactResult:
        if not self.enabled:
            return ArtifactResult(
                artifact_id=artifact_id,
                artifact_type="html",
                title=title,
                content=self._html_fallback(title, spec),
            )

        raw = await self.complete_text(self._build_html_prompt(spec), max_tokens=2600)
        content = self._extract_html(raw) if raw else ""
        if not content:
            content = self._html_fallback(title, spec)
        return ArtifactResult(
            artifact_id=artifact_id,
            artifact_type="html",
            title=title,
            content=content,
        )

    async def _generate_markdown(self, artifact_id: str, title: str, spec: dict) -> ArtifactResult:
        if not self.enabled:
            content = self._markdown_fallback(title, spec)
        else:
            raw = await self.complete_text(self._build_markdown_prompt(spec), max_tokens=1800)
            content = self._strip_fences(raw) if raw else self._markdown_fallback(title, spec)
        return ArtifactResult(
            artifact_id=artifact_id,
            artifact_type="markdown",
            title=title,
            content=content,
        )

    async def _generate_mermaid(self, artifact_id: str, title: str, spec: dict) -> ArtifactResult:
        if not self.enabled:
            content = self._mermaid_fallback(title, spec)
        else:
            raw = await self.complete_text(self._build_mermaid_prompt(spec), max_tokens=1600)
            content = self._extract_mermaid(raw) if raw else self._mermaid_fallback(title, spec)
        return ArtifactResult(
            artifact_id=artifact_id,
            artifact_type="mermaid",
            title=title,
            content=content,
        )

    async def _generate_code(self, artifact_id: str, title: str, spec: dict) -> ArtifactResult:
        if not self.enabled:
            content = self._code_fallback(spec)
        else:
            raw = await self.complete_text(self._build_code_prompt(spec), max_tokens=2200)
            content = self._strip_fences(raw) if raw else self._code_fallback(spec)
        return ArtifactResult(
            artifact_id=artifact_id,
            artifact_type="code",
            title=title,
            content=content,
        )

    def _normalize_flowchart(self, diagnostic: dict) -> dict[str, list]:
        """Ensure we always have navigable nodes/edges for the interactive flow UI."""
        fc = diagnostic.get("flowchart_spec") or {}
        nodes = [dict(n) for n in (fc.get("nodes") or []) if isinstance(n, dict) and n.get("id")]
        edges = [dict(e) for e in (fc.get("edges") or []) if isinstance(e, dict) and e.get("from") and e.get("to")]

        ids = {n["id"] for n in nodes}
        edges = [e for e in edges if e["from"] in ids and e["to"] in ids]

        if nodes and edges:
            return {"nodes": nodes, "edges": edges}

        steps = [str(s) for s in (diagnostic.get("steps") or []) if s]
        summary = str(diagnostic.get("summary") or "Work through these checks in order.")

        if not steps:
            return {
                "nodes": [
                    {"id": "start", "label": summary, "type": "start"},
                    {"id": "end", "label": "See the owner manual troubleshooting section for more detail.", "type": "outcome"},
                ],
                "edges": [{"from": "start", "to": "end", "label": "OK"}],
            }

        out_nodes: list[dict] = [{"id": "start", "label": summary, "type": "start"}]
        out_edges: list[dict] = []
        prev = "start"
        for i, step in enumerate(steps[:10]):
            nid = f"step-{i}"
            out_nodes.append({"id": nid, "label": step, "type": "action"})
            out_edges.append({"from": prev, "to": nid, "label": "Next"})
            prev = nid
        out_nodes.append(
            {
                "id": "end",
                "label": "If the issue persists, re-check polarity, gas, and grounding before changing advanced settings.",
                "type": "outcome",
            }
        )
        out_edges.append({"from": prev, "to": "end", "label": "Done"})
        return {"nodes": out_nodes, "edges": out_edges}

    def _build_react_prompt(self, spec: dict) -> str:
        title = spec.get("title", "Welder Tool")
        question = spec.get("question", "")
        retrieval = spec.get("retrieval", {})
        diagnostic = spec.get("diagnostic")
        vision = spec.get("vision")

        duty_cycles = retrieval.get("duty_cycles", [])
        wire_settings = retrieval.get("wire_settings", {})
        excerpts = retrieval.get("excerpts", [])
        trouble = retrieval.get("troubleshooting_snippets", [])
        style = spec.get("artifact_style") or ""

        task = ""

        if style == "flowchart":
            diagnostic = diagnostic or {}
            fc = self._normalize_flowchart(diagnostic)
            data = json.dumps(fc, indent=2)
            summary = diagnostic.get("summary", "")
            causes = diagnostic.get("likely_causes", [])
            task = f"""Generate an INTERACTIVE TROUBLESHOOTING FLOWCHART (click-through decision tree).

USER ISSUE: {question}
SUMMARY: {summary}
LIKELY CAUSES (render as small chips or a list at the top): {json.dumps(causes)}

FLOWGRAPH — hardcode nodes and edges exactly as JSON inside the component:
{data}

Behavior:
- Find start node: prefer id "start", else first node in the array.
- Display the current node label in a large readable card.
- List outgoing edges from the current node as primary BUTTONS (edge label = button text).
- On button click, navigate to edge.to node id.
- For nodes with type "outcome", show success styling and a "Start over" button to reset to start.
- Show a subtle breadcrumb of visited node labels (optional).
- Use slate background, orange-500 accents, rounded-xl, mobile-friendly tap targets."""

        elif style == "visual_walkthrough":
            vision = spec.get("vision") or {}
            extracted = vision.get("extracted") or {}
            summ = vision.get("summary", "") or ""
            ex_json = json.dumps(extracted, indent=2)[:2200]
            excerpts_text = "\n".join(retrieval.get("excerpts", [])[:4])
            pages_json = json.dumps(retrieval.get("pages", []), indent=2)[:900]
            task = f"""Generate an INTERACTIVE VISUAL WALKTHROUGH for the Vulcan OmniPro 220: stepped UI + on-screen schematic built from code (Tailwind divs), not a static image.

USER QUESTION: {question}

VISION SUMMARY: {summ}
EXTRACTED FACTS — use ONLY these for labels and positions on the schematic (no invented socket names, voltages, or knob functions):
{ex_json}

MANUAL EXCERPTS (extra labels if needed):
{excerpts_text[:2200]}

PAGE REFERENCES: {pages_json}

Build:
- React.useState for current step index (0 .. N-1)
- Large center "stage": each step shows a simplified schematic (rounded rectangles, flex/grid, border-2, bg-slate-50/white, text-sm labels) representing layout, cable path, control area, or sequence — driven by FACTS/EXCERPTS
- Use → or flex + border-l-4 for flow between elements where it helps
- One short caption line under the stage per step
- Prev / Next buttons; optional row of step dots (clickable)
- Orange-500 accents, slate-800 text, mobile-friendly tap targets
- If context is thin for a step, say "Check your manual page …" using PAGE REFERENCES — do not guess specs"""

        elif style == "setup_checklist":
            ctx = json.dumps({"excerpts": excerpts, "snippets": trouble[:8]}, indent=2)[:4000]
            task = f"""Generate a FIRST-USE SETUP CHECKLIST for the Vulcan OmniPro 220.

CONTEXT (manual excerpts + snippets — use verbatim checkbox text when possible):
{ctx}

Build:
- Sections: Safety → Input power → Work clamp / stinger → Gas & wire (if applicable) → Controls → Test weld
- Each item is a checkbox with React.useState
- Optional "Expand" hints using excerpts only"""

        elif style == "process_selector":
            ctx = json.dumps({"excerpts": excerpts}, indent=2)[:2800]
            task = f"""Generate a PROCESS SELECTION WIZARD for the Vulcan OmniPro 220.

USER: {question}

CONTEXT:
{ctx}

Build:
- 3–4 wizard steps as full-width choice buttons (material family, thickness bucket, job type, skill comfort)
- Final card recommends when MIG, flux-cored, TIG, or stick is appropriate for OmniPro 220 — only claim what context supports
- "Back" button on steps after the first"""

        elif style == "maintenance":
            ctx = json.dumps({"excerpts": excerpts, "snippets": trouble[:8]}, indent=2)[:4000]
            task = f"""Generate a MAINTENANCE CHECKLIST for the Vulcan OmniPro 220.

CONTEXT:
{ctx}

Build:
- 8–14 checkboxes grouped (cooling, intake, cables, gun/torch, consumables)
- Short notes field is optional (single React.useState string)"""

        elif style == "safety":
            ctx = json.dumps({"excerpts": excerpts, "snippets": trouble[:8]}, indent=2)[:4000]
            task = f"""Generate a SAFETY QUICK REFERENCE for the Vulcan OmniPro 220.

CONTEXT:
{ctx}

Build:
- 4–6 collapsible sections (React.useState object or separate booleans) — Electric shock, Fumes/ventilation, Fire, PPE, Workspace
- Bullets only; no scary filler"""

        elif style == "visual_reference" and vision:
            extracted = vision.get("extracted") or {}
            summ = vision.get("summary", "")
            ex_json = json.dumps(extracted, indent=2)[:2400]
            task = f"""Generate a CONTROL / DIAGRAM REFERENCE from vision analysis (no invented specs).

SUMMARY: {summ}
EXTRACTED (use keys as section titles where helpful):
{ex_json}

Build:
- Tabbed or accordion UI for clusters of facts
- If extracted is empty, show a simple "See manual pages" card listing page refs from context only"""

        elif style == "visual_reference":
            excerpts_text = "\n".join(excerpts[:5])
            pages_json = json.dumps(retrieval.get("pages", []), indent=2)[:800]
            task = f"""Generate a CONTROL / DIAGRAM REFERENCE from manual text (no vision payload).

QUESTION: {question}

EXCERPTS:
{excerpts_text[:3000]}

PAGE REFERENCES: {pages_json}

Build:
- Accordion or tabs grouping excerpt themes
- Add one simple schematic strip (Tailwind boxes + labels) only where excerpts name concrete parts; otherwise page pointers only"""

        elif style == "duty_cycle" and duty_cycles:
            data_json = json.dumps(duty_cycles[:24], indent=2)
            task = f"""Generate a Duty Cycle Calculator for the Vulcan OmniPro 220.

DATA (hardcode this array in the component):
{data_json}

Build:
- Three dropdowns: Process, Input Voltage, Amperage (each filters from the data)
- A large prominent display of the matching duty_cycle_percent
- A plain-English note: "At X%, you can weld for Y seconds then rest Z seconds per minute."
- Orange accent color (Tailwind orange-500 / #f97316)
- Clean card layout, centered"""

        elif style == "settings" and wire_settings:
            data_json = json.dumps(wire_settings, indent=2)[:1400]
            task = f"""Generate a Settings Configurator for the Vulcan OmniPro 220.

WIRE SETTINGS DATA (hardcode this in the component):
{data_json}

Build:
- Dropdowns for: Welding Process, Material Type, Material Thickness
- Output section showing: recommended Voltage range and Wire Feed Speed
- A safety note: "Always make a test weld on scrap first."
- Orange accent color for the output section"""

        elif diagnostic:
            # Legacy fallback: checklist-style if flowchart style was lost
            summary = diagnostic.get("summary", "Check your welding setup.")
            causes = diagnostic.get("likely_causes", [])
            steps = diagnostic.get("steps", [])
            data = json.dumps({"summary": summary, "causes": causes, "steps": steps})
            task = f"""Generate an interactive Troubleshooting Guide.

DATA (hardcode this in the component):
{data}

Build:
- Problem summary at the top in a highlighted box
- "Steps to Try" section: numbered steps with checkboxes (use React.useState for checked state)
- "Likely Causes" section: simple bulleted list
- Slate/gray color scheme, calm and clinical"""

        elif duty_cycles:
            data_json = json.dumps(duty_cycles[:24], indent=2)
            task = f"""Generate a Duty Cycle Calculator for the Vulcan OmniPro 220.

DATA (hardcode this array in the component):
{data_json}

Build:
- Three dropdowns: Process, Input Voltage, Amperage (each filters from the data)
- A large prominent display of the matching duty_cycle_percent
- A plain-English note: "At X%, you can weld for Y seconds then rest Z seconds per minute."
- Orange accent color (Tailwind orange-500 / #f97316)
- Clean card layout, centered"""

        elif wire_settings:
            data_json = json.dumps(wire_settings, indent=2)[:1400]
            task = f"""Generate a Settings Configurator for the Vulcan OmniPro 220.

WIRE SETTINGS DATA (hardcode this in the component):
{data_json}

Build:
- Dropdowns for: Welding Process, Material Type, Material Thickness
- Output section showing: recommended Voltage range and Wire Feed Speed
- A safety note: "Always make a test weld on scrap first."
- Orange accent color for the output section"""

        else:
            context = "\n".join(excerpts[:3]) or question
            task = f"""Generate a helpful reference card for: {title}

Context: {context}

Build a clean informational component with the key facts presented clearly.
Use orange accents (Tailwind orange-500)."""

        if task:
            task += self._visual_draw_bonus(spec)

        return f"""{task}

CRITICAL — rules that must be followed exactly:
- NO import statements
- export default function App() {{ ... }}
- React.useState() for state (never const [x] = useState())
- Tailwind CSS classes only (no style objects)
- All data hardcoded inside the component
- Return ONLY the JSX code — no markdown fences, no explanation"""

    def _visual_draw_bonus(self, spec: dict) -> str:
        """Extra instructions when the user asked for a diagram / walkthrough."""
        if not spec.get("wants_draw"):
            return ""
        style = spec.get("artifact_style") or ""
        if style in ("visual_walkthrough",):
            return ""
        if style == "flowchart":
            return (
                "\n\nVISUAL DEPTH: Beside the current node, add a compact illustrated strip (colored divs or simple shapes) "
                "hinting what to inspect on the machine for this step — only from provided context."
            )
        if style == "duty_cycle":
            return (
                "\n\nVISUAL DEPTH: Add a horizontal segmented bar or timeline (divs + Tailwind) for weld-on vs cool-off "
                "seconds implied by the selected duty cycle; values must update when dropdowns change."
            )
        if style == "settings":
            return (
                "\n\nVISUAL DEPTH: Add a minimal mock front-panel or wire-feed strip (boxed regions + labels from context only)."
            )
        if style == "setup_checklist":
            return (
                "\n\nVISUAL DEPTH: Add a vertical phase rail or step column so the checklist reads as a walkthrough, not a flat list."
            )
        if style == "process_selector":
            return (
                "\n\nVISUAL DEPTH: Add a simple comparison strip (columns or cards) for process tradeoffs grounded in context."
            )
        if style == "maintenance":
            return "\n\nVISUAL DEPTH: Add a simple machine-outline diagram with numbered zones matching checklist groups."
        if style == "safety":
            return "\n\nVISUAL DEPTH: Add small icon-like divs (not images) per section header for quick scanning."
        if style == "visual_reference":
            return (
                "\n\nVISUAL DEPTH: Add at least one div-built schematic panel summarizing layout from EXTRACTED/EXCERPTS."
            )
        return ""

    def _extract_jsx(self, raw: str) -> str:
        # Strip markdown fences
        fenced = re.search(r"```(?:jsx?|tsx?|javascript|typescript)?\s*([\s\S]*?)\s*```", raw)
        if fenced:
            return fenced.group(1).strip()
        # Already clean code
        return raw.strip()

    def _extract_html(self, raw: str) -> str:
        match = re.search(r"(<!DOCTYPE html>[\s\S]*</html>)", raw, re.I)
        if match:
            return match.group(1).strip()
        fenced = re.search(r"```(?:html)?\s*([\s\S]*?)\s*```", raw, re.I)
        if fenced:
            candidate = fenced.group(1).strip()
            if re.search(r"<html[\s>]|<!DOCTYPE html>", candidate, re.I):
                return candidate
        return ""

    def _extract_mermaid(self, raw: str) -> str:
        fenced = re.search(r"```(?:mermaid)?\s*([\s\S]*?)\s*```", raw, re.I)
        if fenced:
            return fenced.group(1).strip()
        return raw.strip()

    def _strip_fences(self, raw: str) -> str:
        fenced = re.search(r"```(?:[\w+-]+)?\s*([\s\S]*?)\s*```", raw)
        if fenced:
            return fenced.group(1).strip()
        return raw.strip()

    def _react_fallback(self, artifact_id: str, title: str, spec: dict) -> ArtifactResult:
        question = spec.get("question", "")
        return ArtifactResult(
            artifact_id=artifact_id,
            artifact_type="json",
            title=title,
            content=json.dumps({"title": title, "question": question}, indent=2),
        )

    def _html_fallback(self, title: str, spec: dict) -> str:
        question = spec.get("question", "")
        excerpts = (spec.get("retrieval") or {}).get("excerpts", [])[:6]
        items = "".join(f"<li>{self._escape_html(item)}</li>" for item in excerpts) or "<li>No manual context available.</li>"
        return f"""<!DOCTYPE html>
<html>
<head>
  <meta charset="UTF-8" />
  <title>{self._escape_html(title)}</title>
  <style>
    body {{ font-family: system-ui, -apple-system, sans-serif; margin: 0; padding: 24px; background: #f8fafc; color: #0f172a; }}
    .card {{ max-width: 880px; margin: 0 auto; background: white; border-radius: 16px; padding: 24px; box-shadow: 0 10px 30px rgba(15, 23, 42, 0.08); }}
    h1 {{ margin-top: 0; }}
    ul {{ line-height: 1.6; }}
  </style>
</head>
<body>
  <div class="card">
    <h1>{self._escape_html(title)}</h1>
    <p>{self._escape_html(question)}</p>
    <ul>{items}</ul>
  </div>
</body>
</html>"""

    def _markdown_fallback(self, title: str, spec: dict) -> str:
        question = spec.get("question", "")
        excerpts = (spec.get("retrieval") or {}).get("excerpts", [])[:5]
        lines = [f"# {title}", "", f"Question: {question}", "", "## Manual Notes", ""]
        if excerpts:
            lines.extend(f"- {item}" for item in excerpts)
        else:
            lines.append("- No manual context available.")
        return "\n".join(lines)

    def _mermaid_fallback(self, title: str, spec: dict) -> str:
        question = spec.get("question", "")
        return "\n".join(
            [
                "flowchart TD",
                f'    A["{self._mermaid_text(title)}"] --> B["{self._mermaid_text(question or "Review manual context")}"]',
                '    B --> C["Check cited manual pages"]',
            ]
        )

    def _code_fallback(self, spec: dict) -> str:
        language = spec.get("requested_language") or "text"
        question = spec.get("question", "")
        if language == "python":
            return "\n".join(
                [
                    '"""Generated fallback stub."""',
                    "",
                    "def main() -> None:",
                    f'    print({question!r})',
                    "",
                    'if __name__ == "__main__":',
                    "    main()",
                ]
            )
        return f"// Generated fallback stub\n// Request: {question}\n"

    def _build_html_prompt(self, spec: dict) -> str:
        title = spec.get("title", "HTML Artifact")
        question = spec.get("question", "")
        retrieval = json.dumps(spec.get("retrieval", {}), indent=2)[:4000]
        style = spec.get("artifact_style") or ""
        extra = ""
        if style == "setup_checklist":
            extra = (
                "\nBuild a clear checklist UI with sections and checkboxes. "
                "Use vanilla HTML, CSS, and JavaScript only. Keep it mobile-friendly."
            )
        elif style == "process_selector":
            extra = (
                "\nBuild a lightweight chooser UI with buttons or selects that summarize the supported process options. "
                "Only claim what the manual context supports."
            )
        return (
            f"Create a single self-contained HTML artifact.\n"
            f"Title: {title}\n"
            f"User request: {question}\n"
            f"Manual context:\n{retrieval}\n"
            f"{extra}\n"
            "Rules:\n"
            "- Return ONLY a complete single-file HTML document starting with <!DOCTYPE html> and ending with </html>\n"
            "- Do NOT include markdown fences, explanations, introductions, or notes before or after the HTML\n"
            "- Do NOT mention writing, saving, permissions, approval, files, file names, download, or double-clicking\n"
            "- Do NOT say what the artifact includes; emit the artifact directly\n"
            "- Do NOT use React\n"
            "- Use inline CSS and optional inline JavaScript only\n"
            "- Ground labels and guidance in the provided context\n"
            "- If context is incomplete, make the UI conservative and say 'Check the cited manual pages' inside the HTML instead of narrating limitations outside it\n"
        )

    def _build_markdown_prompt(self, spec: dict) -> str:
        title = spec.get("title", "Markdown Artifact")
        question = spec.get("question", "")
        retrieval = json.dumps(spec.get("retrieval", {}), indent=2)[:4000]
        return (
            f"Create a markdown document titled '{title}'.\n"
            f"User request: {question}\n"
            f"Manual context:\n{retrieval}\n"
            "Rules:\n"
            "- Return markdown only\n"
            "- Use headings and flat bullet lists when helpful\n"
            "- Stay grounded in the provided context\n"
        )

    def _build_mermaid_prompt(self, spec: dict) -> str:
        title = spec.get("title", "Mermaid Diagram")
        question = spec.get("question", "")
        retrieval = json.dumps(spec.get("retrieval", {}), indent=2)[:3500]
        diagnostic = json.dumps(spec.get("diagnostic", {}), indent=2)[:1800]
        return (
            f"Create a Mermaid diagram for '{title}'.\n"
            f"User request: {question}\n"
            f"Manual context:\n{retrieval}\n"
            f"Diagnostic context:\n{diagnostic}\n"
            "Rules:\n"
            "- Return Mermaid syntax only\n"
            "- Prefer flowchart TD for troubleshooting or step-by-step guidance\n"
            "- Use concise node labels\n"
            "- Stay grounded in the supplied context\n"
        )

    def _build_code_prompt(self, spec: dict) -> str:
        language = spec.get("requested_language") or "text"
        question = spec.get("question", "")
        retrieval = json.dumps(spec.get("retrieval", {}), indent=2)[:3200]
        return (
            f"Write a self-contained {language} code artifact.\n"
            f"User request: {question}\n"
            f"Manual context:\n{retrieval}\n"
            "Rules:\n"
            "- Return code only\n"
            "- No markdown fences\n"
            "- Keep it reasonably short and usable\n"
            "- If manual facts are referenced, keep them grounded in the supplied context\n"
        )

    def _escape_html(self, text: str) -> str:
        return (
            str(text)
            .replace("&", "&amp;")
            .replace("<", "&lt;")
            .replace(">", "&gt;")
            .replace('"', "&quot;")
        )

    def _mermaid_text(self, text: str) -> str:
        return str(text).replace('"', "'")

    # SVG diagram generation

    async def _generate_svg(self, artifact_id: str, title: str, spec: dict) -> ArtifactResult | None:
        if not self.enabled:
            return None

        prompt = self._build_svg_prompt(spec)
        raw = await self.complete_text(prompt, max_tokens=1800)
        if not raw:
            return None

        return ArtifactResult(
            artifact_id=artifact_id,
            artifact_type="svg",
            title=title,
            content=self._extract_svg(raw),
        )

    def _build_svg_prompt(self, spec: dict) -> str:
        question = spec.get("question", "")
        vision = spec.get("vision") or {}
        extracted = vision.get("extracted", {}) if isinstance(vision, dict) else {}

        # Pull any known socket assignments from vision
        ground_socket = extracted.get("ground_clamp_socket") or extracted.get("ground_socket", "")
        torch_socket = extracted.get("torch_socket") or extracted.get("torch_cable_socket", "")
        polarity = extracted.get("polarity") or extracted.get("polarity_for_TIG", "")

        context = ""
        if ground_socket or torch_socket or polarity:
            context = f"""
Known wiring facts from the manual:
- Polarity: {polarity or 'see manual'}
- Ground clamp socket: {ground_socket or 'see manual'}
- Torch/electrode socket: {torch_socket or 'see manual'}
"""

        depth = ""
        if spec.get("wants_draw"):
            depth = """
- Add a small legend box (e.g. lower corner) and 2–4 numbered callouts with leader lines or arrows to sockets/cables
- If helpful, add a second mini-panel showing cable destination (work vs torch) in outline form
"""

        return f"""Generate an SVG wiring/polarity diagram for the Vulcan OmniPro 220.

Question being answered: {question}
{context}
Requirements:
- Draw a simplified welder front panel (rectangle) with two clearly labeled output sockets: one marked (+) Positive and one marked (-) Negative
- Show cables as thick lines with arrowheads routing from the sockets to their destination (Ground Clamp / Work Piece, TIG Torch / Electrode)
- Label each cable clearly
- Include a title inside the SVG
- viewBox="0 0 520 360", white background, high contrast (slightly larger canvas if callouts need room)
- Use dark gray (#1e293b) for panel, orange (#f97316) for positive, black (#0f172a) for negative
- Font: sans-serif, readable
{depth}
Return ONLY the SVG markup. No markdown fences. No JSON. No explanation."""

    def _extract_svg(self, raw: str) -> str:
        match = re.search(r"(<svg[\s\S]*?</svg>)", raw)
        if match:
            return match.group(1)
        return raw.strip()
