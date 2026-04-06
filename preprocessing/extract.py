from __future__ import annotations

import re
from pathlib import Path
from typing import Any

import fitz

from app.config import Settings
from app.utils import ensure_dir, safe_excerpt, write_json


class Preprocessor:
    CACHE_VERSION = 3

    def __init__(self, settings: Settings) -> None:
        self.settings = settings

    def run(self) -> dict[str, Any]:
        page_index: list[dict[str, Any]] = []
        chunk_index: list[dict[str, Any]] = []
        structured: dict[str, Any] = {
            "duty_cycles": {"pages": [], "entries": []},
            "wire_settings": {"pages": [], "entries": []},
            "troubleshooting": {"pages": [], "entries": []},
            "parts_list": {"pages": [], "entries": []},
        }

        ensure_dir(self.settings.cache_dir)
        ensure_dir(self.settings.pages_dir)
        ensure_dir(self.settings.text_dir)
        ensure_dir(self.settings.structured_dir)

        for pdf_path in sorted(self.settings.files_dir.glob("*.pdf")):
            pages, chunks = self._process_document(pdf_path, structured)
            page_index.extend(pages)
            chunk_index.extend(chunks)

        write_json(self.settings.structured_dir / "page_index.json", page_index)
        write_json(self.settings.structured_dir / "chunk_index.json", chunk_index)
        for name, payload in structured.items():
            write_json(self.settings.structured_dir / f"{name}.json", payload)

        manifest = {
            "documents": sorted(path.name for path in self.settings.files_dir.glob("*.pdf")),
            "pages_indexed": len(page_index),
            "chunks_indexed": len(chunk_index),
            "structured_outputs": sorted(structured.keys()),
            "cache_version": self.CACHE_VERSION,
        }
        write_json(self.settings.structured_dir / "manifest.json", manifest)
        return manifest

    def _process_document(
        self, pdf_path: Path, structured: dict[str, Any]
    ) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
        doc_name = pdf_path.stem
        doc = fitz.open(pdf_path)
        toc = doc.get_toc(simple=False)
        entries: list[dict[str, Any]] = []
        chunk_entries: list[dict[str, Any]] = []
        pages_root = self.settings.pages_dir / doc_name
        text_root = self.settings.text_dir / doc_name
        ensure_dir(pages_root)
        ensure_dir(text_root)

        for page_number in range(doc.page_count):
            page = doc.load_page(page_number)
            page_label = f"{page_number + 1:03d}"
            text = page.get_text("text")
            (text_root / f"{page_label}.txt").write_text(text, encoding="utf-8")

            pix = page.get_pixmap(dpi=self.settings.render_dpi, alpha=False)
            pix.save(pages_root / f"{page_label}.png")

            section = self._find_section(toc, page_number + 1)
            normalized_text = self._normalize_text(text)
            topics = self._infer_topics(normalized_text, section)
            entry = {
                "doc": doc_name,
                "source_file": pdf_path.name,
                "page": page_number + 1,
                "section": section,
                "excerpt": safe_excerpt(normalized_text),
                "topics": topics,
                "full_text": normalized_text,
            }
            entries.append(entry)
            self._harvest_structured(entry, normalized_text, structured)
            chunk_entries.extend(
                self._build_chunk_entries(
                    page=page,
                    page_entry=entry,
                    doc_name=doc_name,
                    page_number=page_number + 1,
                    section=section,
                    normalized_text=normalized_text,
                    topics=topics,
                )
            )

        doc.close()
        return entries, chunk_entries

    def _find_section(self, toc: list[list[Any]], page_number: int) -> str | None:
        current = None
        for entry in toc:
            if len(entry) < 3:
                continue
            if int(entry[2]) <= page_number:
                current = str(entry[1])
            else:
                break
        return current

    def _infer_topics(self, text: str, section: str | None) -> list[str]:
        blob = f"{section or ''}\n{text}".lower()
        topics = []
        rules = {
            "duty_cycle": ["duty cycle", "%", "amp"],
            "polarity": ["polarity", "positive", "negative", "dinse", "ground clamp"],
            "troubleshooting": ["troubleshooting", "porosity", "spatter", "undercut", "diagnosis"],
            "wire_settings": ["wire feed", "voltage", "thickness", "mild steel", "flux", "mig"],
            "parts_list": ["parts list", "assembly diagram", "part no", "part number"],
            "safety": ["warning", "electric shock", "fumes", "fire"],
        }
        for topic, keywords in rules.items():
            if any(keyword in blob for keyword in keywords):
                topics.append(topic)
        return topics

    def _harvest_structured(
        self,
        entry: dict[str, Any],
        text: str,
        structured: dict[str, Any],
    ) -> None:
        lowered = text.lower()
        if "duty cycle" in lowered:
            structured["duty_cycles"]["pages"].append(entry)
            structured["duty_cycles"]["entries"].extend(
                self._extract_duty_cycle_entries(entry, text)
            )

        if any(keyword in lowered for keyword in ["wire feed", "wire speed", "voltage"]):
            structured["wire_settings"]["pages"].append(entry)
            structured["wire_settings"]["entries"].extend(
                self._extract_wire_setting_entries(entry, text)
            )

        if any(keyword in lowered for keyword in ["troubleshooting", "porosity", "spatter", "undercut"]):
            structured["troubleshooting"]["pages"].append(entry)
            structured["troubleshooting"]["entries"].extend(
                self._extract_troubleshooting_entries(entry, text)
            )

        if any(keyword in lowered for keyword in ["parts list", "part no", "part number", "assembly diagram"]):
            structured["parts_list"]["pages"].append(entry)
            lines = [line.strip() for line in text.splitlines() if line.strip()]
            for line in lines:
                if re.match(r"^\d+\s+[A-Za-z]", line):
                    structured["parts_list"]["entries"].append(
                        {
                            "doc": entry["doc"],
                            "page": entry["page"],
                            "section": entry["section"],
                            "line": line,
                        }
                    )

    def _extract_duty_cycle_entries(self, entry: dict[str, Any], text: str) -> list[dict[str, Any]]:
        lines = [line.strip() for line in text.splitlines() if line.strip()]
        entries: list[dict[str, Any]] = []
        current_process: str | None = None
        current_voltages: list[str] = []

        i = 0
        while i < len(lines):
            line = lines[i]
            upper = line.upper()
            if upper in {"MIG", "TIG", "STICK"}:
                current_process = upper
                current_voltages = []
            elif line.lower() == "power input":
                current_voltages = []
                j = i + 1
                while j < len(lines) and len(current_voltages) < 2:
                    voltage_match = re.search(r"\b(120|240)\s*VAC\b", lines[j], re.I)
                    if voltage_match:
                        current_voltages.append(f"{voltage_match.group(1)}V")
                    j += 1
            elif line.lower().startswith("rated duty cycle"):
                duty_lines: list[str] = []
                j = i + 1
                while j < len(lines) and len(duty_lines) < 4:
                    if re.search(r"\d{1,3}\s*%\s*@\s*\d{2,3}\s*A", lines[j], re.I):
                        duty_lines.append(lines[j])
                    elif duty_lines:
                        break
                    j += 1

                for duty_index, duty_line in enumerate(duty_lines):
                    match = re.search(r"(\d{1,3})\s*%\s*@\s*(\d{2,3})\s*A", duty_line, re.I)
                    if not match:
                        continue
                    percent, amps = match.groups()
                    voltage = None
                    if current_voltages:
                        voltage = current_voltages[0 if duty_index < 2 else min(1, len(current_voltages) - 1)]
                    entries.append(
                        {
                            "doc": entry["doc"],
                            "page": entry["page"],
                            "section": entry["section"],
                            "process": current_process,
                            "input_voltage": voltage,
                            "amperage": int(amps),
                            "duty_cycle_percent": int(percent),
                            "source_line": duty_line,
                        }
                    )
            i += 1

        return entries

    def _extract_wire_setting_entries(self, entry: dict[str, Any], text: str) -> list[dict[str, Any]]:
        lines = [line.strip() for line in text.splitlines() if line.strip()]
        entries: list[dict[str, Any]] = []
        current_process = self._infer_process(entry.get("section") or "")
        current_material = self._infer_material(entry.get("section") or "")
        recent_lines: list[str] = []

        for line in lines:
            inferred_process = self._infer_process(line)
            inferred_material = self._infer_material(line)
            if inferred_process:
                current_process = inferred_process
            if inferred_material:
                current_material = inferred_material

            has_setting_value = (
                re.search(r"\b\d{1,3}\s*-\s*\d{1,3}\s*v\b", line, re.I)
                or re.search(r"\b\d{2,4}\s*ipm\b", line, re.I)
            )
            if not has_setting_value:
                recent_lines.append(line)
                recent_lines = recent_lines[-3:]
                continue

            context = recent_lines[-2:] + [line]
            entries.append(
                {
                    "doc": entry["doc"],
                    "page": entry["page"],
                    "section": entry["section"],
                    "process": current_process,
                    "material": current_material,
                    "thickness": self._extract_thickness(" | ".join(context)),
                    "voltage": self._extract_voltage(line),
                    "wire_feed_speed": self._extract_wire_feed(line),
                    "source_line": line,
                    "context_lines": context,
                }
            )
            recent_lines.append(line)
            recent_lines = recent_lines[-3:]

        return entries

    def _extract_troubleshooting_entries(self, entry: dict[str, Any], text: str) -> list[dict[str, Any]]:
        lines = [line.strip() for line in text.splitlines() if line.strip()]
        symptom_terms = ["porosity", "spatter", "undercut", "overlap", "crack", "worm", "fusion", "bead"]
        entries: list[dict[str, Any]] = []

        for idx, line in enumerate(lines):
            lowered = line.lower()
            matches = [term for term in symptom_terms if term in lowered]
            if not matches:
                continue
            context = [lines[i] for i in range(max(0, idx - 1), min(len(lines), idx + 3))]
            entries.append(
                {
                    "doc": entry["doc"],
                    "page": entry["page"],
                    "section": entry["section"],
                    "symptoms": matches,
                    "line": line,
                    "context_lines": context,
                }
            )

        return entries

    def _infer_process(self, text: str) -> str | None:
        lowered = text.lower()
        if "flux" in lowered or "fcaw" in lowered:
            return "FLUX-CORED"
        if "mig" in lowered or "gmaw" in lowered:
            return "MIG"
        if "tig" in lowered or "gtaw" in lowered:
            return "TIG"
        if "stick" in lowered or "smaw" in lowered:
            return "STICK"
        return None

    def _infer_material(self, text: str) -> str | None:
        lowered = text.lower()
        if "mild steel" in lowered:
            return "mild steel"
        if "stainless" in lowered:
            return "stainless steel"
        if "aluminum" in lowered or "aluminium" in lowered:
            return "aluminum"
        if "steel" in lowered:
            return "steel"
        return None

    def _extract_voltage(self, line: str) -> str | None:
        match = re.search(r"\b(\d{1,3}\s*-\s*\d{1,3}\s*v)\b", line, re.I)
        return re.sub(r"\s+", "", match.group(1)) if match else None

    def _extract_wire_feed(self, line: str) -> str | None:
        match = re.search(r"\b(\d{2,4}\s*ipm)\b", line, re.I)
        return re.sub(r"\s+", "", match.group(1)) if match else None

    def _extract_thickness(self, text: str) -> str | None:
        patterns = [
            r"\b\d+\s*/\s*\d+\s*(?:in|inch)\b",
            r"\b\d+(?:\.\d+)?\s*mm\b",
            r"\b\d+\s*gauge\b",
        ]
        for pattern in patterns:
            match = re.search(pattern, text, re.I)
            if match:
                return re.sub(r"\s+", " ", match.group(0)).strip()
        return None

    def _normalize_text(self, text: str) -> str:
        replacements = {
            "\u200a": " ",
            "\u2011": "-",
            "\u2013": "-",
            "\u2014": "-",
            "\u00a0": " ",
            "â€Š": " ",
            "â€“": "-",
            "â€”": "-",
            "â€‘": "-",
        }
        normalized = text
        for old, new in replacements.items():
            normalized = normalized.replace(old, new)
        return normalized

    def _build_chunk_entries(
        self,
        page: fitz.Page,
        page_entry: dict[str, Any],
        doc_name: str,
        page_number: int,
        section: str | None,
        normalized_text: str,
        topics: list[str],
    ) -> list[dict[str, Any]]:
        chunks: list[dict[str, Any]] = []
        text_chunks = self._extract_text_chunks(page, normalized_text)
        if not text_chunks and normalized_text.strip():
            text_chunks = [normalized_text]

        for idx, chunk_text in enumerate(text_chunks):
            cleaned = self._normalize_text(chunk_text)
            if len(cleaned.strip()) < 40:
                continue
            chunks.append(
                {
                    "chunk_id": f"{doc_name}:{page_number}:text:{idx}",
                    "chunk_type": "text",
                    "doc": doc_name,
                    "page": page_number,
                    "section": section,
                    "topics": topics,
                    "excerpt": safe_excerpt(cleaned),
                    "full_text": cleaned,
                    "source_page_excerpt": page_entry.get("excerpt"),
                }
            )

        visual_summary = self._build_visual_summary(page, page_entry, doc_name, page_number, section, normalized_text, topics)
        if visual_summary:
            chunks.append(
                {
                    "chunk_id": f"{doc_name}:{page_number}:visual:0",
                    "chunk_type": "visual",
                    "doc": doc_name,
                    "page": page_number,
                    "section": section,
                    "topics": sorted(set(topics + ["visual"])),
                    "excerpt": safe_excerpt(visual_summary),
                    "full_text": visual_summary,
                    "source_page_excerpt": page_entry.get("excerpt"),
                }
            )

        return chunks

    def _extract_text_chunks(self, page: fitz.Page, normalized_text: str) -> list[str]:
        chunks: list[str] = []
        blocks = page.get_text("blocks")
        for block in blocks:
            if len(block) < 5:
                continue
            text = self._normalize_text(str(block[4]))
            text = " ".join(text.split())
            if len(text) >= 40:
                chunks.append(text)

        if len(chunks) >= 2:
            return self._merge_short_chunks(chunks)

        paragraph_chunks = [
            " ".join(part.split())
            for part in re.split(r"\n\s*\n+", normalized_text)
            if len(" ".join(part.split())) >= 40
        ]
        return self._merge_short_chunks(paragraph_chunks)

    def _merge_short_chunks(self, chunks: list[str], min_len: int = 120) -> list[str]:
        merged: list[str] = []
        buffer = ""
        for chunk in chunks:
            candidate = f"{buffer} {chunk}".strip() if buffer else chunk
            if len(candidate) < min_len:
                buffer = candidate
                continue
            merged.append(candidate)
            buffer = ""
        if buffer:
            if merged:
                merged[-1] = f"{merged[-1]} {buffer}".strip()
            else:
                merged.append(buffer)
        return merged

    def _build_visual_summary(
        self,
        page: fitz.Page,
        page_entry: dict[str, Any],
        doc_name: str,
        page_number: int,
        section: str | None,
        normalized_text: str,
        topics: list[str],
    ) -> str | None:
        drawings = len(page.get_drawings())
        images = len(page.get_images(full=True))
        lowered = normalized_text.lower()
        visual_terms = [
            "diagram",
            "schematic",
            "chart",
            "photo",
            "image",
            "layout",
            "selection",
            "control",
            "diagnosis",
            "socket",
            "polarity",
            "panel",
        ]
        visual_cues = sum(1 for term in visual_terms if term in lowered)
        doc_visual = doc_name in {"selection-chart"} or "selection chart" in lowered
        if not doc_visual and drawings == 0 and images == 0 and visual_cues == 0:
            return None

        labels = self._extract_visual_labels(normalized_text)
        label_text = ", ".join(labels[:10]) if labels else ""
        summary_parts = [
            f"Visual reference for {doc_name} page {page_number}.",
        ]
        if section:
            summary_parts.append(f"Section: {section}.")
        if topics:
            summary_parts.append(f"Topics: {', '.join(topics)}.")
        if doc_visual:
            summary_parts.append("This page is part of the process selection chart and should be retrieved for process-choice questions.")
        if any(term in lowered for term in ["diagnosis", "porosity", "spatter", "undercut"]):
            summary_parts.append("This page contains weld diagnosis visuals and should be retrieved for troubleshooting symptoms.")
        if any(term in lowered for term in ["polarity", "socket", "ground clamp", "torch", "dinse"]):
            summary_parts.append("This page contains wiring, polarity, or socket layout information and should be retrieved for setup and connection questions.")
        if label_text:
            summary_parts.append(f"Visible labels and captions include: {label_text}.")
        summary_parts.append(f"Visual density markers: drawings={drawings}, images={images}, visual_terms={visual_cues}.")
        return " ".join(summary_parts)

    def _extract_visual_labels(self, normalized_text: str) -> list[str]:
        labels: list[str] = []
        for line in normalized_text.splitlines():
            compact = " ".join(line.split())
            if len(compact) < 4:
                continue
            if compact.isupper() or any(token in compact.lower() for token in ["diagram", "chart", "weld", "socket", "polarity", "panel", "selection"]):
                labels.append(compact)
        seen: set[str] = set()
        deduped: list[str] = []
        for label in labels:
            key = label.lower()
            if key in seen:
                continue
            seen.add(key)
            deduped.append(label)
        return deduped
