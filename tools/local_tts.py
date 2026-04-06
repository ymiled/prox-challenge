from __future__ import annotations

import hashlib
import re
import subprocess
import sys
import tempfile
from pathlib import Path

from app.config import Settings

try:
    import pyttsx3
except Exception:  # pragma: no cover - optional fallback
    pyttsx3 = None


class LocalTTS:
    def __init__(self, settings: Settings) -> None:
        self.settings = settings
        self.audio_dir = settings.audio_dir
        self.audio_dir.mkdir(parents=True, exist_ok=True)
        self.script_path = settings.project_root / "tools" / "synthesize_speech.ps1"
        self._windows_voice_available = False
        self._pyttsx3_ready = False
        self._probe_available_engines()

    @property
    def enabled(self) -> bool:
        return self._windows_voice_available or self._pyttsx3_ready

    def synthesize(self, text: str) -> Path | None:
        cleaned = self._clean_text(text)
        if not cleaned:
            return None

        audio_id = hashlib.sha256(cleaned.encode("utf-8")).hexdigest()[:24]
        output_path = self.audio_dir / f"{audio_id}.wav"
        if output_path.exists():
            return output_path

        temp_path = self.audio_dir / f"{audio_id}.tmp.wav"
        try:
            if self._synthesize_windows(cleaned, temp_path):
                temp_path.replace(output_path)
                return output_path
            if self._synthesize_pyttsx3(cleaned, temp_path):
                temp_path.replace(output_path)
                return output_path
        finally:
            temp_path.unlink(missing_ok=True)
        return None

    def _synthesize_windows(self, text: str, output_path: Path) -> bool:
        if not sys.platform.startswith("win") or not self.script_path.exists() or not self._windows_voice_available:
            return False
        command = [
            "powershell",
            "-ExecutionPolicy",
            "Bypass",
            "-File",
            str(self.script_path),
            "-Text",
            text,
            "-OutputPath",
            str(output_path),
        ]
        try:
            completed = subprocess.run(command, capture_output=True, text=True, check=False)
        except OSError:
            return False
        return completed.returncode == 0 and output_path.exists()

    def _detect_windows_voice(self) -> bool:
        command = [
            "powershell",
            "-ExecutionPolicy",
            "Bypass",
            "-Command",
            "Add-Type -AssemblyName System.Speech; "
            "$s = New-Object System.Speech.Synthesis.SpeechSynthesizer; "
            "try { (($s.GetInstalledVoices() | Where-Object { $_.Enabled }).Count -gt 0) } finally { $s.Dispose() }",
        ]
        try:
            completed = subprocess.run(command, capture_output=True, text=True, check=False)
        except OSError:
            return False
        return completed.returncode == 0 and "True" in completed.stdout

    def _probe_available_engines(self) -> None:
        if sys.platform.startswith("win") and self._detect_windows_voice():
            self._windows_voice_available = self._probe_windows_synthesis()
        if pyttsx3 is not None:
            self._pyttsx3_ready = self._probe_pyttsx3_synthesis()

    def _probe_windows_synthesis(self) -> bool:
        probe_path = self._probe_path("windows")
        try:
            return self._synthesize_windows("Prox local voice readiness check.", probe_path)
        finally:
            probe_path.unlink(missing_ok=True)

    def _probe_pyttsx3_synthesis(self) -> bool:
        probe_path = self._probe_path("pyttsx3")
        try:
            return self._run_pyttsx3("Prox local voice readiness check.", probe_path)
        finally:
            probe_path.unlink(missing_ok=True)

    def _probe_path(self, suffix: str) -> Path:
        with tempfile.NamedTemporaryFile(dir=self.audio_dir, prefix=f"probe-{suffix}-", suffix=".wav", delete=False) as handle:
            return Path(handle.name)

    def _synthesize_pyttsx3(self, text: str, output_path: Path) -> bool:
        if pyttsx3 is None or not self._pyttsx3_ready:
            return False
        return self._run_pyttsx3(text, output_path)

    def _run_pyttsx3(self, text: str, output_path: Path) -> bool:
        try:
            engine = pyttsx3.init()
            engine.setProperty("rate", 175)
            engine.setProperty("volume", 1.0)
            voices = engine.getProperty("voices") or []
            preferred_voice_id = None
            for voice in voices:
                voice_name = getattr(voice, "name", "")
                voice_text = f"{getattr(voice, 'id', '')} {voice_name}".lower()
                if voice_name == "Microsoft Zira Desktop":
                    preferred_voice_id = getattr(voice, "id", "")
                    break
                if not preferred_voice_id and voice_name == "Microsoft David Desktop":
                    preferred_voice_id = getattr(voice, "id", "")
                if not preferred_voice_id and ("english" in voice_text or "en-us" in voice_text or "en_gb" in voice_text):
                    preferred_voice_id = getattr(voice, "id", "")
            if preferred_voice_id:
                engine.setProperty("voice", preferred_voice_id)
            engine.save_to_file(text, str(output_path))
            engine.runAndWait()
            engine.stop()
        except Exception:
            return False
        return output_path.exists()

    def _detect_pyttsx3(self) -> bool:
        try:
            engine = pyttsx3.init()
            engine.stop()
            return True
        except Exception:
            return False

    def _clean_text(self, text: str) -> str:
        cleaned = (
            text.replace("•", " ")
            .replace("·", " ")
            .replace("—", ", ")
            .replace("–", ", ")
        )
        cleaned = re.sub(r"```[\s\S]*?```", " ", cleaned)
        cleaned = re.sub(r"`([^`]+)`", r"\1", cleaned)
        cleaned = re.sub(r"\*\*([^*]+)\*\*", r"\1", cleaned)
        cleaned = re.sub(r"\*([^*]+)\*", r"\1", cleaned)
        cleaned = re.sub(r"__([^_]+)__", r"\1", cleaned)
        cleaned = re.sub(r"_([^_]+)_", r"\1", cleaned)
        cleaned = re.sub(r"\[(.*?)\]\((.*?)\)", r"\1", cleaned)
        cleaned = re.sub(r"</?[^>]+>", " ", cleaned)
        cleaned = re.sub(r"\b([A-Za-z0-9_-]+)\s+p\.(\d+)\b", r"\1 page \2", cleaned)
        cleaned = re.sub(r"^\s*[-*]\s+", "", cleaned, flags=re.MULTILINE)
        cleaned = re.sub(r"^\s*\d+\.\s+", "", cleaned, flags=re.MULTILINE)
        cleaned = re.sub(r"\s+", " ", cleaned).strip()
        return cleaned
