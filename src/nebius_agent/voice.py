"""
Voice output via Cartesia (Sonic). The Studio speaks the agent's spoken summary at the end of a run.

    audio = await synthesize("...")   # -> VoiceResult(mode="cartesia", path=".../x.mp3") or mode="browser"

With no CARTESIA_API_KEY the UI falls back to the browser's SpeechSynthesis API so the demo still
talks; the result says which path was used so nobody mistakes one for the other.
"""

from __future__ import annotations

import os
import time
import uuid
from dataclasses import dataclass
from pathlib import Path

from .config import Settings


@dataclass
class VoiceResult:
    mode: str  # cartesia | browser | none
    text: str
    path: str | None = None
    model: str | None = None
    voice_id: str | None = None
    elapsed_ms: float = 0.0
    error: str | None = None


_DEFAULT_MODELS = [m for m in os.getenv("CARTESIA_MODEL", "sonic-3,sonic-2").split(",") if m]


async def pick_voice_id(client) -> str:
    """Use CARTESIA_VOICE_ID, else the first English voice the account can see."""
    vid = os.getenv("CARTESIA_VOICE_ID")
    if vid:
        return vid
    page = await client.voices.list(limit=20)
    voices = list(getattr(page, "data", None) or page or [])
    for v in voices:
        lang = getattr(v, "language", None) or (v.get("language") if isinstance(v, dict) else None)
        if lang in (None, "en"):
            return getattr(v, "id", None) or v["id"]
    if voices:
        return getattr(voices[0], "id", None) or voices[0]["id"]
    raise RuntimeError("no Cartesia voices available; set CARTESIA_VOICE_ID")


async def _collect(res) -> bytes:
    """Cartesia SDK versions differ: tts.bytes() may return bytes, an async iterator, or a coroutine
    resolving to either. Accept all of them."""
    import inspect

    if inspect.isawaitable(res):
        res = await res
    if isinstance(res, (bytes, bytearray)):
        return bytes(res)
    if hasattr(res, "__aiter__"):
        return b"".join([c async for c in res])
    if hasattr(res, "__iter__"):
        return b"".join(res)
    raise TypeError(f"unexpected Cartesia response type {type(res).__name__}")


async def synthesize(text: str, settings: Settings, out_dir: str = "audio") -> VoiceResult:
    key = os.getenv("CARTESIA_API_KEY", "")
    text = text.strip()
    if not key:
        return VoiceResult(mode="browser", text=text)
    from cartesia import AsyncCartesia

    t0 = time.perf_counter()
    client = AsyncCartesia(api_key=key)
    try:
        voice_id = await pick_voice_id(client)
        last_err: Exception | None = None
        for model_id in _DEFAULT_MODELS:
            try:
                res = client.tts.bytes(
                    model_id=model_id,
                    transcript=text,
                    voice={"mode": "id", "id": voice_id},
                    output_format={"container": "mp3", "bit_rate": 128000, "sample_rate": 44100},
                    language="en",
                )
                data = await _collect(res)
                if not data:
                    raise RuntimeError("empty audio")
                Path(out_dir).mkdir(parents=True, exist_ok=True)
                path = Path(out_dir) / f"{uuid.uuid4().hex[:12]}.mp3"
                path.write_bytes(data)
                return VoiceResult("cartesia", text, str(path), model_id, voice_id, round((time.perf_counter() - t0) * 1000, 1))
            except Exception as e:  # noqa: BLE001 - try the next model id
                last_err = e
        return VoiceResult("browser", text, error=f"cartesia failed: {last_err}")
    finally:
        try:
            await client.close()
        except Exception:  # noqa: BLE001
            pass
