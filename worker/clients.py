"""HTTP-клиенты: OpenRouter (черновик whisper + LLM-судья) и ElevenLabs Scribe."""
from __future__ import annotations

import json
from pathlib import Path

import httpx

import config

# контейнер ходит наружу напрямую (host TUN-прокси прозрачен), env-прокси не наследуем
_client = httpx.Client(timeout=600, trust_env=False)


def draft_transcribe(path: Path) -> tuple[dict, float]:
    """Черновик через OpenRouter whisper-turbo, verbose_json. Возвращает (ответ, цена $).

    language обязателен: без него whisper на русской речи скатывается в ПЕРЕВОД на
    английский и сыпет галлюцинациями «thank you» на паузах (проверено на реальной записи).
    """
    data = {"model": config.DRAFT_MODEL, "response_format": "verbose_json"}
    if config.DRAFT_LANGUAGE:
        data["language"] = config.DRAFT_LANGUAGE
    r = _client.post(
        "https://openrouter.ai/api/v1/audio/transcriptions",
        headers={"Authorization": f"Bearer {config.OPENROUTER_API_KEY}"},
        data=data,
        files={"file": (path.name, path.read_bytes())},
    )
    r.raise_for_status()
    data = r.json()
    cost = float((data.get("usage") or {}).get("cost") or 0.0)
    return data, cost


def scribe_transcribe(path: Path, duration_sec: float) -> tuple[dict, float]:
    """Качественный проход ElevenLabs Scribe с диаризацией. Возвращает (ответ, цена $)."""
    r = _client.post(
        "https://api.elevenlabs.io/v1/speech-to-text",
        headers={"xi-api-key": config.ELEVENLABS_API_KEY},
        data={"model_id": config.SCRIBE_MODEL, "diarize": "true", "tag_audio_events": "true"},
        files={"file": (path.name, path.read_bytes())},
    )
    r.raise_for_status()
    data = r.json()
    billed = float(data.get("audio_duration_secs") or duration_sec)
    cost = billed / 3600.0 * config.SCRIBE_PRICE_PER_HOUR
    return data, cost


def judge(prompt: str) -> tuple[dict, float]:
    """Текст-судья конфликтов диаризации (дешёвая LLM, JSON-ответ). (вердикт, цена $)."""
    r = _client.post(
        "https://openrouter.ai/api/v1/chat/completions",
        headers={"Authorization": f"Bearer {config.OPENROUTER_API_KEY}"},
        json={
            "model": config.JUDGE_MODEL,
            "response_format": {"type": "json_object"},
            "usage": {"include": True},
            "messages": [{"role": "user", "content": prompt}],
        },
    )
    r.raise_for_status()
    data = r.json()
    cost = float((data.get("usage") or {}).get("cost") or 0.0)
    try:
        verdict = json.loads(data["choices"][0]["message"]["content"])
    except Exception:
        verdict = {}
    return verdict, cost
