"""HTTP-клиенты: OpenRouter (черновик whisper + LLM-судья) и ElevenLabs Scribe."""
from __future__ import annotations

import json
import random
import time
from pathlib import Path

import httpx

import config

# контейнер ходит наружу напрямую (host TUN-прокси прозрачен), env-прокси не наследуем
_client = httpx.Client(timeout=600, trust_env=False)

# Коды, которые лечатся повтором: перегрузка провайдера, лимит одновременных
# запросов (у ElevenLabs он привязан к тарифу), таймауты шлюзов.
RETRY_STATUS = {408, 409, 425, 429, 500, 502, 503, 504, 520, 521, 522, 524, 529}


class ApiError(RuntimeError):
    """Ошибка API с телом ответа: без него по одному коду 400 диагноз не поставить."""

    def __init__(self, name: str, status: int, body: str):
        self.status, self.body = status, body
        super().__init__(f"{name} вернул {status}: {body[:400]}")


def _post(name: str, url: str, tries: int = 5, **kw) -> httpx.Response:
    """POST с ретраями: сетевые обрывы, временные коды, разовые 400.

    400 ретраим намеренно. При переключении VPN на хосте наружу уходит рваный
    запрос, и в ответ прилетает 400 на совершенно корректных данных: у записи #13
    так упал один регион, а повторный прогон всех её 12 регионов прошёл целиком.
    Настоящая ошибка формата переживёт повтор и поднимется наружу с телом ответа.
    """
    delay = 2.0
    for attempt in range(1, tries + 1):
        try:
            r = _client.post(url, **kw)
            if r.status_code < 400:
                return r
            if r.status_code not in RETRY_STATUS and r.status_code != 400:
                raise ApiError(name, r.status_code, r.text)
            if attempt == tries:
                raise ApiError(name, r.status_code, r.text)
            reason = f"{r.status_code} {r.text[:120]}"
        except (httpx.TransportError, httpx.StreamError) as exc:
            if attempt == tries:
                raise
            reason = f"{type(exc).__name__}: {exc}"
        # джиттер разводит параллельные обработчики: без него после общего сбоя
        # они синхронно ломятся обратно и снова упираются в лимит
        wait = delay * (1 + random.random() * 0.3)
        print(f"[retry] {name}: {reason} — повтор {attempt}/{tries - 1} через {wait:.0f}с", flush=True)
        time.sleep(wait)
        delay = min(delay * 3, 60.0)
    raise RuntimeError("недостижимо")


def draft_transcribe(path: Path) -> tuple[dict, float]:
    """Черновик через OpenRouter whisper-turbo, verbose_json. Возвращает (ответ, цена $)."""
    data = {"model": config.DRAFT_MODEL, "response_format": "verbose_json"}
    if config.DRAFT_LANGUAGE:
        data["language"] = config.DRAFT_LANGUAGE
    r = _post("OpenRouter", "https://openrouter.ai/api/v1/audio/transcriptions",
              headers={"Authorization": f"Bearer {config.OPENROUTER_API_KEY}"},
              data=data, files={"file": (path.name, path.read_bytes())})
    data = r.json()
    return data, float((data.get("usage") or {}).get("cost") or 0.0)


def scribe_transcribe(path: Path, duration_sec: float) -> tuple[dict, float]:
    """Качественный проход ElevenLabs Scribe с диаризацией. Возвращает (ответ, цена $)."""
    r = _post("ElevenLabs", "https://api.elevenlabs.io/v1/speech-to-text",
              headers={"xi-api-key": config.ELEVENLABS_API_KEY},
              data={"model_id": config.SCRIBE_MODEL, "diarize": "true", "tag_audio_events": "true"},
              files={"file": (path.name, path.read_bytes())})
    data = r.json()
    billed = float(data.get("audio_duration_secs") or duration_sec)
    return data, billed / 3600.0 * config.SCRIBE_PRICE_PER_HOUR


def judge(prompt: str) -> tuple[dict, float]:
    """Текст-судья конфликтов диаризации (дешёвая LLM, JSON-ответ). (вердикт, цена $)."""
    r = _post("OpenRouter", "https://openrouter.ai/api/v1/chat/completions",
              headers={"Authorization": f"Bearer {config.OPENROUTER_API_KEY}"},
              json={"model": config.JUDGE_MODEL,
                    "response_format": {"type": "json_object"},
                    "usage": {"include": True},
                    "messages": [{"role": "user", "content": prompt}]})
    data = r.json()
    cost = float((data.get("usage") or {}).get("cost") or 0.0)
    try:
        verdict = json.loads(data["choices"][0]["message"]["content"])
    except Exception:
        verdict = {}
    return verdict, cost


def balances() -> list[dict]:
    """Остатки на счетах провайдеров — чтобы очередь не встала молча из-за нуля.

    ElevenLabs отдаёт квоту только ключу с правом user_read; без него честно
    говорим, что остаток недоступен, вместо того чтобы показать ноль.
    """
    out = []
    try:
        r = _client.get("https://openrouter.ai/api/v1/credits",
                        headers={"Authorization": f"Bearer {config.OPENROUTER_API_KEY}"}, timeout=20)
        d = r.json()["data"]
        left = float(d["total_credits"]) - float(d["total_usage"])
        out.append({"api": "OpenRouter", "ok": True, "left": round(left, 2), "unit": "usd",
                    "text": f"${left:,.2f}", "note": "судья диаризации + черновики"})
    except Exception as exc:
        out.append({"api": "OpenRouter", "ok": False, "text": "нет данных", "note": str(exc)[:120]})
    try:
        r = _client.get("https://api.elevenlabs.io/v1/user/subscription",
                        headers={"xi-api-key": config.ELEVENLABS_API_KEY}, timeout=20)
        if r.status_code == 401:
            out.append({"api": "ElevenLabs", "ok": False, "text": "ключу нужно право user_read",
                        "note": "elevenlabs.io → ключ → включить User: Read"})
        else:
            d = r.json()
            used, lim = int(d.get("character_count", 0)), int(d.get("character_limit", 0))
            out.append({"api": "ElevenLabs", "ok": True, "left": lim - used, "unit": "credits",
                        "text": f"{lim - used:,} из {lim:,} кредитов".replace(",", " "),
                        "note": f"тариф {d.get('tier', '?')}, транскрипция Scribe"})
    except Exception as exc:
        out.append({"api": "ElevenLabs", "ok": False, "text": "нет данных", "note": str(exc)[:120]})
    return out
