"""Уведомления в Telegram: запись готова, запись упала.

Молчаливый конвейер требует, чтобы человек сам ходил смотреть, что происходит.
Пять дней мёртвого токена Plaud прошли незамеченными именно поэтому.

Бот и получатель настраиваются в .env: TELEGRAM_BOT_TOKEN обязателен,
TELEGRAM_CHAT_ID можно не задавать — он определится сам, как только человек нажмёт
Start в боте, и запомнится в /data/telegram_chat_id.
"""
from __future__ import annotations

import time
from pathlib import Path

import httpx

import config

_client = httpx.Client(timeout=20, trust_env=False)   # системный прокси не для api.telegram
_CHAT_FILE = config.DATA / "telegram_chat_id"
_last: dict[str, float] = {}


def _api(method: str, **params):
    if not config.TELEGRAM_TOKEN:
        return None
    r = _client.post(f"https://api.telegram.org/bot{config.TELEGRAM_TOKEN}/{method}", data=params)
    if r.status_code >= 400:
        return None
    return r.json().get("result")


def chat_id() -> str | None:
    """Кому писать. Ищется один раз: из .env, из файла или из первого /start."""
    if config.TELEGRAM_CHAT_ID:
        return config.TELEGRAM_CHAT_ID
    if _CHAT_FILE.is_file():
        return _CHAT_FILE.read_text(encoding="utf-8").strip() or None
    ups = _api("getUpdates") or []
    for u in ups:
        msg = u.get("message") or u.get("edited_message") or {}
        cid = (msg.get("chat") or {}).get("id")
        if cid:
            _CHAT_FILE.write_text(str(cid), encoding="utf-8")
            return str(cid)
    return None


def send(text: str, key: str | None = None, every_sec: float = 3600) -> bool:
    """Отправить сообщение. key гасит повторы: одна и та же тема не чаще раза в час."""
    if key:
        if time.time() - _last.get(key, 0) < every_sec:
            return False
        _last[key] = time.time()
    cid = chat_id()
    if not cid:
        return False
    return bool(_api("sendMessage", chat_id=cid, text=text,
                     parse_mode="HTML", disable_web_page_preview="true"))


def recording_done(rec_id: int, title: str, minutes: float, speakers: list[str],
                   named_pct: float, cost: float) -> None:
    who = ", ".join(speakers[:4]) or "никто не опознан"
    send(f"✅ <b>Готов транскрипт</b>\n{title[:80]}\n"
         f"#{rec_id} · {minutes:.0f} мин · имена у {named_pct:.0f}% речи · ${cost:.2f}\n"
         f"Голоса: {who}")


def recording_failed(rec_id: int, title: str, stage: str, error: str) -> None:
    send(f"❌ <b>Запись упала</b>\n{title[:80]}\n#{rec_id} · этап {stage}\n{error[:200]}",
         key=f"fail:{rec_id}:{stage}")


def pipeline_problem(what: str, detail: str) -> None:
    """Общая беда конвейера: кончилась квота, облако не отвечает, токен протух."""
    send(f"⚠️ <b>{what}</b>\n{detail[:300]}", key=f"problem:{what}", every_sec=6 * 3600)
