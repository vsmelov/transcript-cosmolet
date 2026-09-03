"""Клиент Plaud: список записей и ссылки на аудио.

Публичного API у Plaud нет, но есть официальный CLI, который ходит в
platform.plaud.ai/developer/api с OAuth-токеном из ~/.plaud/tokens.json.
Мы используем тот же токен напрямую — так в контейнер не нужен Node.
Токен обновляется по refresh_token, когда истекает.
"""
from __future__ import annotations

import json
import time
from pathlib import Path

import httpx

import config

API = "https://platform.plaud.ai/developer/api"
_client = httpx.Client(timeout=120, trust_env=False, follow_redirects=True)


def _tokens_path() -> Path:
    return Path(config.PLAUD_TOKENS)


def _load() -> dict:
    p = _tokens_path()
    if not p.is_file():
        raise RuntimeError(
            f"нет {p} — на хосте нужно один раз выполнить `plaud login`, "
            f"а файл токенов примонтировать в контейнер")
    return json.loads(p.read_text(encoding="utf-8"))


def _save(data: dict) -> None:
    try:
        _tokens_path().write_text(json.dumps(data, indent=2), encoding="utf-8")
    except Exception:
        pass    # том может быть примонтирован только на чтение — не критично


def access_token() -> str:
    t = _load()
    # expires_at в миллисекундах; обновляем заранее, с запасом в минуту
    if int(t.get("expires_at", 0)) - 60_000 < int(time.time() * 1000):
        r = _client.post(f"{API}/oauth/third-party/access-token/refresh",
                         json={"refresh_token": t["refresh_token"]})
        r.raise_for_status()
        data = r.json()
        fresh = data.get("data") or data
        t = {**t, **{k: fresh[k] for k in ("access_token", "refresh_token", "expires_at")
                     if k in fresh}}
        _save(t)
    return t["access_token"]


def _get(path: str, **params) -> dict:
    r = _client.get(f"{API}{path}", params=params or None,
                    headers={"Authorization": f"Bearer {access_token()}"})
    r.raise_for_status()
    data = r.json()
    return data.get("data", data)


def list_files(page: int = 1, page_size: int = 100) -> list[dict]:
    """Записи в облаке, свежие первыми."""
    data = _get("/open/third-party/files/", page=page, page_size=page_size)
    items = data if isinstance(data, list) else (data.get("data") or data.get("items") or [])
    return [i for i in items if isinstance(i, dict)]


def file_info(file_id: str) -> dict:
    """Полная карточка записи, включая presigned_url на аудио (живёт 24 часа)."""
    return _get(f"/open/third-party/files/{file_id}")


def audio_url(file_id: str, tries: int = 20, wait: float = 30.0) -> str | None:
    """Ссылка на аудио. Plaud готовит её асинхронно: первый запрос обычно возвращает
    null и лишь запускает подготовку, поэтому повторяем с паузой.

    Терпение большое (до ~10 минут): многочасовым записям облако готовит файл дольше,
    и на них ожидание в пару минут стабильно не срабатывало."""
    for i in range(tries):
        try:
            info = file_info(file_id)
            url = info.get("presigned_url") or (info.get("file") or {}).get("presigned_url")
            if url:
                return url
        except (httpx.HTTPStatusError, httpx.TransportError) as exc:
            # 502/503 у Plaud — обычное дело, особенно когда он готовит длинный файл.
            # Такой ответ не должен сжигать попытку целиком: ждём и спрашиваем снова,
            # иначе запись уезжает в failed из-за чужой пятиминутной аварии.
            code = getattr(getattr(exc, "response", None), "status_code", None)
            if code and code < 500 and code != 429:
                raise
        if i < tries - 1:
            time.sleep(wait)
    return None


def download(url: str, dest: Path) -> int:
    """Качаем во временный .part и переименовываем — воркер не должен увидеть недокачанное."""
    tmp = dest.with_suffix(dest.suffix + ".part")
    dest.parent.mkdir(parents=True, exist_ok=True)
    with _client.stream("GET", url) as r:
        r.raise_for_status()
        with open(tmp, "wb") as f:
            for chunk in r.iter_bytes(1 << 20):
                f.write(chunk)
    tmp.replace(dest)
    return dest.stat().st_size
