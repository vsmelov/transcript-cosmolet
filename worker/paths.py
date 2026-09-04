"""Где лежат артефакты записи.

Папки назывались просто `rec_17`, и в проводнике архив выглядел как список
безымянных номеров: чтобы понять, что внутри, приходилось открывать файл. Теперь
к номеру добавляется дата и человекочитаемый заголовок, а номер остаётся впереди —
он ключ, по нему папка и находится, как бы её ни переименовали потом.
"""
from __future__ import annotations

import re
from datetime import timedelta
from pathlib import Path
from zoneinfo import ZoneInfo

import config
import db

_UNSAFE = re.compile(r'[<>:"/\\|?*\x00-\x1f]')


def slug(text: str, limit: int = 60) -> str:
    s = _UNSAFE.sub("", text or "").strip()
    s = re.sub(r"\s+", "_", s)
    return s[:limit].rstrip("_.")


def dir_name(rec_id: int, title: str | None, started) -> str:
    """rec_<id>_<дата>_<название>. Дата местная — как в названиях самого Plaud."""
    parts = [f"rec_{rec_id}"]
    if started:
        parts.append(f"{started.astimezone(ZoneInfo(config.LOCAL_TZ)):%Y-%m-%d_%H-%M}")
    if title:
        parts.append(slug(title))
    return "_".join(parts)


def rec_dir(rec_id: int, create: bool = True) -> Path:
    """Папка записи: найти существующую по номеру, при надобности переименовать.

    Заголовок у записи появляется не сразу — Plaud дообрабатывает её своим ИИ и
    присылает человеческое название позже. Поэтому имя папки подтягивается при
    каждом обращении, а не фиксируется в момент создания.
    """
    root = config.ARTIFACTS
    row = db.q1("SELECT title, filename, started_at FROM recordings WHERE id = %s", rec_id)
    title, filename, started = (row or (None, None, None))
    want = root / dir_name(rec_id, title or filename, started)

    # существующая папка ищется по номеру: она могла быть создана под старым именем
    found = None
    for p in root.glob(f"rec_{rec_id}_*"):
        if p.is_dir():
            found = p
            break
    if found is None and (root / f"rec_{rec_id}").is_dir():
        found = root / f"rec_{rec_id}"

    if found is not None and found != want:
        try:
            found.rename(want)
            found = want
        except OSError:
            return found          # занято другим процессом — работаем со старым именем
    if found is None:
        if create:
            want.mkdir(parents=True, exist_ok=True)
        return want
    return found
