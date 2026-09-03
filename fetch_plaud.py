"""Скачивание записи Plaud в inbox вместе с метаданными.

Рядом с аудио кладётся sidecar <файл>.meta.json — воркер его подхватит и запишет
человеческое название и время начала записи вместо голого имени файла.

Использование:
    python fetch_plaud.py <presigned_url> <file_id> <title> <start_at_iso> [duration_ms]
"""
import json
import re
import sys
import urllib.request
from datetime import datetime
from pathlib import Path

INBOX = Path(__file__).with_name("data") / "inbox"


def slug(s: str, limit: int = 60) -> str:
    s = re.sub(r"[^\w\s-]", "", s, flags=re.UNICODE).strip()
    return re.sub(r"\s+", "_", s)[:limit] or "recording"


def main() -> None:
    url, file_id, title, start_at = sys.argv[1], sys.argv[2], sys.argv[3], sys.argv[4]
    dt = datetime.fromisoformat(start_at)
    # имя файла: дата-время + человеческое название + короткий id (уникальность)
    name = f"{dt:%Y-%m-%d_%H-%M}_{slug(title)}_{file_id[:8]}.mp3"
    INBOX.mkdir(parents=True, exist_ok=True)
    dest = INBOX / name

    # качаем во временное имя и переименовываем: воркер не должен увидеть недокачанный файл
    tmp = dest.with_suffix(dest.suffix + ".part")
    opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))  # мимо системного прокси
    with opener.open(url, timeout=900) as r, open(tmp, "wb") as f:
        while chunk := r.read(1 << 20):
            f.write(chunk)
    tmp.replace(dest)

    dest.with_suffix(dest.suffix + ".meta.json").write_text(json.dumps({
        "title": title,
        "started_at": dt.isoformat(),
        "source": "plaud",
        "plaud_file_id": file_id,
    }, ensure_ascii=False, indent=2), encoding="utf-8")

    print(f"OK {dest.name} {dest.stat().st_size / 1024 / 1024:.1f} MB")


if __name__ == "__main__":
    main()
