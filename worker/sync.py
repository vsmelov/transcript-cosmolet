"""Синхронизация с облаком Plaud: новые записи -> очередь на скачивание.

Отдельной очереди (celery и т.п.) не заводим: очередь у нас уже есть — таблица
recordings со статусами и jobs с этапами. Синк лишь добавляет строки со статусом
'pending_download', а воркер разбирает их тем же циклом, что и остальную работу.
"""
from __future__ import annotations

import json
import re
import time
from datetime import datetime
from pathlib import Path

import config
import db
import plaud

_UNSAFE = re.compile(r'[<>:"/\\|?*\x00-\x1f]')


def safe_name(title: str, started: str, file_id: str) -> str:
    """Человекочитаемое имя файла: дата, название, короткий id для уникальности."""
    try:
        dt = datetime.fromisoformat(started)
        prefix = f"{dt:%Y-%m-%d_%H-%M}"
    except Exception:
        prefix = "unknown-date"
    clean = _UNSAFE.sub("", title or "").strip()
    clean = re.sub(r"\s+", "_", clean)[:60] or "recording"
    return f"{prefix}_{clean}_{file_id[:8]}.mp3"


def sync(log=print) -> dict:
    """Сверить облако с базой; на новые записи завести строки в очередь."""
    added, seen, renamed = 0, 0, 0
    page = 1
    while True:
        try:
            items = plaud.list_files(page=page, page_size=100)
        except Exception as exc:
            log(f"синк: не удалось получить список (стр. {page}): {exc}")
            break
        if not items:
            break
        for it in items:
            fid = it.get("id")
            if not fid:
                continue
            seen += 1
            title = it.get("name") or fid
            row = db.q1("SELECT id, title FROM recordings WHERE meta->>'plaud_file_id' = %s", fid)
            if row:
                # Plaud дообрабатывает записи и переименовывает их в облаке — подтягиваем
                if row[1] != title:
                    db.q("UPDATE recordings SET title = %s WHERE id = %s", title, row[0])
                    renamed += 1
                continue
            dur = float(it.get("duration") or 0) / 1000.0
            meta = {"plaud_file_id": fid, "source": "plaud",
                    "serial_number": it.get("serial_number"),
                    "cloud_name": title}
            db.q("""INSERT INTO recordings
                    (filename, title, source, duration_sec, started_at, status, meta)
                    VALUES (%s,%s,'plaud',%s,%s,'pending_download',%s::jsonb)""",
                 safe_name(title, it.get("start_at") or "", fid), title, dur,
                 it.get("start_at"), json.dumps(meta, ensure_ascii=False))
            added += 1
        if len(items) < 100:
            break
        page += 1
    if added or renamed:
        log(f"синк: в облаке {seen}, новых {added}, переименовано {renamed}")
    return {"seen": seen, "added": added, "renamed": renamed}


def download_one(rec_id: int, log=print) -> bool:
    """Скачать одну запись из облака. Этап виден в UI как job 'download'."""
    rec = db.q1("SELECT filename, title, meta FROM recordings WHERE id = %s", rec_id)
    if not rec:
        return False
    filename, title, meta = rec
    if isinstance(meta, str):
        meta = json.loads(meta)
    fid = (meta or {}).get("plaud_file_id")
    if not fid:
        db.q("UPDATE recordings SET status='failed' WHERE id=%s", rec_id)
        return False

    job = db.job_start(rec_id, "download")
    dest = config.AUDIO / filename
    try:
        t0 = time.time()
        url = plaud.audio_url(fid)
        if not url:
            raise RuntimeError("облако не отдало ссылку на аудио (presigned_url пуст)")
        size = plaud.download(url, dest)
        db.q("""UPDATE recordings SET audio_path=%s, size_bytes=%s, status='new'
                WHERE id=%s""", str(dest), size, rec_id)
        db.job_done(job, 0.0, None, {"mb": round(size / 1048576, 1),
                                     "sec": round(time.time() - t0)})
        log(f"скачано: {title} ({size/1048576:.0f} МБ за {time.time()-t0:.0f}с)")
        return True
    except Exception as exc:
        db.job_fail(job, str(exc)[:500])
        # не хороним запись насмерть: облако могло просто не успеть подготовить файл.
        # После нескольких неудач ставим статус failed, до этого — обратно в очередь.
        tries = int((meta or {}).get("download_tries", 0)) + 1
        status = "failed" if tries >= 3 else "pending_download"
        db.q("""UPDATE recordings
                SET status=%s, meta = meta || jsonb_build_object('download_tries', %s::int)
                WHERE id=%s""", status, tries, rec_id)
        log(f"не скачалось #{rec_id} (попытка {tries}): {str(exc)[:160]}")
        # частичный файл не оставляем — иначе следующий заход примет его за целый
        Path(str(dest) + ".part").unlink(missing_ok=True)
        return False


def retry_failed(log=print) -> int:
    """Вернуть в очередь записи, которые не скачались меньше трёх раз."""
    rows = db.q("""UPDATE recordings SET status='pending_download'
                   WHERE status='failed' AND audio_path IS NULL
                     AND coalesce((meta->>'download_tries')::int, 0) < 3
                   RETURNING id""") or []
    if rows:
        log(f"возвращено в очередь скачивания: {len(rows)}")
    return len(rows)


def sync_loop(log=print) -> None:
    """Опрос облака Plaud: новые записи попадают в очередь скачивания."""
    while True:
        try:
            sync(log)
            retry_failed(log)
        except Exception as exc:
            log("синк с облаком:", str(exc)[:200])
        time.sleep(config.PLAUD_SYNC_EVERY_SEC)


def downloader_loop(num: int = 1, log=print) -> None:
    """Качальщик: берёт файл из очереди и тянет его из облака.

    Живёт отдельно от обработки — иначе выкачивание сотни часов заблокировало бы
    транскрибацию до последнего файла. Захват атомарный: несколько качальщиков
    работают параллельно и не берут один файл дважды.
    """
    while True:
        try:
            row = db.q1("""
                UPDATE recordings SET status='downloading'
                WHERE id = (SELECT id FROM recordings WHERE status='pending_download'
                            ORDER BY started_at DESC NULLS LAST, id
                            FOR UPDATE SKIP LOCKED LIMIT 1)
                RETURNING id""")
            if row:
                download_one(row[0], log)
            else:
                time.sleep(30)
        except Exception as exc:
            log(f"качальщик {num}:", str(exc)[:200])
            time.sleep(60)
