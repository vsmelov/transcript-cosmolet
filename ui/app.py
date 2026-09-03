"""cosmolet UI — FastAPI поверх Postgres: статика, API, аудио с HTTP Range, разметка голосов."""
import json
import mimetypes
import os
import re
import threading
import time
from pathlib import Path

import psycopg
from psycopg.rows import dict_row
from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import FileResponse, Response, StreamingResponse

import voice

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.environ.get("DATA_DIR", "/data")
DB_URL = os.environ.get("DATABASE_URL", "postgresql://cosmolet:cosmolet@localhost:5432/cosmolet")
DATA_FOLDERS = ["inbox", "in_progress", "done", "failed", "artifacts", "audio"]
SPEAKERS_DIR = os.path.join(DATA_DIR, "speakers")

try:  # тот же предохранитель, что у воркера — показываем его на вкладке Usage
    DAILY_BUDGET_USD = float(os.environ.get("DAILY_BUDGET_USD", "2.0"))
except ValueError:
    DAILY_BUDGET_USD = 2.0

# мягкое исключение сэмпла из эталона (можно вернуть) — колонки может не быть в старой БД
MIGRATIONS = [
    "ALTER TABLE speaker_samples ADD COLUMN IF NOT EXISTS is_active boolean NOT NULL DEFAULT true",
]

app = FastAPI(title="cosmolet ui")

# --- БД: одно sync-соединение, переподключение при обрыве --------------------

_lock = threading.Lock()
_conn = None


def _run(fn):
    """Выполнить fn(conn) под локом; при обрыве соединения переподключиться и повторить один раз."""
    global _conn
    with _lock:
        last = None
        for attempt in (0, 1):
            try:
                if _conn is None or _conn.closed:
                    _conn = psycopg.connect(DB_URL, autocommit=True, row_factory=dict_row)
                    for sql in MIGRATIONS:          # идемпотентно, на каждое новое соединение
                        _conn.execute(sql)
                return fn(_conn)
            except psycopg.OperationalError as e:
                last = e
                try:
                    if _conn is not None:
                        _conn.close()
                except Exception:
                    pass
                _conn = None
        raise HTTPException(503, f"database unavailable: {last}")


def q(sql, params=()):
    return _run(lambda c: c.execute(sql, params).fetchall())


def q1(sql, params=()):
    return _run(lambda c: c.execute(sql, params).fetchone())


# --- Файлы: защита от traversal + отдача с Range -----------------------------

def safe_data_path(p, allow_abs=False):
    """Вернуть реальный путь файла, только если он внутри DATA_DIR.

    allow_abs=True дополнительно разрешает существующий абсолютный путь из БД
    (сэмплы голосов могут лежать вне /data)."""
    if not p:
        return None
    p = str(p)
    cands = []
    if os.path.isabs(p):
        cands.append(p)
    else:
        norm = p.replace("\\", "/").lstrip("/")
        cands.append(os.path.join(DATA_DIR, norm))
        if norm.startswith("data/"):  # в БД путь может быть записан как data/artifacts/...
            cands.append(os.path.join(DATA_DIR, norm[5:]))
    root = os.path.realpath(DATA_DIR)
    for c in cands:
        rp = os.path.realpath(c)
        if not os.path.isfile(rp):
            continue
        if rp == root or rp.startswith(root + os.sep):
            return rp
        if allow_abs and os.path.isabs(p):
            return rp
    return None


AUDIO_TYPES = {
    ".m4a": "audio/mp4", ".mp4": "audio/mp4", ".mp3": "audio/mpeg", ".mpga": "audio/mpeg",
    ".wav": "audio/wav", ".ogg": "audio/ogg", ".oga": "audio/ogg", ".opus": "audio/ogg",
    ".flac": "audio/flac", ".webm": "audio/webm",
}

_RANGE_RE = re.compile(r"bytes=(\d*)-(\d*)$")


def serve_range(path, request):
    """Отдать файл с поддержкой HTTP Range (перемотка в <audio>)."""
    size = os.path.getsize(path)
    ext = os.path.splitext(path)[1].lower()
    mt = AUDIO_TYPES.get(ext) or mimetypes.guess_type(path)[0] or "application/octet-stream"
    headers = {"Accept-Ranges": "bytes"}
    start, end, status = 0, size - 1, 200
    m = _RANGE_RE.match(request.headers.get("range", "") or "")
    if m and (m.group(1) or m.group(2)):
        if m.group(1):
            start = int(m.group(1))
            if m.group(2):
                end = min(int(m.group(2)), size - 1)
        else:  # bytes=-N — последние N байт
            start = max(size - int(m.group(2)), 0)
        if start >= size:
            return Response(status_code=416, headers={"Content-Range": f"bytes */{size}"})
        status = 206
        headers["Content-Range"] = f"bytes {start}-{end}/{size}"
    length = max(end - start + 1, 0)
    headers["Content-Length"] = str(length)

    def stream(s=start, n=length):
        with open(path, "rb") as f:
            f.seek(s)
            while n > 0:
                chunk = f.read(min(1 << 16, n))
                if not chunk:
                    break
                n -= len(chunk)
                yield chunk

    return StreamingResponse(stream(), status_code=status, headers=headers, media_type=mt)


# --- Эндпоинты ---------------------------------------------------------------

@app.get("/")
def index():
    # интерфейс меняется постоянно; закэшированная страница со старыми обработчиками
    # выглядит как «кнопки не работают» — поэтому явный запрет кэширования
    return FileResponse(os.path.join(BASE_DIR, "static", "index.html"),
                        headers={"Cache-Control": "no-store, must-revalidate"})


# этап скачивания записи из облака идёт ДО черновика — он и в конвейере первый
STAGE_ORDER = "CASE stage WHEN 'download' THEN 0 WHEN 'draft' THEN 1 WHEN 'quality' THEN 2 " \
              "WHEN 'resolve' THEN 3 WHEN 'store' THEN 4 ELSE 5 END"

# незавершённое скачивание: запись уже в базе, но файла ещё нет — это не поломка
DOWNLOADING_SQL = """SELECT recording_id, count(*) AS n FROM jobs
                      WHERE stage = 'download' AND status = 'running'
                      GROUP BY recording_id"""


def with_audio(rows, drop=True):
    """Добавить строкам has_audio: аудиофайл записи реально лежит на диске.

    В БД путь может остаться, а файл быть удалён (или ещё не скачан) — UI должен
    гасить кнопки воспроизведения, а не молча ничего не делать."""
    for r in rows:
        p = r.pop("audio_path", None) if drop else r.get("audio_path")
        r["has_audio"] = safe_data_path(p) is not None
    return rows


@app.get("/api/recordings")
def recordings():
    return with_audio(q(f"""
        SELECT r.id, r.filename, r.title, r.status, r.source, r.audio_path,
               r.duration_sec, r.size_bytes, r.started_at, r.created_at,
               coalesce(j.cost, 0)  AS cost_usd,
               coalesce(sg.n, 0)    AS segments,
               coalesce(cf.n, 0)    AS open_conflicts,
               coalesce(dl.n, 0) > 0 AS downloading
        FROM recordings r
        LEFT JOIN (SELECT recording_id, sum(cost_usd) AS cost FROM jobs GROUP BY recording_id) j
               ON j.recording_id = r.id
        LEFT JOIN (SELECT recording_id, count(*) AS n FROM segments GROUP BY recording_id) sg
               ON sg.recording_id = r.id
        LEFT JOIN (SELECT s.recording_id, count(*) AS n
                     FROM conflicts c JOIN segments s ON s.id = c.segment_id
                    WHERE c.status = 'open' GROUP BY s.recording_id) cf
               ON cf.recording_id = r.id
        LEFT JOIN ({DOWNLOADING_SQL}) dl
               ON dl.recording_id = r.id
        ORDER BY r.created_at DESC, r.id DESC"""))


@app.get("/api/recordings/summary")
def recordings_summary():
    """Список записей со сводкой для вкладки «Транскрипт».

    Агрегаты по спикерам приходят одним запросом (json_agg по сгруппированным
    сегментам), а не запросом на запись — список может быть длинным."""
    return with_audio(q(f"""
        SELECT r.id, r.filename, r.title, r.status, r.source, r.audio_path,
               r.duration_sec, r.size_bytes, r.started_at, r.created_at,
               coalesce(sg.n, 0)          AS segments,
               coalesce(sg.amb, 0)        AS ambiguous,
               coalesce(sg.speech_sec, 0) AS speech_sec,
               coalesce(j.cost, 0)        AS cost_usd,
               coalesce(cf.n, 0)          AS open_conflicts,
               coalesce(dl.n, 0) > 0      AS downloading,
               coalesce(sp.speakers, '[]'::json) AS speakers
        FROM recordings r
        LEFT JOIN (SELECT recording_id, count(*) AS n,
                          count(*) FILTER (WHERE ambiguous) AS amb,
                          sum(end_sec - start_sec) AS speech_sec
                     FROM segments GROUP BY recording_id) sg
               ON sg.recording_id = r.id
        LEFT JOIN (SELECT recording_id, sum(cost_usd) AS cost FROM jobs GROUP BY recording_id) j
               ON j.recording_id = r.id
        LEFT JOIN (SELECT s.recording_id, count(*) AS n
                     FROM conflicts c JOIN segments s ON s.id = c.segment_id
                    WHERE c.status = 'open' GROUP BY s.recording_id) cf
               ON cf.recording_id = r.id
        LEFT JOIN (SELECT recording_id,
                          json_agg(json_build_object('name', speaker_name, 'segments', n,
                                                     'sec', round(sec::numeric, 1))
                                   ORDER BY sec DESC) AS speakers
                     FROM (SELECT recording_id, speaker_name, count(*) AS n,
                                  sum(end_sec - start_sec) AS sec
                             FROM segments GROUP BY 1, 2) t
                    GROUP BY recording_id) sp
               ON sp.recording_id = r.id
        LEFT JOIN ({DOWNLOADING_SQL}) dl
               ON dl.recording_id = r.id
        ORDER BY r.created_at DESC, r.id DESC"""))


@app.get("/api/recordings/{rec_id}")
def recording_detail(rec_id: int):
    rec = q1("SELECT * FROM recordings WHERE id = %s", (rec_id,))
    if rec is None:
        raise HTTPException(404, "recording not found")
    with_audio([rec], drop=False)   # audio_path нужен в карточке, флаг — кнопкам
    rec["jobs"] = q(f"""
        SELECT id, stage, status, started_at, finished_at, cost_usd, error, artifact_path, meta
        FROM jobs WHERE recording_id = %s
        ORDER BY {STAGE_ORDER}, id""", (rec_id,))
    return rec


@app.get("/api/recordings/{rec_id}/segments")
def recording_segments(rec_id: int):
    return q("""
        SELECT id, start_sec, end_sec, text, speaker_name, confidence,
               inherited, ambiguous, detail, asr_logprob
        FROM segments WHERE recording_id = %s
        ORDER BY start_sec, id""", (rec_id,))


@app.get("/api/artifacts/{job_id}")
def artifact(job_id: int):
    row = q1("SELECT artifact_path FROM jobs WHERE id = %s", (job_id,))
    if row is None:
        raise HTTPException(404, "job not found")
    p = safe_data_path(row["artifact_path"])
    if p is None:
        raise HTTPException(404, "artifact file not found")
    return FileResponse(p, media_type="application/json")


@app.get("/api/audio/{rec_id}")
def audio(rec_id: int, request: Request):
    row = q1("SELECT audio_path FROM recordings WHERE id = %s", (rec_id,))
    if row is None:
        raise HTTPException(404, "recording not found")
    p = safe_data_path(row["audio_path"])
    if p is None:
        raise HTTPException(404, "audio file not found")
    return serve_range(p, request)


# --- голоса: эталоны, когерентность, матчинг --------------------------------

def load_samples(where: str = "", params=()):
    return q(f"""SELECT id, speaker_id, path, kind, duration_sec, source, is_active, added_at,
                        embedding
                 FROM speaker_samples {where} ORDER BY speaker_id, id""", params)


def decorate(rows):
    """Убрать вектора из выдачи, добавить coherence (внутри каждого человека) и has_path."""
    by_sp = {}
    for r in rows:
        by_sp.setdefault(r["speaker_id"], []).append(r)
    for items in by_sp.values():
        coh = voice.coherences([(r["id"], voice.parse_vec(r["embedding"]), r["is_active"])
                                for r in items])
        for r in items:
            r["coherence"] = coh.get(r["id"])
    for r in rows:
        r["has_embedding"] = r.pop("embedding") is not None
        r["has_path"] = bool(r["path"]) and safe_data_path(r["path"], allow_abs=True) is not None
    return rows


def stats(items):
    act = [r for r in items if r["is_active"]]
    cohs = [r["coherence"] for r in act if r["coherence"] is not None]
    return {
        "active": len(act),
        "inactive": len(items) - len(act),
        "avg_coherence": round(sum(cohs) / len(cohs), 4) if cohs else None,
    }


def known_refs() -> dict:
    """Эталон каждого человека — среднее нормированных векторов его АКТИВНЫХ сэмплов."""
    rows = q("""SELECT s.name, ss.embedding FROM speakers s
                JOIN speaker_samples ss ON ss.speaker_id = s.id
                WHERE ss.embedding IS NOT NULL AND ss.is_active""")
    acc = {}
    for r in rows:
        v = voice.parse_vec(r["embedding"])
        if v is not None:
            acc.setdefault(r["name"], []).append(v)
    return {n: voice.centroid(vs) for n, vs in acc.items()}


_SPAN_CACHE = {}


def span_vec(key, src, start, end):
    """Вектор куска записи с кэшем — экран разметки перезапрашивается часто."""
    k = (key, round(float(start), 2))
    v = _SPAN_CACHE.get(k)
    if v is None:
        dur = max(float(end) - float(start), voice.CLIP_MIN_SEC)
        v = voice.embed_span(Path(src), float(start), dur)
        if len(_SPAN_CACHE) > 400:
            _SPAN_CACHE.clear()
        _SPAN_CACHE[k] = v
    return v


def rec_audio(rec_id: int):
    row = q1("SELECT id, audio_path, title, filename FROM recordings WHERE id = %s", (rec_id,))
    if row is None:
        raise HTTPException(404, "recording not found")
    return row, safe_data_path(row["audio_path"])


@app.get("/api/speakers")
def speakers():
    people = q("SELECT id, name, aliases, created_at FROM speakers ORDER BY name")
    samples = decorate(load_samples())
    by_sp = {}
    for s in samples:
        by_sp.setdefault(s["speaker_id"], []).append(s)
    for p in people:
        ss = by_sp.get(p["id"], [])
        counts = {}
        for s in ss:
            counts[s["kind"]] = counts.get(s["kind"], 0) + 1
        p["sample_counts"] = counts
        p["samples"] = ss
        p.update(stats(ss))
    return people


@app.post("/api/speakers")
def speaker_create(body: dict):
    name = str((body or {}).get("name", "")).strip()
    if not name:
        raise HTTPException(400, "name is required")
    aliases = [str(a).strip() for a in (body or {}).get("aliases") or [] if str(a).strip()]
    row = q1("""INSERT INTO speakers (name, aliases) VALUES (%s, %s)
                ON CONFLICT (name) DO UPDATE SET aliases = EXCLUDED.aliases
                RETURNING id, name, aliases, created_at""", (name, aliases))
    return row


@app.get("/api/speakers/{sp_id}/detail")
def speaker_detail(sp_id: int):
    sp = q1("SELECT id, name, aliases, created_at FROM speakers WHERE id = %s", (sp_id,))
    if sp is None:
        raise HTTPException(404, "speaker not found")
    sp["samples"] = decorate(load_samples("WHERE speaker_id = %s", (sp_id,)))
    sp.update(stats(sp["samples"]))
    return sp


@app.post("/api/speakers/sample/{sample_id}/toggle")
def sample_toggle(sample_id: int):
    row = q1("""UPDATE speaker_samples SET is_active = NOT is_active
                WHERE id = %s RETURNING id, speaker_id, is_active""", (sample_id,))
    if row is None:
        raise HTTPException(404, "sample not found")
    items = decorate(load_samples("WHERE speaker_id = %s", (row["speaker_id"],)))
    return {"ok": True, "sample_id": sample_id, "is_active": row["is_active"],
            "speaker_id": row["speaker_id"],
            "coherence": {r["id"]: r["coherence"] for r in items}, **stats(items)}


@app.delete("/api/speakers/sample/{sample_id}")
def sample_delete(sample_id: int):
    row = q1("DELETE FROM speaker_samples WHERE id = %s RETURNING id, speaker_id, path",
             (sample_id,))
    if row is None:
        raise HTTPException(404, "sample not found")
    items = decorate(load_samples("WHERE speaker_id = %s", (row["speaker_id"],)))
    # аудиофайл на диске не трогаем — запись из базы убрана, клип при желании можно вернуть
    return {"ok": True, "sample_id": sample_id, "speaker_id": row["speaker_id"],
            "file_kept": row["path"],
            "coherence": {r["id"]: r["coherence"] for r in items}, **stats(items)}


# --- разметка: кандидаты, эталон из записи, переименование кластера ----------

ANON_RE = r"^(s|speaker)[_ -]?([0-9]+|\?)$"


@app.get("/api/enroll/candidates")
def enroll_candidates(recording_id: int):
    rec, src = rec_audio(recording_id)
    segs = q("""SELECT id, start_sec, end_sec, text, confidence, speaker_name, ambiguous, detail
                FROM segments WHERE recording_id = %s ORDER BY start_sec""", (recording_id,))
    names = {r["name"] for r in q("SELECT name FROM speakers")}
    refs = known_refs()
    groups = {}
    for s in segs:
        groups.setdefault(s["speaker_name"], []).append(s)
    out = []
    for label, items in groups.items():
        items.sort(key=lambda x: (x["end_sec"] - x["start_sec"]), reverse=True)
        # показываем много фрагментов: по пяти обрывкам голос не опознать.
        # reason важен человеку: top2_close («похож и на другого») и low_confidence —
        # это разные ситуации, и разбираются они по-разному
        top = []
        for s in items[:25]:
            d = s.get("detail") or {}
            if isinstance(d, str):
                try:
                    d = json.loads(d)
                except Exception:
                    d = {}
            top.append({k: s[k] for k in ("id", "start_sec", "end_sec", "text",
                                          "confidence", "ambiguous")}
                       | {"reason": d.get("reason")})
        # эталон и матч считаем по самому длинному НЕспорному фрагменту: спорный может
        # оказаться чужой репликой, попавшей в кластер по ошибке провайдера
        clean = [s for s in items if not s["ambiguous"]] or items
        best = clean[0]
        match, err = [], None
        if src and refs:
            try:
                v = span_vec(best["id"], src, best["start_sec"], best["end_sec"])
                match = voice.top_matches(v, refs, 3)
            except Exception as e:
                err = str(e)[:200]
        elif not src:
            err = "аудио записи недоступно"
        out.append({
            "label": label,
            "known": label in names,
            "total_sec": round(sum(s["end_sec"] - s["start_sec"] for s in items), 1),
            "count": len(items),
            "top": top,
            # именно этот фрагмент уйдёт в эталон при подтверждении (самый длинный чистый)
            "enroll_span": {"id": best["id"], "start_sec": best["start_sec"],
                            "end_sec": best["end_sec"]},
            "ambiguous_count": sum(1 for s in items if s["ambiguous"]),
            "match": match,
            "match_error": err,
        })
    out.sort(key=lambda g: g["total_sec"], reverse=True)
    return {"recording_id": recording_id, "title": rec["title"] or rec["filename"],
            "has_audio": src is not None, "groups": out}


@app.post("/api/enroll")
def enroll(body: dict):
    b = body or {}
    try:
        rec_id = int(b["recording_id"])
        start = float(b["start_sec"])
        end = float(b["end_sec"])
    except Exception:
        raise HTTPException(400, "recording_id, start_sec, end_sec are required")
    name = str(b.get("speaker_name", "")).strip()
    if not name:
        raise HTTPException(400, "speaker_name is required")
    dur = min(max(end - start, 0.0), voice.CLIP_MAX_SEC)
    if dur < voice.CLIP_MIN_SEC:
        raise HTTPException(400, "fragment is too short")
    rec, src = rec_audio(rec_id)
    if src is None:
        raise HTTPException(404, "audio file not found")

    sp = q1("SELECT id, name FROM speakers WHERE lower(name) = lower(%s)", (name,))
    if sp is None:
        if b.get("mode") != "new_speaker":
            raise HTTPException(404, f"speaker '{name}' not found")
        aliases = [str(a).strip() for a in b.get("aliases") or [] if str(a).strip()]
        sp = q1("""INSERT INTO speakers (name, aliases) VALUES (%s, %s)
                   ON CONFLICT (name) DO UPDATE SET aliases = EXCLUDED.aliases
                   RETURNING id, name""", (name, aliases))

    out_dir = Path(SPEAKERS_DIR) / voice.slug(sp["name"])
    out_dir.mkdir(parents=True, exist_ok=True)
    clip = out_dir / f"enroll_{int(time.time())}_{rec_id}.m4a"
    try:
        voice.cut(Path(src), start, dur, clip)
        vec = voice.embed_span(clip, 0, dur)
    except Exception as e:
        if clip.exists():
            clip.unlink()
        raise HTTPException(500, f"cut/embed failed: {str(e)[:300]}")

    row = q1("""INSERT INTO speaker_samples (speaker_id, path, kind, duration_sec, embedding, source)
                VALUES (%s, %s, 'embed', %s, %s::vector, %s) RETURNING id""",
             (sp["id"], str(clip), round(dur, 2), voice.vec_literal(vec),
              f"enroll:rec{rec_id}@{start:.1f}"))
    items = decorate(load_samples("WHERE speaker_id = %s", (sp["id"],)))
    new = next((r for r in items if r["id"] == row["id"]), None)
    return {
        "ok": True, "speaker_id": sp["id"], "speaker_name": sp["name"],
        "sample_id": row["id"], "path": str(clip),
        "samples_total": len(items),
        "coherence": new["coherence"] if new else None,
        "match": voice.top_matches(vec, known_refs(), 3),
        **stats(items),
    }


@app.post("/api/enroll/confirm_segment")
def confirm_segment(body: dict):
    """Подтвердить конкретную (обычно спорную) реплику: сделать её эталоном голоса.

    Спорные реплики — самый ценный материал: голос там звучит непривычно (эмоция,
    громкость, расстояние до микрофона), и именно они раздвигают похожих людей.
    Подтверждение снимает пометку спорности, закрывает конфликт и кладёт фрагмент
    в эталоны человека.
    """
    b = body or {}
    seg_id = b.get("segment_id")
    name = str(b.get("name", "")).strip()
    if seg_id is None or not name:
        raise HTTPException(400, "segment_id and name are required")
    seg = q1("""SELECT s.id, s.recording_id, s.start_sec, s.end_sec, r.audio_path
                FROM segments s JOIN recordings r ON r.id = s.recording_id
                WHERE s.id = %s""", (int(seg_id),))
    if seg is None:
        raise HTTPException(404, "segment not found")

    res = enroll({"recording_id": seg["recording_id"], "start_sec": seg["start_sec"],
                  "end_sec": seg["end_sec"], "speaker_name": name})

    def run(c):
        with c.transaction():
            c.execute("""
                UPDATE segments
                SET speaker_name = %s,
                    speaker_id = (SELECT id FROM speakers WHERE name = %s),
                    ambiguous = false,
                    detail = coalesce(detail, '{}'::jsonb)
                             || jsonb_build_object('resolution', %s::text)
                WHERE id = %s""", (name, name, f"user:{name} (подтверждён эталоном)", seg["id"]))
            c.execute("""UPDATE conflicts SET status='resolved', resolved_name=%s,
                         resolved_by='user', resolved_at=now()
                         WHERE segment_id = %s AND status='open'""", (name, seg["id"]))
    _run(run)
    return {**res, "segment_id": seg["id"], "confirmed": True}


@app.post("/api/enroll/rename_cluster")
def rename_cluster(body: dict):
    b = body or {}
    rec_id = b.get("recording_id")
    src_label = str(b.get("from_label", ""))
    to_name = str(b.get("to_name", "")).strip()
    if rec_id is None or not src_label or not to_name:
        raise HTTPException(400, "recording_id, from_label, to_name are required")
    ids = [r["id"] for r in q(
        "SELECT id FROM segments WHERE recording_id = %s AND speaker_name = %s",
        (int(rec_id), src_label))]
    if not ids:
        return {"ok": True, "updated": 0, "conflicts_closed": 0}
    sp = q1("SELECT id FROM speakers WHERE lower(name) = lower(%s)", (to_name,))

    def run(c):
        with c.transaction():
            c.execute("UPDATE segments SET speaker_name = %s, speaker_id = %s WHERE id = ANY(%s)",
                      (to_name, sp["id"] if sp else None, ids))
            cur = c.execute("""UPDATE conflicts SET status = 'resolved', resolved_name = %s,
                                      resolved_by = 'user', resolved_at = now()
                               WHERE status = 'open' AND segment_id = ANY(%s) RETURNING id""",
                            (to_name, ids))
            return len(cur.fetchall())

    closed = _run(run)
    return {"ok": True, "updated": len(ids), "conflicts_closed": closed,
            "speaker_id": sp["id"] if sp else None}


@app.get("/api/enroll/suggestions")
def suggestions():
    unnamed = q(f"""
        SELECT * FROM (
          SELECT DISTINCT ON (s.recording_id, s.speaker_name)
                 s.id, s.recording_id, s.start_sec, s.end_sec, s.text, s.speaker_name,
                 s.confidence, coalesce(r.title, r.filename) AS rec_title, r.audio_path
          FROM segments s JOIN recordings r ON r.id = s.recording_id
          WHERE s.end_sec - s.start_sec >= 6 AND s.speaker_name ~* '{ANON_RE}'
          ORDER BY s.recording_id, s.speaker_name, (s.end_sec - s.start_sec) DESC
        ) t ORDER BY (t.end_sec - t.start_sec) DESC LIMIT 30""")
    enrich = q("""
        SELECT * FROM (
          SELECT DISTINCT ON (s.recording_id, s.speaker_name)
                 s.id, s.recording_id, s.start_sec, s.end_sec, s.text, s.speaker_name,
                 s.confidence, coalesce(r.title, r.filename) AS rec_title, r.audio_path
          FROM segments s JOIN recordings r ON r.id = s.recording_id
          WHERE s.confidence BETWEEN 0.5 AND 0.75
            AND s.speaker_name IN (SELECT name FROM speakers)
            AND s.end_sec - s.start_sec >= 3
          ORDER BY s.recording_id, s.speaker_name, (s.end_sec - s.start_sec) DESC
        ) t ORDER BY t.confidence LIMIT 30""")
    samples = decorate(load_samples())
    names = {r["id"]: r["name"] for r in q("SELECT id, name FROM speakers")}
    bad = [{**s, "speaker_name": names.get(s["speaker_id"])}
           for s in samples
           if s["is_active"] and s["coherence"] is not None and s["coherence"] < 0.5]
    bad.sort(key=lambda s: s["coherence"])
    return {"unnamed": with_audio(unnamed), "enrich": with_audio(enrich), "bad_samples": bad}


@app.get("/api/speakers/sample/{sample_id}")
def speaker_sample(sample_id: int, request: Request):
    row = q1("SELECT path FROM speaker_samples WHERE id = %s", (sample_id,))
    if row is None:
        raise HTTPException(404, "sample not found")
    p = safe_data_path(row["path"], allow_abs=True)
    if p is None:
        raise HTTPException(404, "sample file not found")
    return serve_range(p, request)


@app.get("/api/conflicts")
def conflicts(status: str = "open"):
    return with_audio(q("""
        SELECT c.id, c.segment_id, c.reason, c.status, c.judge_verdict,
               c.resolved_name, c.resolved_by, c.created_at, c.resolved_at,
               s.recording_id, s.start_sec, s.end_sec, s.text,
               s.speaker_name, s.confidence, s.detail, r.audio_path
        FROM conflicts c JOIN segments s ON s.id = c.segment_id
                         JOIN recordings r ON r.id = s.recording_id
        WHERE c.status = %s
        ORDER BY c.id""", (status,)))


@app.post("/api/conflicts/{conflict_id}/resolve")
def resolve_conflict(conflict_id: int, body: dict):
    name = str((body or {}).get("name", "")).strip()
    if not name:
        raise HTTPException(400, "name is required")

    def run(c):
        with c.transaction():
            row = c.execute("SELECT segment_id FROM conflicts WHERE id = %s",
                            (conflict_id,)).fetchone()
            if row is None:
                return None
            # %s::text обязателен: без явного типа Postgres не выводит тип аргумента
            # jsonb_build_object и падает с IndeterminateDatatype
            c.execute("""
                UPDATE segments
                SET speaker_name = %s,
                    speaker_id = (SELECT id FROM speakers WHERE name = %s),
                    ambiguous = true,
                    detail = coalesce(detail, '{}'::jsonb)
                             || jsonb_build_object('resolution', %s::text)
                WHERE id = %s""", (name, name, "user:" + name, row["segment_id"]))
            c.execute("""
                UPDATE conflicts
                SET status = 'resolved', resolved_name = %s, resolved_by = 'user',
                    resolved_at = now()
                WHERE id = %s""", (name, conflict_id))
            return row

    row = _run(run)
    if row is None:
        raise HTTPException(404, "conflict not found")
    return {"ok": True, "conflict_id": conflict_id, "segment_id": row["segment_id"], "name": name}


@app.get("/api/disk")
def disk():
    files = []
    if os.path.isdir(DATA_DIR):
        for dirpath, _dirs, names in os.walk(DATA_DIR):
            for n in names:
                fp = os.path.join(dirpath, n)
                try:
                    files.append((fp, os.path.getsize(fp)))
                except OSError:
                    pass
    folders = {name: 0 for name in DATA_FOLDERS}
    data_total = 0
    for fp, sz in files:
        data_total += sz
        top = os.path.relpath(fp, DATA_DIR).split(os.sep)[0]
        if top in folders:
            folders[top] += sz
    row = q1("SELECT pg_database_size(current_database()) AS b")
    db_bytes = int(row["b"]) if row else 0

    def mb(b):
        return round(b / 1048576, 1)

    top15 = sorted(files, key=lambda t: t[1], reverse=True)[:15]
    return {
        "folders": {k: mb(v) for k, v in folders.items()},
        "data_mb": mb(data_total),
        "db_mb": mb(db_bytes),
        "total_mb": mb(data_total + db_bytes),
        "top_files": [{"path": fp, "mb": mb(sz)} for fp, sz in top15],
    }


@app.post("/api/recordings/{rec_id}/reprocess")
def reprocess(rec_id: int, body: dict):
    stage = (body or {}).get("from_stage")
    if stage not in ("draft", "quality", "resolve"):
        raise HTTPException(400, "from_stage must be draft|quality|resolve")

    def run(c):
        return c.execute("""
            UPDATE recordings
            SET meta = meta || jsonb_build_object('reprocess_from', %s), status = 'new'
            WHERE id = %s RETURNING id""", (stage, rec_id)).fetchone()

    if _run(run) is None:
        raise HTTPException(404, "recording not found")
    return {"ok": True, "id": rec_id, "from_stage": stage}


@app.get("/api/costs/today")
def costs_today():
    row = q1("""
        SELECT coalesce(sum(usd) FILTER (WHERE day = current_date), 0) AS today,
               coalesce(sum(usd), 0) AS total
        FROM costs""")
    return {"today_usd": float(row["today"]), "total_usd": float(row["total"])}


def _f(v, default=0.0):
    """numeric из Postgres приезжает Decimal — считать производные метрики в float."""
    return default if v is None else float(v)


@app.get("/api/usage")
def usage():
    """Куда уходят деньги: итоги, разбивка по моделям и записям, динамика по дням."""
    tot = q1("""
        SELECT coalesce(sum(usd) FILTER (WHERE day = current_date), 0)      AS today,
               coalesce(sum(usd) FILTER (WHERE day >= current_date - 6), 0) AS week,
               coalesce(sum(usd), 0)                                        AS total,
               count(*)                                                     AS calls,
               min(day)                                                     AS first_day
        FROM costs""")

    # знаменатели для «$ за час»: для черновика — вся длительность записи,
    # для качества — только речевые регионы, размеченные черновиком (jobs.meta.speech_sec)
    models = q("""
        WITH c AS (SELECT model, kind, recording_id, sum(usd) AS usd, count(*) AS calls
                     FROM costs GROUP BY 1, 2, 3),
             d AS (SELECT r.id, r.duration_sec::numeric AS audio_sec,
                          (SELECT max(nullif(j.meta->>'speech_sec', '')::numeric)
                             FROM jobs j
                            WHERE j.recording_id = r.id AND j.stage = 'draft') AS speech_sec
                     FROM recordings r)
        SELECT c.model, c.kind,
               sum(c.usd)                     AS usd,
               sum(c.calls)                   AS calls,
               count(DISTINCT c.recording_id) AS recordings,
               sum(d.audio_sec)               AS audio_sec,
               sum(d.speech_sec)              AS speech_sec
          FROM c LEFT JOIN d ON d.id = c.recording_id
         GROUP BY 1, 2
         ORDER BY sum(c.usd) DESC""")

    by_model = []
    for m in models:
        usd = _f(m["usd"])
        audio_h = _f(m["audio_sec"]) / 3600
        speech_h = _f(m["speech_sec"]) / 3600
        base = {"draft": audio_h, "quality": speech_h}.get(m["kind"], 0.0)
        by_model.append({
            "model": m["model"], "kind": m["kind"],
            "usd": round(usd, 6),
            "calls": int(m["calls"] or 0),
            "recordings": int(m["recordings"] or 0),
            "audio_hours": round(audio_h, 3) if audio_h else None,
            "speech_hours": round(speech_h, 3) if speech_h else None,
            # null, если знаменателя нет (нет длительностей / нет карты речи) — не выдумываем
            "usd_per_audio_hour": round(usd / base, 4) if base > 0 else None,
            "basis": {"draft": "запись", "quality": "речь"}.get(m["kind"]),
        })

    recs = q("""
        SELECT r.id, coalesce(r.title, r.filename) AS title, r.filename,
               r.duration_sec, r.started_at, r.created_at,
               coalesce(sum(c.usd) FILTER (WHERE c.kind = 'draft'), 0)   AS draft_usd,
               coalesce(sum(c.usd) FILTER (WHERE c.kind = 'quality'), 0) AS quality_usd,
               coalesce(sum(c.usd) FILTER (WHERE c.kind = 'judge'), 0)   AS judge_usd,
               coalesce(sum(c.usd), 0)                                   AS total_usd,
               count(c.id)                                               AS calls
          FROM recordings r JOIN costs c ON c.recording_id = r.id
         GROUP BY r.id
         ORDER BY sum(c.usd) DESC, r.id DESC""")

    by_recording = []
    for r in recs:
        total = _f(r["total_usd"])
        hours = _f(r["duration_sec"]) / 3600
        by_recording.append({
            "id": r["id"], "title": r["title"], "filename": r["filename"],
            "duration_sec": _f(r["duration_sec"], None),
            "started_at": r["started_at"], "created_at": r["created_at"],
            "draft_usd": round(_f(r["draft_usd"]), 6),
            "quality_usd": round(_f(r["quality_usd"]), 6),
            "judge_usd": round(_f(r["judge_usd"]), 6),
            "total_usd": round(total, 6),
            "calls": int(r["calls"] or 0),
            "usd_per_hour": round(total / hours, 4) if hours > 0 else None,
        })

    daily = q("""
        SELECT to_char(g.d, 'YYYY-MM-DD') AS day,
               coalesce(sum(c.usd), 0) AS usd, count(c.id) AS calls
          FROM generate_series((current_date - 13)::timestamptz, current_date::timestamptz,
                               interval '1 day') g(d)
          LEFT JOIN costs c ON c.day = g.d::date
         GROUP BY g.d ORDER BY g.d""")

    today = _f(tot["today"])
    return {
        "budget_usd": DAILY_BUDGET_USD,
        "today_usd": round(today, 6),
        "week_usd": round(_f(tot["week"]), 6),
        "total_usd": round(_f(tot["total"]), 6),
        "left_today_usd": round(DAILY_BUDGET_USD - today, 6),
        "calls": int(tot["calls"] or 0),
        "first_day": tot["first_day"],
        "by_model": by_model,
        "by_recording": by_recording,
        "daily": [{"day": d["day"], "usd": round(_f(d["usd"]), 6), "calls": int(d["calls"] or 0)}
                  for d in daily],
    }
