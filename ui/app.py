"""cosmolet UI — FastAPI поверх Postgres: статика, API, аудио с HTTP Range."""
import mimetypes
import os
import re
import threading

import psycopg
from psycopg.rows import dict_row
from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import FileResponse, Response, StreamingResponse

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.environ.get("DATA_DIR", "/data")
DB_URL = os.environ.get("DATABASE_URL", "postgresql://cosmolet:cosmolet@localhost:5432/cosmolet")
DATA_FOLDERS = ["inbox", "in_progress", "done", "failed", "artifacts", "audio"]

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
    return FileResponse(os.path.join(BASE_DIR, "static", "index.html"))


@app.get("/api/recordings")
def recordings():
    return q("""
        SELECT r.id, r.filename, r.status, r.duration_sec, r.size_bytes, r.created_at,
               coalesce(j.cost, 0)  AS cost_usd,
               coalesce(sg.n, 0)    AS segments,
               coalesce(cf.n, 0)    AS open_conflicts
        FROM recordings r
        LEFT JOIN (SELECT recording_id, sum(cost_usd) AS cost FROM jobs GROUP BY recording_id) j
               ON j.recording_id = r.id
        LEFT JOIN (SELECT recording_id, count(*) AS n FROM segments GROUP BY recording_id) sg
               ON sg.recording_id = r.id
        LEFT JOIN (SELECT s.recording_id, count(*) AS n
                     FROM conflicts c JOIN segments s ON s.id = c.segment_id
                    WHERE c.status = 'open' GROUP BY s.recording_id) cf
               ON cf.recording_id = r.id
        ORDER BY r.created_at DESC, r.id DESC""")


@app.get("/api/recordings/{rec_id}")
def recording_detail(rec_id: int):
    rec = q1("SELECT * FROM recordings WHERE id = %s", (rec_id,))
    if rec is None:
        raise HTTPException(404, "recording not found")
    rec["jobs"] = q("""
        SELECT id, stage, status, started_at, finished_at, cost_usd, error, artifact_path, meta
        FROM jobs WHERE recording_id = %s
        ORDER BY CASE stage WHEN 'draft' THEN 1 WHEN 'quality' THEN 2
                            WHEN 'resolve' THEN 3 WHEN 'store' THEN 4 ELSE 5 END, id""", (rec_id,))
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


@app.get("/api/speakers")
def speakers():
    people = q("SELECT id, name, aliases, created_at FROM speakers ORDER BY name")
    samples = q("""
        SELECT id, speaker_id, kind, duration_sec, source, added_at,
               (path IS NOT NULL) AS has_path
        FROM speaker_samples ORDER BY id""")
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
    return people


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
    return q("""
        SELECT c.id, c.segment_id, c.reason, c.status, c.judge_verdict,
               c.resolved_name, c.resolved_by, c.created_at, c.resolved_at,
               s.recording_id, s.start_sec, s.end_sec, s.text,
               s.speaker_name, s.confidence, s.detail
        FROM conflicts c JOIN segments s ON s.id = c.segment_id
        WHERE c.status = %s
        ORDER BY c.id""", (status,))


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
            c.execute("""
                UPDATE segments
                SET speaker_name = %s, ambiguous = true,
                    detail = coalesce(detail, '{}'::jsonb) || jsonb_build_object('resolution', %s)
                WHERE id = %s""", (name, "user:" + name, row["segment_id"]))
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
