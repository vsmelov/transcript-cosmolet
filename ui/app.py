"""cosmolet UI — FastAPI поверх Postgres: статика, API, аудио с HTTP Range, разметка голосов."""
import json
import mimetypes
import os
import re
import threading
import time
from pathlib import Path

import numpy as np
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

# --- ручной разрез реплики (POST /api/segments/{id}/split) --------------------
# Пороги те же по смыслу, что у автосплиттера воркера (worker/diarize.py): режем,
# только если голос по обе стороны реально разный и в каждой половине есть что слушать.
SPLIT_MAX_COS = 0.60        # минимум косинуса выше порога — смены говорящего нет
SPLIT_MIN_SIDE_SEC = 2.5    # меньше речи в половине — вектор половины недостоверен
SPLIT_PROBES = 9            # узлов на проход (грубый + уточняющий): каждый узел = 2 эмбеддинга

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


def as_dict(d):
    """detail из jsonb: psycopg отдаёт dict, но в старых записях мог остаться текст."""
    if isinstance(d, str):
        try:
            d = json.loads(d)
        except Exception:
            return {}
    return d if isinstance(d, dict) else {}


def base_matrix():
    """Центроиды известных голосов: (имена, матрица). Для матча фрагментов на лету."""
    acc = {}
    for r in q("""SELECT sp.name, ss.embedding FROM speakers sp
                    JOIN speaker_samples ss ON ss.speaker_id = sp.id
                   WHERE ss.embedding IS NOT NULL AND ss.is_active"""):
        v = np.array(json.loads(r["embedding"]) if isinstance(r["embedding"], str) else r["embedding"],
                     dtype=np.float64)
        acc.setdefault(r["name"], []).append(v / (np.linalg.norm(v) or 1))
    names = sorted(acc)
    if not names:
        return [], None
    M = np.array([np.mean(acc[n], axis=0) / (np.linalg.norm(np.mean(acc[n], axis=0)) or 1)
                  for n in names])
    return names, M


def live_top(vec, names, M, k=3):
    """Кандидаты для одного фрагмента, посчитанные сейчас.

    В базе список кандидатов лежит ТОЛЬКО у спорных реплик — так решено, чтобы не
    раздувать хранилище. Но человеку он нужен у любой: реплика может быть уверенно
    своей внутри безымянного кластера, и тогда без кандидатов непонятно, кому её
    отдавать. Считается это мгновенно — матрица из десятка центроидов.
    """
    if M is None or vec is None:
        return []
    v = np.array(vec, dtype=np.float64)
    n = np.linalg.norm(v)
    if not n:
        return []
    sim = M @ (v / n)
    return [{"name": names[i], "cos": round(float(sim[i]), 3)} for i in np.argsort(-sim)[:k]]


def utt(r, has_audio=False, **extra):
    """Единый формат ФРАЗЫ для всех экранов UI (компонент «Фраза»).

    Один и тот же объект приезжает из транскрипта, разметки, предложений и конфликтов,
    поэтому UI рисует их одной функцией и не расходится в наборе кнопок и полей.
    """
    d = as_dict(r.get("detail"))
    out = {
        "id": int(r["id"]),
        "recording_id": r.get("recording_id"),
        "start_sec": float(r["start_sec"]),
        "end_sec": float(r["end_sec"]),
        "text": r.get("text") or "",
        "speaker_name": r.get("speaker_name"),
        "confidence": None if r.get("confidence") is None else float(r["confidence"]),
        "inherited": bool(r.get("inherited")),
        "ambiguous": bool(r.get("ambiguous")),
        # почему спорная: из открытого конфликта, иначе из detail черновика
        "reason": r.get("reason") or d.get("reason"),
        "detail": d,
        "has_audio": bool(has_audio),
    }
    out.update(extra)
    return out


# колонки сегмента, из которых собирается фраза
SEG_COLS = """s.id, s.recording_id, s.start_sec, s.end_sec, s.text, s.speaker_name,
              s.confidence, s.inherited, s.ambiguous, s.detail, s.asr_logprob"""


def segments_by_ids(ids):
    rows = q(f"""SELECT {SEG_COLS}, r.audio_path FROM segments s
                 JOIN recordings r ON r.id = s.recording_id
                 WHERE s.id = ANY(%s) ORDER BY s.start_sec, s.id""", (list(ids),))
    return [utt(r, safe_data_path(r["audio_path"]) is not None,
                asr_logprob=r["asr_logprob"]) for r in rows]


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
        ORDER BY coalesce(r.started_at, r.created_at) DESC, r.id DESC"""))


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
        ORDER BY coalesce(r.started_at, r.created_at) DESC, r.id DESC"""))


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
    """Лента транскрипта — список фраз в едином формате.

    Открытый конфликт подтягивается сразу (LATERAL, чтобы сегмент не задвоился при
    нескольких конфликтах): UI рисует бейдж «спорный» с причиной без второго запроса.
    """
    row = q1("SELECT audio_path FROM recordings WHERE id = %s", (rec_id,))
    if row is None:
        raise HTTPException(404, "recording not found")
    has = safe_data_path(row["audio_path"]) is not None
    rows = q(f"""
        SELECT {SEG_COLS}, c.id AS conflict_id, c.reason
        FROM segments s
        LEFT JOIN LATERAL (SELECT id, reason FROM conflicts
                            WHERE segment_id = s.id AND status = 'open'
                            ORDER BY id LIMIT 1) c ON true
        WHERE s.recording_id = %s
        ORDER BY s.start_sec, s.id""", (rec_id,))
    return [utt(r, has, asr_logprob=r["asr_logprob"], conflict_id=r["conflict_id"]) for r in rows]


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
    people = q("SELECT id, name, aliases, about, created_at FROM speakers ORDER BY name")
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


@app.get("/api/recordings/{rec_id}/join_scan")
def join_scan(rec_id: int):
    """Что получится при разных порогах склейки голосов — таблица для выбора глазами.

    Считается по сохранённым векторам сегментов: ни аудио, ни платных вызовов, поэтому
    можно прогнать весь диапазон разом. Единого правильного порога не существует —
    он зависит от акустики и от того, насколько похожи голоса в этой записи.
    """
    rows = q("""SELECT start_sec, end_sec, embedding FROM segments
                 WHERE recording_id = %s AND embedding IS NOT NULL
                 ORDER BY start_sec""", (rec_id,))
    if not rows:
        return {"scan": [], "note": "у записи нет векторов — этап опознания ещё не проходил"}
    V = np.array([json.loads(r["embedding"]) if isinstance(r["embedding"], str) else r["embedding"]
                  for r in rows], dtype=np.float64)
    V /= np.linalg.norm(V, axis=1, keepdims=True)
    dur = np.array([float(r["end_sec"]) - float(r["start_sec"]) for r in rows])
    anchors = np.where(dur >= 3.0)[0]

    base = {}
    for r in q("""SELECT sp.name, ss.embedding FROM speakers sp
                    JOIN speaker_samples ss ON ss.speaker_id = sp.id
                   WHERE ss.embedding IS NOT NULL AND ss.is_active"""):
        v = np.array(json.loads(r["embedding"]) if isinstance(r["embedding"], str) else r["embedding"])
        base.setdefault(r["name"], []).append(v / (np.linalg.norm(v) or 1))
    names = sorted(base)
    B = np.array([np.mean(base[n], axis=0) / (np.linalg.norm(np.mean(base[n], axis=0)) or 1)
                  for n in names]) if names else np.zeros((0, V.shape[1]))

    out = []
    for k in range(9):
        th = round(0.50 + k * 0.025, 3)
        groups = [[int(i)] for i in anchors]
        cents = [V[i].copy() for i in anchors]
        while len(groups) > 1:
            C = np.array(cents)
            M = C @ C.T
            np.fill_diagonal(M, -1.0)
            i, j = np.unravel_index(int(M.argmax()), M.shape)
            if M[i, j] < th:
                break
            groups[i] += groups[j]
            m = np.mean([cents[i], cents[j]], axis=0)
            cents[i] = m / (np.linalg.norm(m) or 1)
            groups.pop(j); cents.pop(j)
        people, named = [], 0.0
        for g, c in sorted(zip(groups, cents), key=lambda gc: -dur[gc[0]].sum()):
            sec = float(dur[g].sum())
            if sec < 30:
                continue
            if len(B):
                sim = B @ c
                o = np.argsort(-sim)
                gap = float(sim[o[0]] - (sim[o[1]] if len(o) > 1 else 0))
                sure = sim[o[0]] >= 0.70 and gap >= 0.15
                if sure:
                    named += sec
                people.append({"minutes": round(sec / 60, 1), "sure": bool(sure),
                               "name": names[o[0]], "cos": round(float(sim[o[0]]), 2)})
            else:
                people.append({"minutes": round(sec / 60, 1), "sure": False,
                               "name": "?", "cos": 0.0})
        out.append({"join": th, "clusters": len(people),
                    "named_min": round(named / 60, 1), "people": people[:10]})
    return {"scan": out, "utterances": len(rows), "anchors": int(len(anchors))}


@app.post("/api/recordings/{rec_id}/join")
def set_join(rec_id: int, body: dict):
    """Зафиксировать порог склейки и переиграть опознание с ним (бесплатно, локально)."""
    join = (body or {}).get("join")
    def run(c):
        if join is None:
            c.execute("UPDATE recordings SET meta = meta - 'join' WHERE id=%s", (rec_id,))
        else:
            c.execute("""UPDATE recordings
                         SET meta = meta || jsonb_build_object('join', %s::float)
                         WHERE id=%s""", (float(join), rec_id))
        return c.execute("UPDATE recordings SET status='transcribed' WHERE id=%s RETURNING id",
                         (rec_id,)).fetchone()
    if _run(run) is None:
        raise HTTPException(404, "recording not found")
    return {"ok": True, "id": rec_id, "join": join}


@app.post("/api/speakers/{sp_id}/about")
def speaker_about(sp_id: int, body: dict):
    """Свободная заметка о человеке: контакты в мессенджерах, почты, контекст.

    Держим одним текстовым полем без схемы намеренно: это опора для связывания
    спикера с ним же в других источниках, а формат таких пометок заранее не угадать.
    """
    row = q1("UPDATE speakers SET about = %s WHERE id = %s RETURNING id, about",
             (str((body or {}).get("about", ""))[:4000], sp_id))
    if row is None:
        raise HTTPException(404, "speaker not found")
    return row


@app.get("/api/speakers/{sp_id}/detail")
def speaker_detail(sp_id: int):
    sp = q1("SELECT id, name, aliases, about, created_at FROM speakers WHERE id = %s", (sp_id,))
    if sp is None:
        raise HTTPException(404, "speaker not found")
    sp["samples"] = decorate(load_samples("WHERE speaker_id = %s", (sp_id,)))
    sp.update(stats(sp["samples"]))
    return sp


@app.get("/api/speakers/unknown")
def unknown_speakers(min_sec: float = 60.0):
    """Кто часто говорит, но до сих пор не заведён в базе.

    Неопознанные метки (S1, S2, ...) внутри одной записи — это уже кластеры голосов.
    Здесь мы склеиваем их МЕЖДУ записями по косинусу центроидов: один и тот же
    человек, встреченный в пяти разговорах, должен предлагаться один раз, а не пять.
    Заодно показываем ближайшего известного — часто это не новый человек, а промах
    опознания, и правильное действие тогда «добавить эталон», а не «завести людей».
    """
    rows = q("""
        SELECT s.recording_id, s.speaker_name, s.embedding, s.id AS seg_id,
               s.start_sec, s.end_sec, s.text, r.title
          FROM segments s JOIN recordings r ON r.id = s.recording_id
         WHERE s.embedding IS NOT NULL AND s.speaker_name ~ '^S[0-9?]'
         ORDER BY s.recording_id, s.speaker_name, (s.end_sec - s.start_sec) DESC""")
    if not rows:
        return {"groups": []}

    def vec(e):
        v = np.array(json.loads(e) if isinstance(e, str) else e, dtype=np.float64)
        n = np.linalg.norm(v)
        return v / n if n else v

    # шаг 1: метка внутри записи -> центроид и самая длинная реплика как образец
    labels = {}
    for r in rows:
        key = (r["recording_id"], r["speaker_name"])
        g = labels.setdefault(key, {"vecs": [], "sec": 0.0, "n": 0, "best": r,
                                    "title": r["title"], "rec": r["recording_id"]})
        g["vecs"].append(vec(r["embedding"]))
        g["sec"] += float(r["end_sec"]) - float(r["start_sec"])
        g["n"] += 1
    for g in labels.values():
        m = np.mean(g["vecs"], axis=0)
        n = np.linalg.norm(m)
        g["c"] = m / n if n else m

    # шаг 2: жадная склейка меток между записями (порог тот же, что у воркера)
    JOIN = 0.70
    groups = []
    for key, g in sorted(labels.items(), key=lambda kv: -kv[1]["sec"]):
        for gr in groups:
            if float(np.dot(gr["c"], g["c"])) >= JOIN:
                gr["members"].append(g)
                gr["sec"] += g["sec"]
                gr["n"] += g["n"]
                w = np.mean([m["c"] for m in gr["members"]], axis=0)
                gr["c"] = w / (np.linalg.norm(w) or 1)
                break
        else:
            groups.append({"c": g["c"], "members": [g], "sec": g["sec"], "n": g["n"]})

    known = {}
    for r in q("""SELECT sp.name, ss.embedding FROM speakers sp
                    JOIN speaker_samples ss ON ss.speaker_id = sp.id
                   WHERE ss.embedding IS NOT NULL AND ss.is_active"""):
        known.setdefault(r["name"], []).append(vec(r["embedding"]))
    base = {}
    for name, vs in known.items():
        m = np.mean(vs, axis=0)
        base[name] = m / (np.linalg.norm(m) or 1)

    out = []
    for gr in sorted(groups, key=lambda g: -g["sec"]):
        if gr["sec"] < min_sec:
            continue
        near = sorted(((float(np.dot(gr["c"], v)), n) for n, v in base.items()), reverse=True)
        best = max(gr["members"], key=lambda m: m["sec"])["best"]
        out.append({
            "labels": [f'#{m["rec"]} {m["best"]["speaker_name"]}' for m in gr["members"]],
            "recordings": sorted({m["title"] for m in gr["members"]}),
            "rec_count": len({m["rec"] for m in gr["members"]}),
            "segments": gr["n"], "minutes": round(gr["sec"] / 60, 1),
            "sample": {"segment_id": best["seg_id"], "recording_id": best["recording_id"],
                       "start": round(float(best["start_sec"]), 2),
                       "end": round(float(best["end_sec"]), 2),
                       "text": (best["text"] or "")[:160]},
            # близко к известному — скорее промах опознания, чем новый человек
            "closest": [{"name": n, "cos": round(c, 2)} for c, n in near[:3]],
            "likely_known": bool(near and near[0][0] >= 0.62),
        })
    return {"groups": out}


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


# Глобальные персоны собираются ДВУХУРОВНЕВО: сначала кластеры внутри записей
# (их считает воркер), затем склейка их центроидов между записями. Прямая
# кластеризация 6485 отдельных фрагментов проверена и отвергнута: 20 минут счёта
# и 1113 раздробленных групп, крупнейшая из которых — химера из нескольких
# голосов с матчем 0.66. Склейка центроидов даёт то же самое за пару секунд и
# чище: усреднённый вектор кластера намного устойчивее одиночной реплики.
GLOBAL_JOIN = 0.80        # порог склейки центроидов; на 0.62 слипались разные люди
# Насколько фрагмент должен быть похож на голос группы, чтобы получить её имя.
# Группа наследует примесь из локальных кластеров: её аутсайдеры с типичностью
# около 0.1 — уже чужой голос, и тащить их за компанию нельзя. Такие фрагменты
# остаются безымянными: честное «не знаем» лучше уверенно неверного имени.
ASSIGN_MIN_TYPICALITY = 0.35


@app.get("/api/global/persons")
def global_persons(only_unnamed: bool = True, top: int = 10, min_typ: float = ASSIGN_MIN_TYPICALITY):
    """Голоса, сведённые по ВСЕМУ архиву: один человек — одна строка, а не по строке
    на каждую запись, где он говорил.

    У каждой группы отдаётся ядро (самые типичные фрагменты) и аутсайдеры (самые
    далёкие от центра): по ним человек за минуту проверяет, один ли это голос,
    прежде чем подтвердить группу целиком во всех записях сразу.
    """
    where = "speaker_name ~ '^S[0-9?]'" if only_unnamed else "speaker_name <> '[noise]'"
    rows = q(f"""SELECT recording_id, speaker_name, count(*) AS n,
                        sum(end_sec - start_sec) AS sec, avg(embedding)::text AS cent
                   FROM segments
                  WHERE embedding IS NOT NULL AND end_sec - start_sec >= 2 AND {where}
                  GROUP BY 1, 2 HAVING sum(end_sec - start_sec) >= 20""")
    if not rows:
        return {"persons": [], "total_minutes": 0, "note": "нечего группировать"}
    cents, meta = [], []
    for r in rows:
        v = np.array(json.loads(r["cent"]), dtype=np.float64)
        cents.append(v / (np.linalg.norm(v) or 1))
        meta.append(r)

    groups = [[i] for i in range(len(cents))]
    cur = [c.copy() for c in cents]
    while len(groups) > 1:
        M = np.array(cur) @ np.array(cur).T
        np.fill_diagonal(M, -1.0)
        i, j = np.unravel_index(int(M.argmax()), M.shape)
        if M[i, j] < GLOBAL_JOIN:
            break
        groups[i] += groups[j]
        m = np.mean([cur[i], cur[j]], axis=0)
        cur[i] = m / (np.linalg.norm(m) or 1)
        groups.pop(j)
        cur.pop(j)

    base = {}
    for r in q("""SELECT sp.name, ss.embedding FROM speakers sp
                    JOIN speaker_samples ss ON ss.speaker_id = sp.id
                   WHERE ss.embedding IS NOT NULL AND ss.is_active"""):
        v = np.array(json.loads(r["embedding"]) if isinstance(r["embedding"], str) else r["embedding"])
        base.setdefault(r["name"], []).append(v / (np.linalg.norm(v) or 1))
    names = sorted(base)
    B = (np.array([np.mean(base[n], axis=0) / (np.linalg.norm(np.mean(base[n], axis=0)) or 1)
                   for n in names]) if names else None)
    gnames, gM = names, B

    out = []
    for g, c in sorted(zip(groups, cur), key=lambda gc: -sum(float(meta[i]["sec"]) for i in gc[0])):
        parts = [{"recording_id": meta[i]["recording_id"], "label": meta[i]["speaker_name"],
                  "minutes": round(float(meta[i]["sec"]) / 60, 1)} for i in g]
        sec = sum(float(meta[i]["sec"]) for i in g)

        # фрагменты всей группы, отсортированные по близости к её общему центру
        conds = " OR ".join(["(recording_id = %s AND speaker_name = %s)"] * len(parts))
        args = [x for p in parts for x in (p["recording_id"], p["label"])]
        # полный набор полей: фрагменты рисуются тем же компонентом «Фраза», что и
        # везде, — с текстом, плеером и кнопками поштучного переназначения
        frags = q(f"""SELECT {SEG_COLS}, s.embedding FROM segments s
                       WHERE s.embedding IS NOT NULL
                         AND s.end_sec - s.start_sec >= 1.2 AND ({conds})""", tuple(args))
        core, outliers, fit, fit_sec = [], [], 0, 0.0
        if frags:
            M = np.array([json.loads(f["embedding"]) if isinstance(f["embedding"], str)
                          else f["embedding"] for f in frags], dtype=np.float64)
            M /= np.linalg.norm(M, axis=1, keepdims=True)
            sims = M @ c
            # Показываем ТОЛЬКО те фрагменты, которые реально получат имя. Смотреть
            # на отсеянных бессмысленно: решение по ним всё равно не принимается.
            # Поэтому «аутсайдеры» здесь — худшие ИЗ ПРОХОДЯЩИХ, то есть то самое
            # пограничное, ради чего человек и проверяет группу перед подтверждением.
            keep = [i for i in range(len(frags)) if sims[i] >= min_typ]
            keep.sort(key=lambda i: -sims[i])
            fit = len(keep)
            fit_sec = float(sum(float(frags[i]["end_sec"]) - float(frags[i]["start_sec"])
                                for i in keep))

            def fmt(i):
                return utt(frags[i], True, typicality=round(float(sims[i]), 3),
                           live_top=live_top(M[i], gnames, gM))

            core = [fmt(i) for i in keep[:top]]
            outliers = [fmt(i) for i in keep[-top:]][::-1] if len(keep) > top else []

        match = []
        if B is not None:
            sim = B @ c
            match = [{"name": names[k], "cos": round(float(sim[k]), 2)} for k in np.argsort(-sim)[:3]]
        out.append({"minutes": round(sec / 60, 1), "fragments": sum(int(meta[i]["n"]) for i in g),
                    "recordings": len({p["recording_id"] for p in parts}), "parts": parts,
                    "match": match, "core": core, "outliers": outliers,
                    # сколько фрагментов достаточно похожи, чтобы получить имя
                    "fit": fit, "fit_minutes": round(fit_sec / 60, 1),
                    "checked": len(frags), "min_typicality": min_typ,
                    "centroid": [round(float(x), 6) for x in c]})
    total = sum(p["minutes"] for p in out) or 1
    acc = 0.0
    for p in out:
        acc += p["minutes"]
        p["covers_pct"] = round(acc / total * 100)
    return {"persons": out, "total_minutes": round(total, 1)}


@app.post("/api/global/assign")
def global_assign(body: dict):
    """Отдать человеку целую глобальную группу — все её кластеры во всех записях.

    В эталоны идёт по одной самой типичной реплике ИЗ КАЖДОЙ записи, а не все
    подряд: ценность базы голосов в разнообразии акустики, а десяток реплик из
    одного разговора — по сути один эталон, который ещё и перетягивает центроид.
    """
    b = body or {}
    name = str(b.get("name", "")).strip()
    parts = b.get("parts") or []
    if not name or not parts:
        raise HTTPException(400, "name and parts are required")
    enroll_core = bool(b.get("enroll", True)) and name != "[noise]"
    # Центроид группы приходит с экрана — по нему отсеиваем чужаков. Без фильтра
    # подтверждение тащило бы в человека и аутсайдеров с типичностью 0.1.
    cent = b.get("centroid")
    cent = (np.array(cent, dtype=np.float64) if cent else None)
    if cent is not None:
        cent = cent / (np.linalg.norm(cent) or 1)
    min_typ = float(b.get("min_typicality", ASSIGN_MIN_TYPICALITY))

    updated, skipped, enrolled, errors = 0, 0, 0, []
    for part in parts:
        rid, label = int(part["recording_id"]), str(part["label"])

        # какие фрагменты этого кластера достаточно похожи на голос группы
        rows = q("""SELECT id, embedding FROM segments
                     WHERE recording_id = %s AND speaker_name = %s""", (rid, label))
        if cent is not None:
            scored = []
            for r in rows:
                if r["embedding"] is None:
                    continue        # без вектора судить не по чему — оставляем как есть
                v = np.array(json.loads(r["embedding"]) if isinstance(r["embedding"], str)
                             else r["embedding"], dtype=np.float64)
                cos = float(cent @ (v / (np.linalg.norm(v) or 1)))
                if cos >= min_typ:
                    scored.append((cos, r["id"]))
            # по убыванию похожести: первый — самый типичный, он и пойдёт в эталоны
            scored.sort(reverse=True)
            ids = [i for _, i in scored]
            skipped += len(rows) - len(ids)
        else:
            ids = [r["id"] for r in rows]
        if not ids:
            continue

        def run(c, ids=ids):
            with c.transaction():
                got = c.execute("""
                    UPDATE segments SET speaker_name = %s,
                           speaker_id = (SELECT id FROM speakers WHERE name = %s),
                           ambiguous = false,
                           detail = coalesce(detail, '{}'::jsonb)
                                    || jsonb_build_object('resolution', %s::text)
                     WHERE id = ANY(%s) RETURNING id""",
                    (name, name, f"user:{name} (глобально)", ids)).fetchall()
                if got:
                    # курсор настроен на словари: r["id"], а не r[0] — иначе KeyError
                    c.execute("""UPDATE conflicts SET status='resolved', resolved_name=%s,
                                 resolved_by='user', resolved_at=now()
                                 WHERE status='open' AND segment_id = ANY(%s)""",
                              (name, [r["id"] for r in got]))
                return len(got)
        got_n = _run(run)
        updated += got_n

        # Эталон берём ТОЛЬКО после успешной разметки этой записи. Раньше он
        # создавался раньше UPDATE, и упавший запрос оставлял за собой эталоны при
        # том, что разметка откатывалась: база голосов росла от неудавшихся попыток.
        if enroll_core and got_n and ids:
            best_id = ids[0]        # ids отсортированы по убыванию похожести на группу
            seg = q1("""SELECT recording_id, start_sec, end_sec FROM segments
                         WHERE id = %s AND end_sec - start_sec >= 2""", (best_id,))
            if seg:
                try:
                    enroll({"recording_id": seg["recording_id"], "start_sec": seg["start_sec"],
                            "end_sec": seg["end_sec"], "speaker_name": name})
                    enrolled += 1
                except HTTPException as e:
                    errors.append(f"#{rid}: {e.detail}")
    return {"ok": True, "name": name, "updated": updated, "skipped": skipped,
            "enrolled": enrolled, "errors": errors[:5]}


@app.post("/api/sync/now")
def sync_now():
    """Сходить в облако Plaud прямо сейчас, не дожидаясь планового обхода.

    Воркер опрашивает облако раз в пятнадцать минут, и после «я только что залил
    запись» ждать этот цикл незачем. Заодно возвращаем в очередь всё, что зависло
    на скачивании: раздача Plaud умеет замирать, не разрывая соединения.
    """
    import sys
    sys.path.insert(0, "/worker")
    stuck = _run(lambda c: c.execute("""
        UPDATE recordings SET status='pending_download'
         WHERE status='downloading'
           AND id IN (SELECT recording_id FROM jobs
                       WHERE stage='download' AND status='running'
                         AND started_at < now() - interval '10 minutes')
        RETURNING id""").fetchall())
    if stuck:
        _run(lambda c: c.execute("""
            UPDATE jobs SET status='failed', finished_at=now(),
                   error=coalesce(error,'') || ' [зависло, возвращено в очередь]'
             WHERE stage='download' AND status='running'
               AND recording_id = ANY(%s)""", ([r[0] for r in stuck],)))
    return {"ok": True, "requeued": len(stuck or []),
            "note": "воркер подхватит очередь в течение полуминуты"}


@app.get("/api/review/queue")
def review_queue():
    """Записи, которые ещё ждут разметки, — самые «дешёвые» сверху.

    Дешёвая — та, где мало кластеров без имени и много уже опознанного: разобрав её,
    человек за пару кликов закрывает много речи. Записи, где всё названо и спорных
    нет, из очереди уходят совсем.
    """
    rows = q("""
        SELECT r.id, r.title, r.filename, r.duration_sec, r.started_at,
               count(*) FILTER (WHERE s.speaker_name ~ '^S[0-9?]') AS unnamed,
               count(*) FILTER (WHERE s.ambiguous) AS ambiguous,
               count(*) AS segments,
               coalesce(sum(s.end_sec - s.start_sec)
                        FILTER (WHERE s.speaker_name ~ '^S[0-9?]'), 0) AS unnamed_sec
          FROM recordings r JOIN segments s ON s.recording_id = r.id
         GROUP BY r.id
        HAVING count(*) FILTER (WHERE s.speaker_name ~ '^S[0-9?]' OR s.ambiguous) > 0
         ORDER BY coalesce(sum(s.end_sec - s.start_sec)
                           FILTER (WHERE s.speaker_name ~ '^S[0-9?]'), 0) DESC""")
    return {"queue": [{**r, "unnamed_min": round(float(r["unnamed_sec"]) / 60, 1)} for r in rows]}


@app.get("/api/enroll/candidates")
def enroll_candidates(recording_id: int):
    rec, src = rec_audio(recording_id)
    segs = q(f"""SELECT {SEG_COLS}, c.reason FROM segments s
                 LEFT JOIN LATERAL (SELECT reason FROM conflicts
                                     WHERE segment_id = s.id AND status = 'open'
                                     ORDER BY id LIMIT 1) c ON true
                 WHERE s.recording_id = %s ORDER BY s.start_sec""", (recording_id,))
    names = {r["name"] for r in q("SELECT name FROM speakers")}
    refs = known_refs()
    base_names, base_M = base_matrix()
    # векторы отдельным запросом: в общий список колонок их класть нельзя — это
    # 192 float на каждую реплику, а нужны они только здесь
    vec_by_id = {r["id"]: (json.loads(r["embedding"]) if isinstance(r["embedding"], str)
                           else r["embedding"])
                 for r in q("""SELECT id, embedding FROM segments
                                WHERE recording_id = %s AND embedding IS NOT NULL""",
                            (recording_id,))}
    groups = {}
    for s in segs:
        groups.setdefault(s["speaker_name"], []).append(s)
    out = []
    for label, items in groups.items():
        items.sort(key=lambda x: (x["end_sec"] - x["start_sec"]), reverse=True)
        # отдаём кластер целиком (с потолком на всякий случай): по пяти обрывкам голос
        # не опознать, а фильтры и счётчики в UI должны считать по всем репликам.
        # reason важен человеку: top2_close («похож и на другого») и low_confidence —
        # это разные ситуации, и разбираются они по-разному
        top = [utt(s, src is not None,
                   live_top=live_top(vec_by_id.get(s["id"]), base_names, base_M))
               for s in items[:300]]
        # эталон и матч считаем по самому длинному НЕспорному фрагменту: спорный может
        # оказаться чужой репликой, попавшей в кластер по ошибке провайдера
        clean = [s for s in items if not s["ambiguous"]] or items
        best = clean[0]

        # ЯДРО кластера — реплики, ближайшие к его центроиду: самые типичные для
        # этого голоса. Именно они годятся в эталоны, тогда как «самый длинный
        # фрагмент» запросто оказывается длинным из-за шума или чужой вставки.
        # Считается по сохранённым векторам, без аудио и без единого ffmpeg.
        # ЯДРО и АУТСАЙДЕРЫ кластера. Ядро — реплики, ближайшие к центроиду: самые
        # типичные для этого голоса, они и годятся в эталоны («самый длинный
        # фрагмент» запросто длинный из-за шума или чужой вставки). Аутсайдеры —
        # наоборот, самые далёкие: если чужая реплика попала в кластер по ошибке
        # диаризации, она окажется именно здесь. Послушав то и другое, человек
        # проверяет кластер целиком: кто это и не затесалось ли постороннее.
        # Всё считается по сохранённым векторам — без аудио и без ffmpeg.
        core, outliers, cent = [], [], None
        vecs = [(x, np.array(vec_by_id[x["id"]], dtype=np.float64))
                for x in items if x["id"] in vec_by_id]
        if vecs:
            M = np.array([v / (np.linalg.norm(v) or 1) for _, v in vecs])
            c = M.mean(axis=0)
            cent = c / (np.linalg.norm(c) or 1)
            sims = M @ cent
            order = list(np.argsort(-sims))

            def pick(indexes, min_sec, limit):
                out = []
                for i in indexes:
                    seg = vecs[int(i)][0]
                    if seg["end_sec"] - seg["start_sec"] < min_sec:
                        continue
                    # тот же формат «Фразы», что и везде: с текстом, плеером и
                    # поштучными действиями — иначе чужака в кластере видно, а
                    # переназначить его прямо здесь нечем
                    out.append(utt(seg, src is not None,
                                   typicality=round(float(sims[int(i)]), 3),
                                   live_top=live_top(vec_by_id.get(seg["id"]), base_names, base_M)))
                    if len(out) >= limit:
                        break
                return out

            core = pick(order, 2.0, 5)
            core_ids = {c0["id"] for c0 in core}
            # порог короче: аутсайдеры нужны для прослушивания, а не для эталонов
            outliers = [o for o in pick(reversed(order), 1.2, 5) if o["id"] not in core_ids]

        # Матч — по ЦЕНТРОИДУ кластера, а не по одной реплике: усреднение по всем
        # репликам заметно устойчивее. Одиночный фрагмент давал заниженную близость
        # (на записи #24 — 0.48 против 0.87 по центроиду), и голос выглядел чужим.
        match, err = [], None
        if cent is not None and refs:
            match = voice.top_matches(cent, refs, 3)
        elif not refs:
            err = "в базе нет эталонов"
        elif src:
            try:
                v = span_vec(best["id"], src, best["start_sec"], best["end_sec"])
                match = voice.top_matches(v, refs, 3)
            except Exception as e:
                err = str(e)[:200]
        else:
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
            "core": core,            # типичные реплики — кандидаты в эталоны
            "outliers": outliers,    # наименее типичные — здесь всплывает чужое
            "ambiguous_count": sum(1 for s in items if s["ambiguous"]),
            # уже разобранные человеком — чтобы в заголовке кластера было видно прогресс
            "confirmed_count": sum(1 for s in items if as_dict(s["detail"]).get("resolution")),
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
    Подтверждение снимает пометку спорности, закрывает конфликт и (если enroll=true)
    кладёт фрагмент в эталоны человека.

    enroll=false — «просто смени имя»: пользователь уверен в спикере, но фрагмент
    как эталон не годится (обрывок, шум на фоне, чужой микрофон).
    """
    b = body or {}
    seg_id = b.get("segment_id")
    name = str(b.get("name", "")).strip()
    want = bool(b.get("enroll", True))
    if seg_id is None or not name:
        raise HTTPException(400, "segment_id and name are required")
    seg = q1("""SELECT s.id, s.recording_id, s.start_sec, s.end_sec, r.audio_path
                FROM segments s JOIN recordings r ON r.id = s.recording_id
                WHERE s.id = %s""", (int(seg_id),))
    if seg is None:
        raise HTTPException(404, "segment not found")

    # шум/фон эталоном не делаем — только переклеиваем метку у реплики
    res, err = {"enrolled": False}, None
    if name == "[noise]":
        res["noise"] = True
    elif want:
        try:
            res = {**enroll({"recording_id": seg["recording_id"], "start_sec": seg["start_sec"],
                             "end_sec": seg["end_sec"], "speaker_name": name}), "enrolled": True}
        except HTTPException as e:
            # эталон не вышел (коротыш, нет аудио) — но имя спикера всё равно меняем:
            # решение человека важнее, чем неудача с клипом
            err = str(e.detail)

    note = "подтверждён эталоном" if res.get("enrolled") else "без эталона"

    def run(c):
        with c.transaction():
            c.execute("""
                UPDATE segments
                SET speaker_name = %s,
                    speaker_id = (SELECT id FROM speakers WHERE name = %s),
                    ambiguous = false,
                    detail = coalesce(detail, '{}'::jsonb)
                             || jsonb_build_object('resolution', %s::text)
                WHERE id = %s""", (name, name, f"user:{name} ({note})", seg["id"]))
            c.execute("""UPDATE conflicts SET status='resolved', resolved_name=%s,
                         resolved_by='user', resolved_at=now()
                         WHERE segment_id = %s AND status='open'""", (name, seg["id"]))
            # Решение человека храним отдельно от сегментов: переигрывание опознания
            # пересоздаёт segments, и иначе ручная разметка исчезала бы при каждой
            # смене порога склейки. Привязка по времени — границы реплик плавают.
            c.execute("""INSERT INTO manual_labels
                         (recording_id, start_sec, end_sec, speaker_name, note)
                         VALUES (%s,%s,%s,%s,%s)""",
                      (seg["recording_id"], seg["start_sec"], seg["end_sec"], name, note))
    _run(run)
    out = segments_by_ids([seg["id"]])
    return {**res, "segment_id": seg["id"], "confirmed": True, "name": name,
            "enroll_error": err, "utterance": out[0] if out else None}


@app.post("/api/segments/bulk")
def segments_bulk(body: dict):
    """Массовое действие над выбранными фразами: всем один спикер (или '[noise]').

    Эталон по умолчанию НЕ создаётся: пачка выбрана глазами по списку, среди неё почти
    наверняка есть обрывки. enroll="longest" берёт одну самую длинную фразу, "all" —
    каждую подходящую: когда человек прослушал и уверен во всех, это самый быстрый
    способ набрать базу голоса. Слишком короткие пропускаем молча — эталон из
    полусекундного обрывка только портит центроид.
    """
    b = body or {}
    ids = []
    for x in (b.get("ids") or []):
        try:
            ids.append(int(x))
        except (TypeError, ValueError):
            pass
    name = str(b.get("name", "")).strip()
    if not ids or not name:
        raise HTTPException(400, "ids and name are required")

    # enroll: false/"none" | true/"longest" | "all"
    mode = b.get("enroll", False)
    mode = "longest" if mode is True else ("none" if mode is False else str(mode))
    enrolled, err, added = None, None, 0
    if mode in ("longest", "all") and name != "[noise]":
        rows = q("""SELECT id, recording_id, start_sec, end_sec FROM segments
                     WHERE id = ANY(%s) ORDER BY end_sec - start_sec DESC""", (ids,))
        if mode == "longest":
            rows = rows[:1]
        for r in rows:
            if float(r["end_sec"]) - float(r["start_sec"]) < 1.5:
                continue          # короче полутора секунд — сигнала мало, центроид испортит
            try:
                enrolled = enroll({"recording_id": r["recording_id"], "start_sec": r["start_sec"],
                                   "end_sec": r["end_sec"], "speaker_name": name})
                added += 1
            except HTTPException as e:
                err = str(e.detail)

    note = f"user:{name} (массово, {f'эталонов +{added}' if added else 'без эталона'})"

    def run(c):
        with c.transaction():
            cur = c.execute("""
                UPDATE segments
                SET speaker_name = %s,
                    speaker_id = (SELECT id FROM speakers WHERE name = %s),
                    ambiguous = false,
                    detail = coalesce(detail, '{}'::jsonb)
                             || jsonb_build_object('resolution', %s::text)
                WHERE id = ANY(%s) RETURNING id""", (name, name, note, ids))
            n = len(cur.fetchall())
            cur = c.execute("""UPDATE conflicts SET status='resolved', resolved_name=%s,
                               resolved_by='user', resolved_at=now()
                               WHERE status='open' AND segment_id = ANY(%s) RETURNING id""",
                            (name, ids))
            return n, len(cur.fetchall())

    updated, closed = _run(run)
    return {"ok": True, "updated": updated, "conflicts_closed": closed, "name": name,
            "enrolled_added": added,
            "enrolled": enrolled, "enroll_error": err,
            "segments": segments_by_ids(ids)}


def _words(raw):
    """words из jsonb -> список слов с валидными ГЛОБАЛЬНЫМИ таймкодами."""
    if isinstance(raw, str):
        try:
            raw = json.loads(raw)
        except Exception:
            return []
    out = []
    for w in (raw or []):
        if not isinstance(w, dict):
            continue
        try:
            s, e = float(w["start"]), float(w["end"])
        except (KeyError, TypeError, ValueError):
            continue
        if e > s:
            out.append({**w, "start": s, "end": e})
    return out


def _grid(n, k):
    """k равномерных индексов из диапазона [0, n)."""
    if n <= k:
        return list(range(n))
    return sorted({round(i * (n - 1) / (k - 1)) for i in range(k)})


def _best_cut(clip, spans, cands):
    """Точка разреза с минимальным косинусом между половинами.

    Полный перебор границ слов стоит 2 эмбеддинга на границу и на длинной реплике
    занимает минуты, поэтому идём в два прохода: грубая сетка, затем уточнение
    вокруг найденного минимума.
    """
    seen = {}

    def cos_at(i):
        if i not in seen:
            try:
                seen[i] = voice.cos(clip.embed_spans(spans[:i]), clip.embed_spans(spans[i:]))
            except Exception:
                seen[i] = 1.0                 # половина не эмбеддится — точка не годится
        return seen[i]

    n = len(cands)
    coarse = _grid(n, SPLIT_PROBES)
    bp = min(coarse, key=lambda p: cos_at(cands[p]))
    step = max(1, (n - 1) // max(SPLIT_PROBES - 1, 1))
    lo, hi = max(0, bp - step), min(n - 1, bp + step)
    fine = [lo + p for p in _grid(hi - lo + 1, SPLIT_PROBES)]
    bp = min(fine + [bp], key=lambda p: cos_at(cands[p]))
    return cands[bp], seen[cands[bp]]


@app.post("/api/segments/{seg_id}/split")
def split_segment(seg_id: int):
    """Разрезать реплику там, где внутри неё сменился говорящий.

    Кандидаты — границы слов; для каждой считаем вектор левой и правой части ПО
    СКЛЕЕННОЙ РЕЧИ (паузы выброшены) и ищем минимум косинуса. Если минимум выше
    SPLIT_MAX_COS или в половине меньше SPLIT_MIN_SIDE_SEC речи — НЕ режем и
    объясняем почему: молчаливый отказ выглядел бы как сломанная кнопка.
    """
    seg = q1("""SELECT s.id, s.recording_id, s.start_sec, s.end_sec, s.text, s.speaker_id,
                       s.speaker_name, s.detail, s.words, s.asr_logprob, r.audio_path
                FROM segments s JOIN recordings r ON r.id = s.recording_id
                WHERE s.id = %s""", (seg_id,))
    if seg is None:
        raise HTTPException(404, "segment not found")

    no = lambda why: {"split": False, "segment_id": seg_id, "reason": why}

    words = _words(seg["words"])
    if len(words) < 6:
        return no("у реплики нет пословных таймкодов (или их меньше шести) — резать не по чему")
    spans = [(w["start"], w["end"]) for w in words]
    total = sum(e - s for s, e in spans)
    if total < 2 * SPLIT_MIN_SIDE_SEC:
        return no(f"в реплике всего {total:.1f}с речи — на две половины "
                  f"по {SPLIT_MIN_SIDE_SEC:g}с не хватает")
    src = safe_data_path(seg["audio_path"])
    if src is None:
        return no("аудиофайл записи недоступен — посчитать голоса половин нечем")

    # кандидаты: границы слов, по обе стороны которых достаточно речи
    acc, left = [], 0.0
    for s, e in spans:
        left += e - s
        acc.append(left)
    cands = [i for i in range(1, len(words))
             if acc[i - 1] >= SPLIT_MIN_SIDE_SEC and total - acc[i - 1] >= SPLIT_MIN_SIDE_SEC]
    if not cands:
        return no("нет точки, где по обе стороны набирается "
                  f"{SPLIT_MIN_SIDE_SEC:g}с речи")

    a = max(0.0, min(float(seg["start_sec"]), spans[0][0]) - 0.10)
    b = max(float(seg["end_sec"]), spans[-1][1]) + 0.10
    try:
        clip = voice.Clip(Path(src), a, b - a)
    except Exception as e:
        return no(f"не удалось декодировать аудио реплики: {str(e)[:200]}")
    idx, best = _best_cut(clip, spans, cands)
    if best > SPLIT_MAX_COS:
        return no(f"голос внутри реплики не меняется: самая непохожая пара половин даёт "
                  f"косинус {best:.2f}, а режем только ниже {SPLIT_MAX_COS:.2f}")

    lw, rw = words[:idx], words[idx:]
    cut, r_start = float(lw[-1]["end"]), float(rw[0]["start"])
    txt = lambda ws: " ".join(str(w.get("text", "")) for w in ws).strip()
    vec = {}
    for side, ws in (("l", spans[:idx]), ("r", spans[idx:])):
        try:                                   # вектор половины уже осмыслен — сохраняем
            vec[side] = voice.vec_literal(clip.embed_spans(ws))
        except Exception:
            vec[side] = None
    info = {"split_cos": round(best, 3), "split_at": round(cut, 2)}

    def run(c):
        with c.transaction():
            # левая половина остаётся в исходной строке: ссылки на сегмент не ломаются
            c.execute("""
                UPDATE segments
                SET end_sec = %s, text = %s, words = %s::jsonb, ambiguous = true,
                    confidence = NULL, inherited = false, embedding = %s::vector,
                    detail = coalesce(detail, '{}'::jsonb) || %s::jsonb
                WHERE id = %s""",
                      (cut, txt(lw), json.dumps(lw, ensure_ascii=False), vec["l"],
                       json.dumps({**info, "split_side": "left"}), seg_id))
            row = c.execute("""
                INSERT INTO segments (recording_id, start_sec, end_sec, text, speaker_id,
                                      speaker_name, confidence, inherited, ambiguous, detail,
                                      embedding, words, asr_logprob)
                VALUES (%s, %s, %s, %s, %s, %s, NULL, false, true, %s::jsonb, %s::vector,
                        %s::jsonb, %s)
                RETURNING id""",
                            (seg["recording_id"], r_start, float(seg["end_sec"]), txt(rw),
                             seg["speaker_id"], seg["speaker_name"],
                             json.dumps({**info, "split_from": seg_id, "split_side": "right"}),
                             vec["r"], json.dumps(rw, ensure_ascii=False), seg["asr_logprob"]))
            return row.fetchone()["id"]

    new_id = _run(run)
    return {"split": True, "cos": round(best, 3), "cut_sec": round(cut, 2),
            "segment_id": seg_id, "new_id": new_id,
            "segments": segments_by_ids([seg_id, new_id])}


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
                 {SEG_COLS}, coalesce(r.title, r.filename) AS rec_title, r.audio_path
          FROM segments s JOIN recordings r ON r.id = s.recording_id
          WHERE s.end_sec - s.start_sec >= 6 AND s.speaker_name ~* '{ANON_RE}'
          ORDER BY s.recording_id, s.speaker_name, (s.end_sec - s.start_sec) DESC
        ) t ORDER BY (t.end_sec - t.start_sec) DESC LIMIT 30""")
    enrich = q(f"""
        SELECT * FROM (
          SELECT DISTINCT ON (s.recording_id, s.speaker_name)
                 {SEG_COLS}, coalesce(r.title, r.filename) AS rec_title, r.audio_path
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
    sug = lambda rows: [utt(r, safe_data_path(r["audio_path"]) is not None,
                            rec_title=r["rec_title"], asr_logprob=r["asr_logprob"])
                        for r in rows]
    return {"unnamed": sug(unnamed), "enrich": sug(enrich), "bad_samples": bad}


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
    """Спорные фразы — тот же формат фразы, плюс поля самого конфликта."""
    rows = q(f"""
        SELECT c.id AS conflict_id, c.reason, c.status, c.judge_verdict,
               c.resolved_name, c.resolved_by, c.created_at, c.resolved_at,
               {SEG_COLS}, coalesce(r.title, r.filename) AS rec_title, r.audio_path
        FROM conflicts c JOIN segments s ON s.id = c.segment_id
                         JOIN recordings r ON r.id = s.recording_id
        WHERE c.status = %s
        ORDER BY c.id""", (status,))
    return [utt(r, safe_data_path(r["audio_path"]) is not None,
                conflict_id=r["conflict_id"], segment_id=r["id"], status=r["status"],
                rec_title=r["rec_title"], judge_verdict=r["judge_verdict"],
                resolved_name=r["resolved_name"], resolved_by=r["resolved_by"],
                created_at=r["created_at"], resolved_at=r["resolved_at"])
            for r in rows]


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
    # Ждущий статус очереди нужного этапа: пулы воркеров разбирают записи
    # по статусу, а не по флагу в meta.
    queue = {"detect": "new", "quality": "detected", "resolve": "transcribed"}
    if stage not in queue:
        raise HTTPException(400, "from_stage must be detect|quality|resolve")

    def run(c):
        return c.execute("UPDATE recordings SET status = %s WHERE id = %s RETURNING id",
                         (queue[stage], rec_id)).fetchone()

    if _run(run) is None:
        raise HTTPException(404, "recording not found")
    return {"ok": True, "id": rec_id, "from_stage": stage}


@app.get("/api/balances")
def balances():
    """Остатки на счетах провайдеров: очередь не должна вставать молча из-за нуля.

    ElevenLabs отдаёт квоту только ключу с правом user_read — без него честно
    говорим, что остаток недоступен, вместо того чтобы показать ноль.
    """
    import httpx
    c = httpx.Client(timeout=20, trust_env=False)   # системный прокси не для loopback-мира
    out = []
    try:
        d = c.get("https://openrouter.ai/api/v1/credits",
                  headers={"Authorization": f"Bearer {os.environ['OPENROUTER_API_KEY']}"}).json()["data"]
        left = float(d["total_credits"]) - float(d["total_usage"])
        out.append({"api": "OpenRouter", "ok": True, "text": f"${left:,.2f}",
                    "low": left < 5, "note": "судья спорных реплик"})
    except Exception as exc:
        out.append({"api": "OpenRouter", "ok": False, "text": "нет данных", "note": str(exc)[:100]})
    try:
        r = c.get("https://api.elevenlabs.io/v1/user/subscription",
                  headers={"xi-api-key": os.environ["ELEVENLABS_API_KEY"]})
        if r.status_code == 401:
            out.append({"api": "ElevenLabs", "ok": False, "text": "нужно право user_read",
                        "note": "elevenlabs.io -> API key -> включить User: Read"})
        else:
            d = r.json()
            used, lim = int(d.get("character_count", 0)), int(d.get("character_limit", 0))
            left = lim - used
            out.append({"api": "ElevenLabs", "ok": True,
                        "text": f"{left:,} из {lim:,} кредитов".replace(",", " "),
                        "low": lim and left < lim * 0.1,
                        "note": f"тариф {d.get('tier', '?')}, транскрипция Scribe"})
    except Exception as exc:
        out.append({"api": "ElevenLabs", "ok": False, "text": "нет данных", "note": str(exc)[:100]})
    return {"balances": out}


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


# --- сводный прогресс конвейера (шапка вкладки Jobs) -------------------------

PIPELINE = ["download", "detect", "quality", "resolve", "store"]


# Пул на каждый этап свой: этапы упираются в разные ресурсы (сеть чужих сервисов
# против локального процессора), поэтому и очереди у них раздельные.
# stage -> (env лимита, дефолт, статус «ждёт», статус «в работе»)
POOLS = {
    "download": ("DOWNLOAD_WORKERS", 6, "pending_download", "downloading"),
    "detect":   ("DETECT_WORKERS",   5, "new",              "detecting"),
    "quality":  ("QUALITY_WORKERS",  6, "detected",         "transcribing"),
    "resolve":  ("RESOLVE_WORKERS",  5, "transcribed",      "resolving"),
}


@app.get("/api/progress")
def progress():
    """Сколько записей и часов прошло каждый этап конвейера.

    Считаем по завершённым job'ам: этап пройден, если у записи есть job этой стадии
    со статусом done. Часы берём из длительности записи — так виден реальный объём
    работы, а не просто число файлов (запись на 12 часов и на 5 минут несопоставимы).
    """
    totals = q1("""SELECT count(*) AS recs,
                          coalesce(sum(duration_sec), 0) / 3600.0 AS hours,
                          coalesce(sum(size_bytes), 0) / 1073741824.0 AS gb
                     FROM recordings""") or {}
    # скачивание считаем по факту наличия файла: часть записей пришла не из облака
    downloaded = q1("""SELECT count(*) AS recs,
                              coalesce(sum(duration_sec), 0) / 3600.0 AS hours
                         FROM recordings WHERE audio_path IS NOT NULL""") or {}

    # Старый этап draft (whisper-черновик) засчитываем как пройденный детектор:
    # у двух записей из прошлой эпохи пайплайна detect-джобы нет вовсе, и без этой
    # поправки транскрипция обгоняла детектор — картина, невозможная по устройству
    # конвейера, где каждая запись сначала проходит детект.
    done_by_stage = {r["stage"]: r for r in q("""
        SELECT CASE WHEN j.stage = 'draft' THEN 'detect' ELSE j.stage END AS stage,
               count(DISTINCT r.id) AS recs,
               coalesce(sum(DISTINCT r.duration_sec), 0) / 3600.0 AS hours
          FROM jobs j JOIN recordings r ON r.id = j.recording_id
         WHERE j.status = 'done'
         GROUP BY 1""")}
    running = {r["stage"]: r["n"] for r in q(
        "SELECT stage, count(*) AS n FROM jobs WHERE status='running' GROUP BY stage")}
    # Ошибки показываем ТОЛЬКО актуальные: у записей, которые сейчас в статусе failed.
    # Историю падений считать бессмысленно — почти все они уже переиграны после
    # починок, и в шапке это выглядело так, будто всё разваливается.
    # Ошибки — ТОЛЬКО по последней попытке каждой стадии. Раньше считались все
    # исторические падения записи, и после починки скачивание показывало «16 ошибок»
    # при полностью скачанных 46 файлах: старые записи о давно исправленных сбоях.
    failed = {r["stage"]: r["n"] for r in q("""
        SELECT stage, count(*) AS n FROM (
            SELECT DISTINCT ON (j.recording_id, j.stage) j.stage, j.status
              FROM jobs j JOIN recordings r ON r.id = j.recording_id
             WHERE r.status = 'failed'
             ORDER BY j.recording_id, j.stage, j.id DESC) t
         WHERE t.status = 'failed' GROUP BY stage""")}
    retried = q1("""SELECT count(*) AS n FROM jobs j JOIN recordings r ON r.id = j.recording_id
                     WHERE j.status='failed' AND r.status <> 'failed'""") or {}

    by_status = {r["status"]: r for r in q("""
        SELECT status, count(*) AS n, coalesce(sum(duration_sec), 0) / 3600.0 AS hours
          FROM recordings GROUP BY status""")}

    total_recs = totals.get("recs") or 0
    total_hours = float(totals.get("hours") or 0)
    stages = []
    for st in PIPELINE:
        if st == "download":
            recs = downloaded.get("recs") or 0
            hours = float(downloaded.get("hours") or 0)
        else:
            row = done_by_stage.get(st) or {}
            recs = row.get("recs") or 0
            hours = float(row.get("hours") or 0)
        # у store своего пула нет: он идёт хвостом опознания в том же потоке
        env, default, wait_st, busy_st = POOLS.get(st, (None, 0, None, None))
        w = by_status.get(wait_st) or {}
        b = by_status.get(busy_st) or {}
        stages.append({
            "stage": st,
            "recs": recs, "recs_total": total_recs,
            "hours": round(hours, 1), "hours_total": round(total_hours, 1),
            "pct": round(hours / total_hours * 100, 1) if total_hours else 0.0,
            "running": running.get(st, 0), "failed": failed.get(st, 0),
            # детализация пула: занято потоков из лимита и сколько ждёт своей очереди
            "workers": int(os.environ.get(env, default)) if env else 0,
            "busy": b.get("n", 0),
            "waiting": w.get("n", 0),
            "waiting_hours": round(float(w.get("hours") or 0), 1),
        })
    # Что прямо сейчас в работе и сколько осталось. Скорость берём фактическую —
    # по завершённым за последний час джобам того же этапа, а не по константе:
    # длинная ночная запись и десятиминутный разговор идут совершенно по-разному.
    speed = {r["stage"]: float(r["ratio"]) for r in q("""
        SELECT j.stage,
               sum(r.duration_sec) / nullif(sum(extract(epoch FROM j.finished_at - j.started_at)), 0) AS ratio
          FROM jobs j JOIN recordings r ON r.id = j.recording_id
         WHERE j.status = 'done' AND j.finished_at > now() - interval '2 hours'
         GROUP BY 1""") if r["ratio"]}
    running = []
    for r in q("""SELECT j.id, j.stage, j.recording_id, j.meta, r.title, r.filename, r.duration_sec,
                         extract(epoch FROM now() - j.started_at) AS elapsed
                    FROM jobs j JOIN recordings r ON r.id = j.recording_id
                   WHERE j.status = 'running' ORDER BY j.started_at"""):
        meta = as_dict(r["meta"])
        done, total = float(meta.get("done_sec") or 0), float(meta.get("total_sec") or 0)
        pct = round(done / total * 100) if total else None
        # ETA сначала по собственному прогрессу джобы, а если его ещё нет — по
        # средней скорости этапа: пусть грубо, но не «неизвестно» целый час
        eta = None
        if done > 0 and total > done and r["elapsed"]:
            eta = (total - done) / (done / float(r["elapsed"]))
        elif speed.get(r["stage"]):
            left = float(r["duration_sec"] or 0) / speed[r["stage"]] - float(r["elapsed"] or 0)
            eta = max(left, 0)
        running.append({"stage": r["stage"], "recording_id": r["recording_id"],
                        "title": r["title"] or r["filename"],
                        "minutes": round(float(r["duration_sec"] or 0) / 60),
                        "pct": pct, "elapsed_min": round(float(r["elapsed"] or 0) / 60),
                        "eta_min": None if eta is None else round(eta / 60)})

    return {
        "running": running,
        "speed": {k: round(v, 1) for k, v in speed.items()},
        "recs_total": total_recs,
        "hours_total": round(total_hours, 1),
        "gb_total": round(float(totals.get("gb") or 0), 1),
        "stages": stages,
        # общий прогресс: доля часов, доехавших до конца конвейера
        "overall_pct": stages[-1]["pct"] if stages else 0.0,
        "retried": retried.get("n", 0),   # падения, уже исправленные повтором
        "queue": {r["status"]: r["n"] for r in q(
            "SELECT status, count(*) AS n FROM recordings GROUP BY status")},
    }
