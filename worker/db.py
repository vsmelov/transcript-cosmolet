"""Тонкие хелперы над psycopg3. Соединение переустанавливается при обрыве."""
from __future__ import annotations

import json
from datetime import date

import psycopg

import config

_conn: psycopg.Connection | None = None


def conn() -> psycopg.Connection:
    global _conn
    if _conn is None or _conn.closed:
        _conn = psycopg.connect(config.DATABASE_URL, autocommit=True)
    return _conn


def q(sql: str, *args):
    with conn().cursor() as cur:
        cur.execute(sql, args)
        if cur.description:
            return cur.fetchall()
        return None


def q1(sql: str, *args):
    rows = q(sql, *args)
    return rows[0] if rows else None


def add_cost(usd: float, kind: str, model: str, recording_id: int | None, note: str = "") -> None:
    q("INSERT INTO costs (usd, kind, model, recording_id, note) VALUES (%s,%s,%s,%s,%s)",
      round(usd, 6), kind, model, recording_id, note)


def spent_today() -> float:
    row = q1("SELECT COALESCE(SUM(usd),0) FROM costs WHERE day = %s", date.today())
    return float(row[0])


def budget_left() -> float:
    return config.DAILY_BUDGET_USD - spent_today()


def job_start(recording_id: int, stage: str) -> int:
    row = q1("""INSERT INTO jobs (recording_id, stage, status, started_at)
                VALUES (%s,%s,'running',now()) RETURNING id""", recording_id, stage)
    return row[0]


def job_done(job_id: int, cost: float = 0.0, artifact_path: str | None = None, meta: dict | None = None) -> None:
    q("""UPDATE jobs SET status='done', finished_at=now(), cost_usd=%s,
         artifact_path=%s, meta=meta || %s::jsonb WHERE id=%s""",
      round(cost, 6), artifact_path, json.dumps(meta or {}), job_id)


def job_fail(job_id: int, error: str) -> None:
    q("UPDATE jobs SET status='failed', finished_at=now(), error=%s WHERE id=%s",
      error[:2000], job_id)


def wait_ready(timeout: float = 120.0, log=print) -> None:
    """Дождаться готовности БД при старте.

    Контейнер поднимается быстрее, чем Postgres успевает инициализироваться, и
    воркер падал с 'the database system is starting up', перезапускаясь по кругу.
    """
    import time as _t
    deadline = _t.time() + timeout
    while True:
        try:
            q1("SELECT 1")
            return
        except Exception as exc:
            if _t.time() > deadline:
                raise
            log("жду базу:", str(exc)[:80])
            _t.sleep(3)
