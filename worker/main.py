"""Воркер cosmolet: очередь папок -> черновик -> качество -> опознание -> хранение.

Каждый этап пишет job в БД (телеметрия для UI) и json-артефакт в data/artifacts/.
Бюджетный предохранитель: как только расход за сутки достигает DAILY_BUDGET_USD,
новые платные этапы не стартуют — очередь ждёт следующего дня, ничего не теряется.
"""
from __future__ import annotations

import json
import shutil
import time
import traceback
from pathlib import Path

import audio as audio_mod
import clients
import config
import db
import diarize
import embed as embed_mod
import speechmap
import store

AUDIO_EXT = {".m4a", ".mp3", ".wav", ".ogg", ".oga", ".opus", ".webm", ".flac", ".mp4", ".mov"}
POLL_SEC = 10


def log(*a):
    print(time.strftime("[%H:%M:%S]"), *a, flush=True)


def artifact(recording_id: int, name: str, data: dict) -> str:
    d = config.ARTIFACTS / f"rec_{recording_id}"
    d.mkdir(parents=True, exist_ok=True)
    p = d / f"{name}.json"
    p.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")
    return str(p)


def intake() -> None:
    """Новые файлы из inbox -> запись в БД (аудио переезжает в data/audio, оно нам нужно)."""
    for f in sorted(config.INBOX.iterdir()) if config.INBOX.exists() else []:
        if not f.is_file() or f.suffix.lower() not in AUDIO_EXT:
            continue
        try:
            dur = audio_mod.duration(f)
        except Exception as exc:
            log("не читается, в failed:", f.name, exc)
            shutil.move(str(f), config.FAILED / f.name)
            continue
        dest = config.AUDIO / f.name
        if dest.exists():
            dest = config.AUDIO / f"{f.stem}_{int(time.time())}{f.suffix}"
        shutil.move(str(f), dest)
        db.q("""INSERT INTO recordings (filename, source, audio_path, duration_sec, size_bytes, status)
                VALUES (%s,'inbox',%s,%s,%s,'new')""",
             dest.name, str(dest), dur, dest.stat().st_size)
        log("принят:", dest.name, f"{dur/60:.1f} мин")


def stage_draft(rec_id: int, src: Path, dur: float) -> dict:
    """Черновик по всему файлу (whisper-turbo) -> карта речи."""
    job = db.job_start(rec_id, "draft")
    try:
        cost_total = 0.0
        parts, pos = [], 0.0
        while pos < dur:
            take = min(config.QUALITY_CHUNK_SEC, dur - pos)
            tmp = config.IN_PROGRESS / f"draft_{rec_id}_{int(pos)}{src.suffix}"
            audio_mod.cut(src, pos, take, tmp)
            resp, cost = clients.draft_transcribe(tmp)
            tmp.unlink(missing_ok=True)
            cost_total += cost
            db.add_cost(cost, "draft", config.DRAFT_MODEL, rec_id, f"offset={pos:.0f}")
            for s in resp.get("segments") or []:
                s["start"] = float(s.get("start", 0)) + pos
                s["end"] = float(s.get("end", 0)) + pos
                parts.append(s)
            pos += take
        smap = speechmap.build({"segments": parts}, dur)
        path = artifact(rec_id, "draft", {"speech_map": smap, "segments": parts})
        db.job_done(job, cost_total, path, {"speech_sec": smap["speech_sec"]})
        log(f"черновик: речь {smap['speech_sec']}с из {smap['total_sec']}с, ${cost_total:.4f}")
        return smap
    except Exception as exc:
        db.job_fail(job, traceback.format_exc())
        raise


def stage_quality(rec_id: int, src: Path, smap: dict) -> list:
    """Scribe по речевым регионам с анонимной диаризацией -> реплики."""
    job = db.job_start(rec_id, "quality")
    try:
        utts, cost_total = [], 0.0
        for ri, (rs, re_) in enumerate(smap["regions"]):
            pos = rs
            while pos < re_:
                take = min(config.QUALITY_CHUNK_SEC, re_ - pos)
                tmp = config.IN_PROGRESS / f"q_{rec_id}_{ri}_{int(pos)}{src.suffix}"
                audio_mod.cut(src, pos, take, tmp)
                resp, cost = clients.scribe_transcribe(tmp, take)
                tmp.unlink(missing_ok=True)
                cost_total += cost
                db.add_cost(cost, "quality", config.SCRIBE_MODEL, rec_id, f"region={ri}")
                utts += diarize.utterances_from_scribe(resp, pos, f"r{ri}p{int(pos)}")
                pos += take
        utts.sort(key=lambda u: u.start)
        path = artifact(rec_id, "quality", {"utterances": [
            {"start": u.start, "end": u.end, "raw_speaker": u.raw_speaker,
             "text": u.text, "asr_logprob": u.asr_logprob} for u in utts]})
        db.job_done(job, cost_total, path, {"utterances": len(utts)})
        log(f"качество: {len(utts)} реплик, ${cost_total:.4f}")
        return utts
    except Exception:
        db.job_fail(job, traceback.format_exc())
        raise


def stage_resolve(rec_id: int, src: Path, utts: list) -> dict:
    """Локальное опознание + текст-судья по конфликтам."""
    job = db.job_start(rec_id, "resolve")
    try:
        base = embed_mod.known_speakers()
        info = diarize.assign_speakers(utts, src, base)
        amb_before = sum(1 for u in utts if u.ambiguous)
        closed, cost = (0, 0.0)
        if amb_before:
            closed, cost = diarize.judge_conflicts(utts)
            if cost:
                db.add_cost(cost, "judge", config.JUDGE_MODEL, rec_id, f"conflicts={amb_before}")
        path = artifact(rec_id, "resolve", {
            "clusters": info["clusters"], "base_speakers": sorted(base),
            "ambiguous_before_judge": amb_before, "closed_by_judge": closed,
            "ambiguous_after": sum(1 for u in utts if u.ambiguous)})
        db.job_done(job, cost, path, {"clusters": len(info["clusters"]), "closed_by_judge": closed})
        log(f"опознание: {[c['name'] for c in info['clusters']]}, "
            f"спорных {amb_before} -> {sum(1 for u in utts if u.ambiguous)}")
        return info
    except Exception:
        db.job_fail(job, traceback.format_exc())
        raise


def process(rec) -> None:
    rec_id, filename, audio_path, dur = rec[0], rec[1], Path(rec[2]), float(rec[3] or 0)
    log(f"=== запись #{rec_id} {filename} ({dur/60:.1f} мин)")
    db.q("UPDATE recordings SET status='processing' WHERE id=%s", rec_id)
    try:
        smap = stage_draft(rec_id, audio_path, dur)
        utts = stage_quality(rec_id, audio_path, smap)
        info = stage_resolve(rec_id, audio_path, utts)
        job = db.job_start(rec_id, "store")
        md = store.save(rec_id, filename, audio_path, dur, utts, info["clusters"], smap)
        db.job_done(job, 0.0, str(md))
        db.q("UPDATE recordings SET status='done' WHERE id=%s", rec_id)
        log(f"=== готово #{rec_id}: {md}")
    except Exception as exc:
        db.q("UPDATE recordings SET status='failed' WHERE id=%s", rec_id)
        log(f"!!! ошибка #{rec_id}: {exc}")


def main() -> None:
    for d in (config.INBOX, config.IN_PROGRESS, config.DONE, config.FAILED,
              config.ARTIFACTS, config.AUDIO):
        d.mkdir(parents=True, exist_ok=True)
    log("воркер запущен, бюджет/день:", config.DAILY_BUDGET_USD)
    while True:
        try:
            intake()
            left = db.budget_left()
            if left <= 0.01:
                log(f"бюджет на сегодня исчерпан (потрачено ${db.spent_today():.3f}) — ждём")
                time.sleep(300)
                continue
            rec = db.q1("""SELECT id, filename, audio_path, duration_sec FROM recordings
                           WHERE status='new' ORDER BY created_at LIMIT 1""")
            if rec:
                process(rec)
            else:
                time.sleep(POLL_SEC)
        except Exception:
            log("цикл упал:", traceback.format_exc()[:500])
            time.sleep(30)


if __name__ == "__main__":
    main()
