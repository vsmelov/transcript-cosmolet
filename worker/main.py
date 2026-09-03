"""Воркер cosmolet: очередь папок -> детектор речи -> качество -> опознание -> хранение.

Каждый этап пишет job в БД (телеметрия для UI) и json-артефакт в data/artifacts/.
Бюджетный предохранитель: как только расход за сутки достигает DAILY_BUDGET_USD,
новые платные этапы не стартуют — очередь ждёт следующего дня, ничего не теряется.
"""
from __future__ import annotations

import json
import os
import shutil
import tempfile
import threading
import time
import traceback
from pathlib import Path

import audio as audio_mod
import clients
import config
import db
import diarize
import embed as embed_mod
import detect
import store
import sync

AUDIO_EXT = {".m4a", ".mp3", ".wav", ".ogg", ".oga", ".opus", ".webm", ".flac", ".mp4", ".mov"}
POLL_SEC = 10
_SIZES: dict[str, int] = {}   # имя -> размер на прошлом проходе (детект «файл ещё пишется»)


def log(*a):
    print(time.strftime("[%H:%M:%S]"), *a, flush=True)


def artifact(recording_id: int, name: str, data: dict) -> str:
    d = config.ARTIFACTS / f"rec_{recording_id}"
    d.mkdir(parents=True, exist_ok=True)
    p = d / f"{name}.json"
    p.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")
    return str(p)


def intake() -> None:
    """Новые файлы из inbox -> запись в БД (аудио переезжает в data/audio, оно нам нужно).

    Рядом с аудио может лежать sidecar <файл>.meta.json (кладёт, например, fetch_plaud.py):
    {"title": ..., "started_at": ISO, "source": ..., ...} — оттуда берём человеческое
    название и время записи вместо голого имени файла.
    """
    for f in sorted(config.INBOX.iterdir()) if config.INBOX.exists() else []:
        if not f.is_file() or f.suffix.lower() not in AUDIO_EXT:
            continue
        # файл может ещё скачиваться/копироваться: берём только когда размер
        # перестал меняться между проходами и он ненулевой
        size = f.stat().st_size
        if size == 0 or _SIZES.get(f.name) != size:
            _SIZES[f.name] = size
            continue
        _SIZES.pop(f.name, None)
        try:
            dur = audio_mod.duration(f)
        except Exception as exc:
            log("не читается, в failed:", f.name, exc)
            shutil.move(str(f), config.FAILED / f.name)
            continue

        sidecar = f.with_suffix(f.suffix + ".meta.json")
        meta = {}
        if sidecar.exists():
            try:
                meta = json.loads(sidecar.read_text(encoding="utf-8"))
            except Exception:
                meta = {}

        # дедупликация: тот же файл (имя+размер) уже принят — не заводим вторую запись
        dup = db.q1("""SELECT id FROM recordings
                       WHERE filename LIKE %s AND size_bytes = %s LIMIT 1""",
                    f"{f.stem}%{f.suffix}", size)
        if dup:
            log(f"дубль (уже запись #{dup[0]}), файл в failed:", f.name)
            shutil.move(str(f), config.FAILED / f.name)
            sidecar.unlink(missing_ok=True)
            continue

        dest = config.AUDIO / f.name
        if dest.exists():
            dest = config.AUDIO / f"{f.stem}_{int(time.time())}{f.suffix}"
        shutil.move(str(f), dest)
        sidecar.unlink(missing_ok=True)

        title = meta.get("title") or dest.stem.replace("_", " ")
        db.q("""INSERT INTO recordings
                (filename, title, source, audio_path, duration_sec, size_bytes, started_at, status, meta)
                VALUES (%s,%s,%s,%s,%s,%s,%s,'new',%s::jsonb)""",
             dest.name, title, meta.get("source", "inbox"), str(dest), dur,
             dest.stat().st_size, meta.get("started_at"), json.dumps(meta, ensure_ascii=False))
        log("принят:", title, f"({dur/60:.1f} мин)")


def stage_detect(rec_id: int, src: Path, dur: float) -> dict:
    """Карта речи локально и бесплатно: Silero VAD + классификатор аудио-событий.

    Пришёл на смену платному whisper-черновику: дешевле (ноль), честнее на тишине
    (whisper там галлюцинировал) и отличает речь от речеподобного шума в кармане.
    """
    job = db.job_start(rec_id, "detect")
    try:
        with tempfile.TemporaryDirectory(prefix="detect_") as td:
            wav = Path(td) / "full.wav"
            audio_mod.to_wav16k(src, wav)
            smap = detect.detect(wav, dur)
        path = artifact(rec_id, "detect", {"speech_map": smap})
        db.job_done(job, 0.0, path, {"speech_sec": smap["speech_sec"],
                                     "vad_sec": smap["vad_sec"]})
        log(f"детектор: VAD {smap['vad_sec']}с -> речь {smap['speech_sec']}с "
            f"из {smap['total_sec']:.0f}с ({smap['speech_sec']/max(dur,1)*100:.1f}%), "
            f"шум отброшен: {smap['dropped_by_tagger_sec']}с")
        return smap
    except Exception:
        db.job_fail(job, traceback.format_exc())
        raise


def stage_quality(rec_id: int, src: Path, smap: dict) -> list:
    """Scribe по речевым регионам с анонимной диаризацией -> реплики."""
    job = db.job_start(rec_id, "quality")
    try:
        utts, cost_total = [], 0.0
        for ri, (rs, re_) in enumerate(smap["regions"]):
            pos = rs
            # Хвост короче минимума пропускаем. Сложение pos += take накапливает
            # погрешность, и без этого порога цикл заходил на лишний виток, вырезая
            # кусок в доли миллисекунды: ffmpeg отдавал огрызок, а Scribe отвечал
            # «File is corrupted» — на записи #13 из-за этого падал весь этап.
            while re_ - pos >= config.QUALITY_MIN_CHUNK_SEC:
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
        # words сохраняем: без них нельзя переиграть resolve (детектор смены говорящего
        # режет реплики по границам слов) без повторной оплаты транскрипции
        path = artifact(rec_id, "quality", {"utterances": [
            {"start": u.start, "end": u.end, "raw_speaker": u.raw_speaker,
             "text": u.text, "asr_logprob": u.asr_logprob, "words": u.words} for u in utts]})
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
        # порог склейки, выбранный руками в UI, перекрывает автоподбор
        meta = db.q1("SELECT meta FROM recordings WHERE id=%s", rec_id)[0] or {}
        if isinstance(meta, str):
            meta = json.loads(meta)
        forced = meta.get("join")
        info = diarize.assign_speakers(utts, src, base, float(forced) if forced else None)
        amb_before = sum(1 for u in utts if u.ambiguous)
        closed, cost = (0, 0.0)
        if amb_before:
            closed, cost = diarize.judge_conflicts(utts)
            if cost:
                db.add_cost(cost, "judge", config.JUDGE_MODEL, rec_id, f"conflicts={amb_before}")
        path = artifact(rec_id, "resolve", {
            "clusters": info["clusters"], "base_speakers": sorted(base),
            "split_utterances": info.get("splits", 0),
            "ambiguous_before_judge": amb_before, "closed_by_judge": closed,
            "ambiguous_after": sum(1 for u in utts if u.ambiguous)})
        db.job_done(job, cost, path, {"clusters": len(info["clusters"]),
                                      "closed_by_judge": closed, "splits": info.get("splits", 0)})
        log(f"опознание: {[c['name'] for c in info['clusters']]}, "
            f"разрезано склеек: {info.get('splits', 0)}, "
            f"спорных {amb_before} -> {sum(1 for u in utts if u.ambiguous)}")
        return info
    except Exception:
        db.job_fail(job, traceback.format_exc())
        raise


def load_artifact(rec_id: int, name: str) -> dict | None:
    p = config.ARTIFACTS / f"rec_{rec_id}" / f"{name}.json"
    if not p.is_file():
        return None
    try:
        return json.loads(p.read_text(encoding="utf-8"))
    except Exception:
        return None


# Конвейер: запись движется по статусам, на каждом этапе её подхватывает
# свой пул. Ждущие статусы (справа) — очередь соответствующего этапа.
#   pending_download -> downloading -> new -> detecting -> detected
#   -> transcribing  -> transcribed -> resolving -> done
# Данные между этапами не передаются в памяти — только через артефакты на диске,
# поэтому этапы независимы, могут идти в разном темпе и переживают рестарт.
STAGES = {
    "detect":  {"wait": "new",         "busy": "detecting",  "next": "detected"},
    "quality": {"wait": "detected",    "busy": "transcribing", "next": "transcribed"},
    "resolve": {"wait": "transcribed", "busy": "resolving",  "next": "done"},
}
# Ждущий статус для записи, у которой этап переигрывают вручную из UI
REPROCESS_STATUS = {"detect": "new", "quality": "detected", "resolve": "transcribed"}


def claim(stage: str):
    """Атомарно забрать одну запись из очереди этапа (два потока одну не возьмут)."""
    st = STAGES[stage]
    return db.q1("""
        UPDATE recordings SET status=%s
        WHERE id = (SELECT id FROM recordings WHERE status=%s
                    ORDER BY coalesce(started_at, created_at) DESC
                    FOR UPDATE SKIP LOCKED LIMIT 1)
        RETURNING id, filename, audio_path, duration_sec""", st["busy"], st["wait"])


def load_utterances(rec_id: int) -> list:
    cached = load_artifact(rec_id, "quality")
    if not cached:
        raise RuntimeError("нет артефакта quality — этап транскрипции не прошёл")
    return [diarize.Utterance(
        start=u["start"], end=u["end"], text=u["text"], raw_speaker=u["raw_speaker"],
        words=u.get("words") or [], asr_logprob=u.get("asr_logprob"))
        for u in cached["utterances"]]


def run_stage(stage: str, rec) -> None:
    """Один этап одной записи. Ошибка помечает запись failed, очередь едет дальше."""
    rec_id, filename, audio_path, dur = rec[0], rec[1], Path(rec[2]), float(rec[3] or 0)
    try:
        if stage == "detect":
            stage_detect(rec_id, audio_path, dur)
        elif stage == "quality":
            cached = load_artifact(rec_id, "detect")
            if not cached:
                raise RuntimeError("нет артефакта detect")
            stage_quality(rec_id, audio_path, cached["speech_map"])
        else:
            utts = load_utterances(rec_id)
            info = stage_resolve(rec_id, audio_path, utts)
            smap = (load_artifact(rec_id, "detect") or {}).get("speech_map", {})
            job = db.job_start(rec_id, "store")
            md = store.save(rec_id, filename, audio_path, dur, utts, info["clusters"], smap)
            db.job_done(job, 0.0, str(md))
            log(f"=== готово #{rec_id}: {md}")
        db.q("UPDATE recordings SET status=%s WHERE id=%s", STAGES[stage]["next"], rec_id)
    except clients.QuotaExceeded:
        raise                       # обрабатывается в stage_loop: запись ждёт, не падает
    except Exception as exc:
        db.q("UPDATE recordings SET status='failed' WHERE id=%s", rec_id)
        log(f"!!! #{rec_id} на этапе {stage}: {exc}")


def stage_loop(stage: str, num: int) -> None:
    """Обработчик одного этапа: тянет свою очередь, пока она не опустеет."""
    paid = stage in ("quality", "resolve")
    # Этапы, которые считают локально, уступают процессор всему остальному:
    # в Linux nice действует на конкретный поток, а запущенный отсюда ffmpeg
    # наследует приоритет. Ждущим сеть этапам это не нужно — они и так простаивают.
    if stage in ("detect", "resolve") and config.WORKER_NICE:
        try:
            os.nice(config.WORKER_NICE)
        except OSError:
            pass
    while True:
        try:
            # бюджет стережём только на платных этапах: детектор бесплатный и
            # должен продолжать готовить работу, даже когда деньги на сегодня кончились
            if paid and db.budget_left() <= 0.01:
                time.sleep(300)
                continue
            rec = claim(stage)
            if rec:
                try:
                    run_stage(stage, rec)
                except clients.QuotaExceeded as exc:
                    # Деньги у провайдера кончились: возвращаем запись в очередь как
                    # была и ждём. Иначе за минуту вся очередь ушла бы в failed по
                    # причине, которая к самим записям отношения не имеет.
                    db.q("UPDATE recordings SET status=%s WHERE id=%s",
                         STAGES[stage]["wait"], rec[0])
                    log(f"{stage}: кончилась квота провайдера, пауза 10 мин — {exc}")
                    time.sleep(600)
            else:
                time.sleep(POLL_SEC)
        except Exception:
            log(f"{stage}/{num} упал:", traceback.format_exc()[:300])
            time.sleep(30)


def recover_zombies() -> None:
    """После рестарта контейнера незавершённые джобы висят в running навсегда.

    Помечаем их прерванными, а записи возвращаем в очередь ТОГО этапа, до которого
    они реально дошли: всё, что уже оплачено и лежит в артефактах, не переигрываем.
    """
    db.q("""UPDATE jobs SET status='failed', finished_at=now(),
            error=COALESCE(error,'') || ' [прервано рестартом воркера]'
            WHERE status='running'""")
    stuck = db.q("""SELECT id FROM recordings
                    WHERE status IN ('detecting','transcribing','resolving','processing')""") or []
    for (rec_id,) in stuck:
        stage = "detect"
        if load_artifact(rec_id, "detect"):
            stage = "quality"
        if load_artifact(rec_id, "quality"):
            stage = "resolve"
        db.q("UPDATE recordings SET status=%s WHERE id=%s", REPROCESS_STATUS[stage], rec_id)
        log(f"после рестарта запись #{rec_id} возвращена в очередь этапа {stage}")
    # скачивание прервалось на полпути — .part докачивать нечем, начинаем файл заново
    back = db.q("""UPDATE recordings SET status='pending_download'
                   WHERE status='downloading' RETURNING id""") or []
    if back:
        log(f"возвращено в очередь скачивания после рестарта: {len(back)}")


def main() -> None:
    for d in (config.INBOX, config.IN_PROGRESS, config.DONE, config.FAILED,
              config.ARTIFACTS, config.AUDIO):
        d.mkdir(parents=True, exist_ok=True)
    db.wait_ready(log=log)
    log("воркер запущен, бюджет/день:", config.DAILY_BUDGET_USD)
    recover_zombies()
    pools = {"detect": config.DETECT_WORKERS, "quality": config.QUALITY_WORKERS,
             "resolve": config.RESOLVE_WORKERS}
    for stage, n in pools.items():
        for i in range(max(1, n)):
            threading.Thread(target=stage_loop, args=(stage, i + 1), daemon=True).start()
    if config.PLAUD_SYNC_ENABLED:
        threading.Thread(target=sync.sync_loop, args=(log,), daemon=True).start()
        for i in range(max(1, config.DOWNLOAD_WORKERS)):
            threading.Thread(target=sync.downloader_loop, args=(i + 1, log), daemon=True).start()
    log("пулы: скачивание %d, детектор %d, транскрипт %d, опознание %d" % (
        config.DOWNLOAD_WORKERS, config.DETECT_WORKERS,
        config.QUALITY_WORKERS, config.RESOLVE_WORKERS))
    while True:
        try:
            intake()
            time.sleep(POLL_SEC)
        except Exception:
            log("цикл приёма файлов упал:", traceback.format_exc()[:300])
            time.sleep(30)


if __name__ == "__main__":
    main()
