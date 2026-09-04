"""Пересборка transcript.md из базы — без переигрывания опознания.

Файл транскрипта пишется в конце обработки, и всё, что человек поправил потом
(подтверждения, массовые назначения, разбор глобальных групп), в него не попадает:
в базе имена уже верные, а в файле — прежние. Здесь мы просто перечитываем сегменты
и переписываем markdown, ничего не считая заново.
"""
from __future__ import annotations

import json
from datetime import timedelta
from pathlib import Path
from zoneinfo import ZoneInfo

import numpy as np

import config
import db
import embed as embed_mod
from store import SCHEMA_MD, display_speaker


def rebuild(recording_id: int) -> Path | None:
    rec = db.q1("""SELECT filename, title, audio_path, duration_sec, started_at
                     FROM recordings WHERE id = %s""", recording_id)
    if not rec:
        return None
    filename, title, audio_path, duration, started = rec
    segs = db.q("""SELECT start_sec, end_sec, text, speaker_name, confidence, ambiguous, detail
                     FROM segments WHERE recording_id = %s ORDER BY start_sec""", recording_id) or []
    if not segs:
        return None

    local = ZoneInfo(config.LOCAL_TZ)
    at = (lambda sec: (started + timedelta(seconds=sec)).astimezone(local)) if started else None

    # Кандидаты в базе хранятся только у спорных реплик (чтобы не раздувать её), а в
    # транскрипте они нужны у каждой неопознанной: без них непонятно, кому фраза
    # могла бы принадлежать. Здесь досчитываем недостающих по эталонам голосов.
    base = embed_mod.known_speakers()
    cands: dict[int, list] = {}
    id_by_start: dict[float, int] = {}
    if base:
        names = sorted(base)
        B = np.array([base[n] for n in names])
        for sid, start, emb in db.q("""SELECT id, start_sec, embedding FROM segments
                                        WHERE recording_id = %s AND embedding IS NOT NULL""",
                                    recording_id) or []:
            v = np.array(json.loads(emb) if isinstance(emb, str) else emb, dtype=np.float64)
            n = np.linalg.norm(v)
            if not n:
                continue
            sim = B @ (v / n)
            cands[sid] = [{"name": names[i], "cos": round(float(sim[i]), 3)}
                          for i in np.argsort(-sim)[:3]]
            id_by_start[round(float(start), 3)] = sid

    by_speaker: dict[str, float] = {}
    manual = 0
    for st, en, _, name, _, _, detail in segs:
        by_speaker[name] = by_speaker.get(name, 0.0) + (float(en) - float(st))
        d = detail if isinstance(detail, dict) else (json.loads(detail) if detail else {})
        if str((d or {}).get("resolution", "")).startswith("user:"):
            manual += 1

    named = sum(v for k, v in by_speaker.items() if not k.startswith("S") and k != "[noise]")
    total = sum(by_speaker.values()) or 1

    lines = [f"# {title or filename}", "",
             (f"- Начало записи: {at(0):%Y-%m-%d %H:%M:%S %Z} (UTC {started:%Y-%m-%d %H:%M:%S})"
              if at else "- Начало записи: неизвестно"),
             f"- Длительность: {float(duration or 0)/60:.1f} мин",
             f"- Аудио: `{audio_path}`",
             f"- Реплик: {len(segs)}, спорных: {sum(1 for s in segs if s[5])}"
             + (f", размечено вручную: {manual}" if manual else ""),
             f"- Речь с именем: {named/60:.1f} из {total/60:.1f} мин ({named/total*100:.0f}%)",
             "- Спикеры: " + ", ".join(
                 f"{display_speaker(n)} (~{v/60:.0f} мин)"
                 for n, v in sorted(by_speaker.items(), key=lambda kv: -kv[1])),
             "- Формат разметки: см. SCHEMA.md рядом", "", "---", ""]

    for st, en, text, name, conf, amb, detail in segs:
        h, rem = divmod(int(float(st)), 3600)
        m, sec = divmod(rem, 60)
        clock = f" {at(float(st)):%H:%M:%S}" if at else ""
        c = "" if conf is None else f" · {float(conf):.2f}"
        flag = " ⚠️" if amb else ""
        lines.append(f"**[{h:02d}:{m:02d}:{sec:02d}{clock}] {display_speaker(name)}{c}{flag}:** {text}")
        # Разбор под спорной или безымянной репликой — тот самый json из SCHEMA.md.
        # Без него транскрипт молчит о том, КТО ЕЩЁ подходил на эту фразу, и читателю
        # (человеку или агенту) нечем оценить, насколько имени можно верить.
        d = detail if isinstance(detail, dict) else (json.loads(detail) if detail else {})
        top = (d or {}).get("top") or cands.get(id_by_start.get(round(float(st), 3)), [])
        if amb or name.startswith("S"):
            info = {k: v for k, v in {
                "top": top,
                "cluster_default": (d or {}).get("cluster_default"),
                "reason": (d or {}).get("reason"),
                "judge": (d or {}).get("judge"),
                "resolution": (d or {}).get("resolution"),
            }.items() if v}
            if info:
                lines.append(f"`{json.dumps(info, ensure_ascii=False)}`")
        lines.append("")

    d = config.ARTIFACTS / f"rec_{recording_id}"
    d.mkdir(parents=True, exist_ok=True)
    (d / "SCHEMA.md").write_text(SCHEMA_MD, encoding="utf-8")
    md = d / "transcript.md"
    md.write_text("\n".join(lines), encoding="utf-8")
    return md


def recheck(log=print) -> int:
    """Пересчитать пометку «спорная» по текущему правилу, не трогая имена.

    Правило уверенности живёт в конфиге и меняется по мере накопления данных, а
    флаг лежит в базе с момента обработки. Переигрывать ради него опознание —
    это часы счёта; здесь достаточно сравнить вектор реплики с эталонами заново.
    Имена не меняются: решение о том, КТО говорит, остаётся прежним.
    """
    base = embed_mod.known_speakers()
    if not base:
        log("база голосов пуста — пересчитывать нечего")
        return 0
    names = sorted(base)
    B = np.array([base[n] for n in names])
    rows = db.q("""SELECT id, speaker_name, embedding, ambiguous FROM segments
                    WHERE embedding IS NOT NULL""") or []
    changed = 0
    for sid, name, emb, was in rows:
        v = np.array(json.loads(emb) if isinstance(emb, str) else emb, dtype=np.float64)
        n = np.linalg.norm(v)
        if not n:
            continue
        sim = B @ (v / n)
        order = np.argsort(-sim)
        own = float(sim[names.index(name)]) if name in names else float(sim[order[0]])
        margin = float(sim[order[0]]) - float(sim[order[1]]) if len(order) > 1 else 1.0
        amb = own < config.UTT_CONF_OK or margin < config.UTT_TOP2_MARGIN
        if bool(was) != amb:
            db.q("UPDATE segments SET ambiguous = %s WHERE id = %s", amb, sid)
            if not amb:
                db.q("""UPDATE conflicts SET status='resolved', resolved_by='rule',
                        resolved_at=now() WHERE segment_id = %s AND status='open'""", sid)
            changed += 1
    log(f"пометка «спорная» пересчитана: изменено {changed} из {len(rows)}")
    return changed


def rebuild_all(log=print) -> int:
    """Перечитать все записи, у которых есть сегменты."""
    ids = [r[0] for r in db.q("SELECT DISTINCT recording_id FROM segments ORDER BY 1") or []]
    n = 0
    for rid in ids:
        try:
            if rebuild(rid):
                n += 1
        except Exception as exc:
            log(f"#{rid}: не пересобрался — {exc}")
    log(f"пересобрано транскриптов: {n} из {len(ids)}")
    return n


if __name__ == "__main__":
    rebuild_all()
