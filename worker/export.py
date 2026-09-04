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

import config
import db
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

    for st, en, text, name, conf, amb, _ in segs:
        h, rem = divmod(int(float(st)), 3600)
        m, sec = divmod(rem, 60)
        clock = f" {at(float(st)):%H:%M:%S}" if at else ""
        c = "" if conf is None else f" · {float(conf):.2f}"
        flag = " ⚠️" if amb else ""
        lines.append(f"**[{h:02d}:{m:02d}:{sec:02d}{clock}] {display_speaker(name)}{c}{flag}:** {text}")
        lines.append("")

    d = config.ARTIFACTS / f"rec_{recording_id}"
    d.mkdir(parents=True, exist_ok=True)
    (d / "SCHEMA.md").write_text(SCHEMA_MD, encoding="utf-8")
    md = d / "transcript.md"
    md.write_text("\n".join(lines), encoding="utf-8")
    return md


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
