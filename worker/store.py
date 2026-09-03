"""Сохранение результата: Postgres + человекочитаемый .md + SCHEMA.md рядом."""
from __future__ import annotations

import json
from pathlib import Path

import config
import db

SCHEMA_MD = """# Формат транскриптов cosmolet

Схема сегмента (версия 1). В обычном случае — плоско:

```json
{"speaker": "Alice Example", "confidence": 0.91}
```

`confidence` — косинусная близость голоса этой реплики к эталону назначенного
человека (0..1). Разворот появляется ТОЛЬКО у спорных реплик, когда сработало
хотя бы одно условие: confidence < 0.70, отрыв top1→top2 < 0.15, либо голос
реплики ближе к другому человеку, чем к дефолту её кластера:

```json
{"speaker": "Alice Example", "confidence": 0.55, "ambiguous": true,
 "top": [{"name": "Alice Example", "cos": 0.55}, {"name": "Bob Example", "cos": 0.49}],
 "cluster_default": {"name": "Bob Example", "cos": 0.58},
 "reason": "cluster_mismatch",
 "judge": {"speaker": "Alice Example", "confidence": 0.8, "why": "..."},
 "resolution": "judge:Alice Example (переназначил)"}
```

Реплики короче 1.5 с не измеряются голосом (слишком мало сигнала): у них
`confidence: null` и `inherited: true` — спикер унаследован от кластера, это
честная пометка «не измеряли», а не выдуманная цифра.

## Как получаются имена

1. **Черновик** (whisper-turbo) слушает весь файл и даёт карту речи —
   где говорят, а где тишина/шум/галлюцинация модели.
2. **Качественный проход** (ElevenLabs Scribe) транскрибирует речевые регионы
   с анонимной диаризацией: speaker_0, speaker_1... — локально для куска.
3. **Опознание** (локально, без API): вектор голоса каждой реплики (CAM++ 192d),
   кластеризация меток по косинусу через всю запись, матч кластеров против базы
   голосов. Имена из базы, неизвестные — S1, S2...
4. **Конфликты**: спорные реплики судит дешёвая LLM по контексту диалога
   (кто говорит о себе, к кому обращаются). Голос+текст сошлись — закрыто.
   Остальное попадает в очередь на ручной разбор в UI.

`asr_logprob` — уверенность самой ASR-модели в словах реплики (ближе к 0 — лучше),
это про качество ТЕКСТА и не связано с `confidence` (это про спикера).
"""


def save(recording_id: int, src_name: str, audio_path: Path, duration: float,
         utts: list, clusters: list[dict], stats: dict) -> Path:
    """Записывает сегменты и конфликты в БД, кладёт .md и SCHEMA.md в artifacts."""
    name_to_id = {r[1]: r[0] for r in (db.q("SELECT id, name FROM speakers") or [])}
    db.q("DELETE FROM segments WHERE recording_id = %s", recording_id)

    for u in utts:
        vec = None if u.vec is None else json.dumps([round(float(x), 6) for x in u.vec])
        row = db.q1("""INSERT INTO segments
            (recording_id, start_sec, end_sec, text, speaker_id, speaker_name, confidence,
             inherited, ambiguous, detail, embedding, words, asr_logprob)
            VALUES (%s,%s,%s,%s,%s,%s,%s,%s,%s,%s::jsonb,%s::vector,%s::jsonb,%s)
            RETURNING id""",
            recording_id, u.start, u.end, u.text, name_to_id.get(u.speaker), u.speaker,
            u.confidence, u.inherited, u.ambiguous,
            json.dumps(u.detail, ensure_ascii=False) if u.detail else None,
            vec, json.dumps(u.words, ensure_ascii=False), u.asr_logprob)
        if u.ambiguous:
            db.q("INSERT INTO conflicts (segment_id, reason, judge_verdict) VALUES (%s,%s,%s::jsonb)",
                 row[0], (u.detail or {}).get("reason", "low_confidence"),
                 json.dumps((u.detail or {}).get("judge"), ensure_ascii=False)
                 if (u.detail or {}).get("judge") else None)

    md_dir = config.ARTIFACTS / f"rec_{recording_id}"
    md_dir.mkdir(parents=True, exist_ok=True)
    (md_dir / "SCHEMA.md").write_text(SCHEMA_MD, encoding="utf-8")

    lines = [f"# {src_name}", "",
             f"- Длительность: {duration/60:.1f} мин",
             f"- Аудио: `{audio_path}`",
             f"- Реплик: {len(utts)}, спорных: {sum(1 for u in utts if u.ambiguous)}",
             f"- Речь по карте черновика: {stats.get('speech_sec')}с из {stats.get('total_sec')}с",
             "- Спикеры: " + ", ".join(f"{c['name']} (~{c['spoke_sec']}с)" for c in clusters),
             "- Формат разметки: см. SCHEMA.md рядом", "", "---", ""]
    for u in utts:
        h, rem = divmod(int(u.start), 3600)
        m, s = divmod(rem, 60)
        conf = "" if u.confidence is None else f" · {u.confidence:.2f}"
        flag = " ⚠️" if u.ambiguous else ""
        lines.append(f"**[{h:02d}:{m:02d}:{s:02d}] {u.speaker}{conf}{flag}:** {u.text}")
        lines.append("")
    md = md_dir / "transcript.md"
    md.write_text("\n".join(lines), encoding="utf-8")
    return md
