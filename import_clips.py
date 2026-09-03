"""Считает эмбеддинги нарезанных клипов (внутри контейнера воркера) и пишет их в БД."""
import json
import sys
from pathlib import Path

sys.path.insert(0, "/app")
import embed  # noqa: E402
import db  # noqa: E402

clips = json.loads(Path("/data/enrich_clips.json").read_text(encoding="utf-8"))
added = skipped = 0
for name, rel, dur, source in clips:
    path = f"/data/speakers/{rel}"
    if not Path(path).exists():
        skipped += 1
        continue
    row = db.q1("SELECT id FROM speakers WHERE name = %s", name)
    if not row:
        skipped += 1
        continue
    if db.q1("SELECT 1 FROM speaker_samples WHERE path = %s", path):
        skipped += 1
        continue
    try:
        vec = embed.embed_span(Path(path), 0, dur)
    except Exception as exc:
        print("не считается:", rel, exc)
        skipped += 1
        continue
    db.q("""INSERT INTO speaker_samples (speaker_id, path, kind, duration_sec, embedding, source)
            VALUES (%s,%s,'embed',%s,%s::vector,%s)""",
         row[0], path, dur, json.dumps([round(float(x), 6) for x in vec]), f"auto:{source}")
    added += 1

print(f"добавлено {added}, пропущено {skipped}")
for name, cnt in db.q("""SELECT s.name, count(ss.id) FROM speakers s
                         LEFT JOIN speaker_samples ss ON ss.speaker_id=s.id
                         GROUP BY s.name ORDER BY 2 DESC"""):
    print(f"  {name}: {cnt}")
