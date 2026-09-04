"""Разовый импорт базы голосов из transcribe-mcp/voiceprints в Postgres cosmolet."""
import json
import shutil
from pathlib import Path

VP = Path(r"C:\Users\v\PycharmProjects\claude-workspace\transcribe-mcp\voiceprints")
DEST = Path(r"C:\Users\v\PycharmProjects\claude-workspace\cosmolet\data\speakers")
OUT = Path(r"C:\Users\v\PycharmProjects\claude-workspace\cosmolet\data\seed_speakers.sql")  # data/ не в git: имена и голоса реальных людей

DEST.mkdir(parents=True, exist_ok=True)
registry = json.loads((VP / "registry.json").read_text(encoding="utf-8"))
embs = json.loads((VP / "embeddings.json").read_text(encoding="utf-8"))


def esc(s) -> str:
    return "'" + str(s).replace("'", "''") + "'"


sql = []
for p in registry:
    aliases = ("ARRAY[" + ",".join(esc(a) for a in p["aliases"]) + "]::text[]"
               if p.get("aliases") else "'{}'::text[]")
    sql.append(f"INSERT INTO speakers (name, aliases) VALUES ({esc(p['name'])}, {aliases}) "
               f"ON CONFLICT (name) DO NOTHING;")
    for s in p.get("samples", []):
        src = VP / s["file"]
        if not src.exists():
            continue
        rel = s["file"].replace("\\", "/")
        dst = DEST / rel
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy(src, dst)
        vec = embs.get(s["file"])
        if not vec:
            continue
        sql.append(
            "INSERT INTO speaker_samples (speaker_id, path, kind, duration_sec, embedding, source) "
            f"SELECT id, {esc('/data/speakers/' + rel)}, {esc(s.get('kind', 'reference'))}, "
            f"{s.get('duration_sec') or 0}, {esc(json.dumps(vec))}::vector, "
            f"{esc(s.get('source', ''))} FROM speakers WHERE name={esc(p['name'])};")

OUT.write_text("\n".join(sql), encoding="utf-8")
print("людей:", len(registry), "| строк SQL:", len(sql))
