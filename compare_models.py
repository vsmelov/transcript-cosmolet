"""Нужен ли качественный проход? Сравниваем текст черновика (whisper-turbo) и Scribe.

Оба прогона уже оплачены и лежат в артефактах записи. Берём одинаковые временные окна
и смотрим: расходятся ли тексты, где именно и насколько.
"""
import json
import re
import sys
from difflib import SequenceMatcher
from pathlib import Path

sys.stdout.reconfigure(encoding="utf-8", errors="replace")
REC = int(sys.argv[1]) if len(sys.argv) > 1 else 3
A = Path(f"data/artifacts/rec_{REC}")

draft = json.loads((A / "draft.json").read_text(encoding="utf-8"))
quality = json.loads((A / "quality.json").read_text(encoding="utf-8"))

dsegs = [s for s in draft["segments"] if (s.get("text") or "").strip()]
qsegs = [u for u in quality["utterances"] if (u.get("text") or "").strip()]


def norm(t: str) -> str:
    return re.sub(r"[^\w\s]", "", (t or "").lower()).strip()


def text_in(segs, a: float, b: float, key_s="start", key_e="end") -> str:
    got = [s["text"] for s in segs
           if float(s[key_s]) < b and float(s[key_e]) > a]
    return norm(" ".join(got))


dur = draft["speech_map"]["total_sec"]
print(f"запись #{REC}: {dur/60:.0f} мин | черновик {len(dsegs)} сегм. | Scribe {len(qsegs)} реплик")
print(f"речь по карте черновика: {draft['speech_map']['speech_sec']:.0f}с "
      f"({draft['speech_map']['speech_sec']/dur*100:.0f}% записи)\n")

# окна по 60с внутри речевых регионов
windows, scores = [], []
for rs, re_ in draft["speech_map"]["regions"]:
    t = rs
    while t + 60 <= re_:
        d, q = text_in(dsegs, t, t + 60), text_in(qsegs, t, t + 60)
        if len(d) > 80 and len(q) > 80:
            r = SequenceMatcher(None, d, q).ratio()
            windows.append((t, r, d, q))
            scores.append(r)
        t += 60

scores.sort()
print(f"сравнимых окон по 60с: {len(scores)}")
if scores:
    n = len(scores)
    print(f"совпадение текстов: медиана {scores[n//2]:.2f}, "
          f"худшие 10% ниже {scores[max(0,n//10)]:.2f}, лучшие 10% выше {scores[min(n-1,9*n//10)]:.2f}")

    print("\n=== ГДЕ ЧЕРНОВИК ХУЖЕ ВСЕГО (3 окна с наибольшим расхождением) ===")
    for t, r, d, q in sorted(windows, key=lambda x: x[1])[:3]:
        m, s = divmod(int(t), 60)
        print(f"\n[{m:02d}:{s:02d}] совпадение {r:.2f}")
        print(f"  черновик: {d[:230]}")
        print(f"  Scribe:   {q[:230]}")

    print("\n=== ГДЕ ОНИ СОВПАДАЮТ (2 окна) ===")
    for t, r, d, q in sorted(windows, key=lambda x: -x[1])[:2]:
        m, s = divmod(int(t), 60)
        print(f"\n[{m:02d}:{s:02d}] совпадение {r:.2f}")
        print(f"  черновик: {d[:180]}")
        print(f"  Scribe:   {q[:180]}")

# что даёт Scribe сверх текста
words = sum(len(u.get("words") or []) for u in qsegs)
spk = {w.get("speaker_id") for u in qsegs for w in (u.get("words") or [])}
print(f"\nScribe сверх текста: {words} слов с таймкодами, {len(spk)} меток спикеров")
print("черновик: пословных таймкодов нет, спикеров нет")
