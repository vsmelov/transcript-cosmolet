"""Подбор параметров карты речи на реальных данных.

Знаем правду: где Scribe нашёл слова — там речь. Для набора (gap, pad) считаем:
  sent    — сколько секунд ушло бы в дорогую модель,
  recall  — какая доля настоящей речи попала в регионы (терять нельзя!),
  cost    — во что это обходится.
"""
import json
import sys
from pathlib import Path

sys.stdout.reconfigure(encoding="utf-8", errors="replace")
REC = int(sys.argv[1]) if len(sys.argv) > 1 else 3
A = Path(f"data/artifacts/rec_{REC}")
draft = json.loads((A / "draft.json").read_text(encoding="utf-8"))
qual = json.loads((A / "quality.json").read_text(encoding="utf-8"))

total = draft["speech_map"]["total_sec"]
truth = [(float(w["start"]), float(w["end"]))
         for u in qual["utterances"] for w in (u.get("words") or [])]
truth_sec = sum(e - s for s, e in truth)
P = 0.22 / 3600


def build(gap, pad, nsp, logp, comp):
    ok = []
    for s in draft["segments"]:
        t = (s.get("text") or "").strip().lower()
        if (not t or float(s.get("no_speech_prob") or 0) > nsp
                or float(s.get("avg_logprob") or 0) < logp
                or float(s.get("compression_ratio") or 1) > comp):
            continue
        ok.append((float(s["start"]), float(s["end"])))
    regions = []
    for s, e in ok:
        s, e = max(0.0, s - pad), min(total, e + pad)
        if regions and s - regions[-1][1] <= gap:
            regions[-1][1] = max(regions[-1][1], e)
        else:
            regions.append([s, e])
    sent = sum(e - s for s, e in regions)
    covered = 0.0
    i = 0
    for ws, we in truth:                       # доля настоящей речи внутри регионов
        while i < len(regions) and regions[i][1] < ws:
            i += 1
        j = i
        while j < len(regions) and regions[j][0] < we:
            covered += max(0.0, min(we, regions[j][1]) - max(ws, regions[j][0]))
            j += 1
    return sent, covered / truth_sec if truth_sec else 0.0


print(f"запись #{REC}: {total/60:.0f} мин, настоящей речи (слова Scribe) {truth_sec:.0f}с "
      f"({truth_sec/total*100:.0f}%)")
print(f"{'gap':>4} {'pad':>4} {'nsp':>5} | {'отправим':>9} {'% зап.':>7} {'recall':>7} {'$Scribe':>8}")
print("-" * 60)
for gap, pad, nsp in [(30, 10, 0.6), (20, 8, 0.6), (12, 5, 0.6), (8, 4, 0.6),
                      (8, 3, 0.5), (5, 3, 0.5), (5, 2, 0.4), (3, 2, 0.4), (2, 1.5, 0.3)]:
    sent, rec = build(gap, pad, nsp, -1.35, 2.4)
    flag = "  <-- ТЕРЯЕМ РЕЧЬ" if rec < 0.97 else ""
    print(f"{gap:>4} {pad:>4} {nsp:>5} | {sent:>8.0f}с {sent/total*100:>6.0f}% "
          f"{rec*100:>6.1f}% {sent*P:>7.3f}${flag}")
print()
print(f"для справки: весь файл = {total*P:.3f}$, только чистая речь = {truth_sec*P:.3f}$")
