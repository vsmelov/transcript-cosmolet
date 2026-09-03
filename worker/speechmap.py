"""Карта речи из чернового verbose_json: какие регионы файла отдавать в качественный проход.

Ничего не «отрезает» акустикой: черновая STT-модель уже прослушала весь файл,
мы лишь выкидываем регионы, где ОНА сказала «тишина/галлюцинация», и добавляем
щедрые поля. Пороги в config, все решения пишутся в артефакт для дебага в UI.
"""
from __future__ import annotations

import config

# типовые галлюцинации whisper на тишине/музыке
HALLUCINATION_MARKERS = (
    "субтитры", "subtitles", "продолжение следует", "спасибо за просмотр",
    "thanks for watching", "dimatorzok",
)


def classify_segment(seg: dict) -> tuple[bool, str]:
    """(это речь?, причина решения) для одного сегмента черновика."""
    text = (seg.get("text") or "").strip().lower()
    if not text:
        return False, "empty"
    if float(seg.get("no_speech_prob") or 0) > config.NO_SPEECH_MAX:
        return False, f"no_speech_prob>{config.NO_SPEECH_MAX}"
    if float(seg.get("avg_logprob") or 0) < config.AVG_LOGPROB_MIN:
        return False, f"avg_logprob<{config.AVG_LOGPROB_MIN}"
    if float(seg.get("compression_ratio") or 1) > config.COMPRESSION_MAX:
        return False, "compression_ratio (зацикливание)"
    if any(m in text for m in HALLUCINATION_MARKERS):
        return False, "маркер галлюцинации"
    return True, "ok"


def build(draft: dict, total_duration: float) -> dict:
    """verbose_json черновика -> {"regions": [[start,end],...], "decisions": [...]}."""
    decisions = []
    speech: list[list[float]] = []
    for seg in draft.get("segments") or []:
        ok, why = classify_segment(seg)
        start, end = float(seg.get("start") or 0), float(seg.get("end") or 0)
        decisions.append({"start": start, "end": end, "speech": ok, "why": why,
                          "text": (seg.get("text") or "")[:80]})
        if ok:
            speech.append([start, end])

    # склейка через короткие паузы + поля
    regions: list[list[float]] = []
    for s, e in speech:
        s = max(0.0, s - config.REGION_PAD_SEC)
        e = min(total_duration, e + config.REGION_PAD_SEC)
        if regions and s - regions[-1][1] <= config.REGION_MERGE_GAP_SEC:
            regions[-1][1] = max(regions[-1][1], e)
        else:
            regions.append([s, e])

    covered = sum(e - s for s, e in regions)
    return {
        "regions": regions,
        "decisions": decisions,
        "total_sec": total_duration,
        "speech_sec": round(covered, 1),
        "dropped_sec": round(total_duration - covered, 1),
    }
