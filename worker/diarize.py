"""Опознание спикеров: анонимная диаризация Scribe -> наши эмбеддинги -> имена + confidence.

Три эшелона (см. SCHEMA.md):
  1) голос: кластеризация speaker_id-ов по косинусу, матч кластеров против базы;
  2) текст: спорные реплики судит дешёвая LLM по контексту диалога;
  3) человек: остаток уходит в conflicts -> ручной разбор в UI.
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

import clients
import config
import embed as embed_mod


@dataclass
class Utterance:
    start: float
    end: float
    text: str
    raw_speaker: str            # speaker_0/1... от Scribe (локальный для куска)
    words: list = field(default_factory=list)
    asr_logprob: float | None = None
    vec: np.ndarray | None = None
    speaker: str = "S?"
    confidence: float | None = None
    inherited: bool = False
    ambiguous: bool = False
    detail: dict | None = None


def utterances_from_scribe(resp: dict, offset: float, chunk_tag: str) -> list[Utterance]:
    """Слова Scribe -> реплики: подряд идущие слова одного speaker_id с паузами < UTTER_GAP."""
    out: list[Utterance] = []
    cur: Utterance | None = None
    for w in resp.get("words") or []:
        if w.get("type") != "word":
            continue
        spk = f"{chunk_tag}:{w.get('speaker_id') or 'unknown'}"
        st, en = float(w.get("start", 0)) + offset, float(w.get("end", 0)) + offset
        # таймкоды слов у Scribe локальны для куска — сохраняем ГЛОБАЛЬНЫЕ, иначе
        # вырезка аудио по словам (сплиттер, эмбеддинги) уедет в другое место файла
        gw = dict(w, start=st, end=en)
        if cur and cur.raw_speaker == spk and st - cur.end <= config.UTTER_GAP_SEC:
            cur.end = en
            cur.text += (" " if not cur.text.endswith(" ") else "") + w.get("text", "")
            cur.words.append(gw)
        else:
            if cur:
                out.append(cur)
            cur = Utterance(start=st, end=en, text=w.get("text", ""), raw_speaker=spk, words=[gw])
    if cur:
        out.append(cur)
    for u in out:
        lps = [float(w["logprob"]) for w in u.words if w.get("logprob") is not None]
        u.asr_logprob = round(sum(lps) / len(lps), 4) if lps else None
        u.text = u.text.strip()
    return out


def _speech_spans(words: list) -> list[tuple[float, float]]:
    return [(float(w["start"]), float(w["end"])) for w in words
            if w.get("start") is not None and w.get("end") is not None]


def _speech_sec(words: list) -> float:
    return sum(e - s for s, e in _speech_spans(words))


def split_mixed(u: Utterance, cache, depth: int = 0) -> list[Utterance]:
    """Делит реплику, если внутри неё сменился говорящий (ошибка диаризации провайдера).

    Точки разреза — границы слов (не слепая сетка): для каждой считаем вектор левой и
    правой половины по СКЛЕЕННОЙ речи, ищем минимум косинуса. Если минимум ниже порога
    и обе половины содержат достаточно речи — режем и проверяем половины рекурсивно.
    """
    words = u.words or []
    if depth >= config.SPLIT_MAX_DEPTH or len(words) < 8:
        return [u]
    if u.end - u.start < config.SPLIT_MIN_SEC or _speech_sec(words) < 2 * config.SPLIT_MIN_SIDE_SEC:
        return [u]

    spans = _speech_spans(words)
    best_i, best_cos = -1, 1.0
    for i in range(3, len(words) - 3):
        left, right = spans[:i], spans[i:]
        if (sum(e - s for s, e in left) < config.SPLIT_MIN_SIDE_SEC or
                sum(e - s for s, e in right) < config.SPLIT_MIN_SIDE_SEC):
            continue
        try:
            cos = float(cache.embed_spans(left) @ cache.embed_spans(right))
        except Exception:
            continue
        if cos < best_cos:
            best_i, best_cos = i, cos

    if best_i < 0 or best_cos > config.SPLIT_MAX_COS:
        return [u]

    lw, rw = words[:best_i], words[best_i:]
    mk = lambda ws, tag: Utterance(
        start=float(ws[0]["start"]), end=float(ws[-1]["end"]),
        text=" ".join(w.get("text", "") for w in ws).strip(),
        raw_speaker=f"{u.raw_speaker}#{tag}", words=ws,
        asr_logprob=u.asr_logprob)
    left_u, right_u = mk(lw, f"a{depth}"), mk(rw, f"b{depth}")
    for part in (left_u, right_u):
        part.detail = {"split_from": u.raw_speaker, "split_cos": round(best_cos, 3)}
    return split_mixed(left_u, cache, depth + 1) + split_mixed(right_u, cache, depth + 1)


def assign_speakers(utts: list[Utterance], src: Path, base: dict[str, np.ndarray]) -> dict:
    """Эмбеддинги -> кластеры -> имена. Возвращает сводку кластеров для артефакта."""
    # 0) чиним склейки провайдера: длинные реплики, внутри которых сменился голос
    cache = embed_mod.AudioCache(src)
    split_count = 0
    fixed: list[Utterance] = []
    for u in utts:
        parts = split_mixed(u, cache)
        split_count += len(parts) - 1
        fixed.extend(parts)
    utts[:] = sorted(fixed, key=lambda x: x.start)

    # 1) вектор каждой достаточно длинной реплики (по склеенной речи, без пауз)
    for u in utts:
        if u.end - u.start < config.MIN_EMBED_SEC:
            continue
        try:
            u.vec = (cache.embed_spans(_speech_spans(u.words)) if u.words
                     else cache.embed(u.start, u.end - u.start))
        except Exception:
            u.vec = None

    # 2) кластеризация raw_speaker-меток (они локальны для куска) по центроидам
    by_label: dict[str, list[np.ndarray]] = {}
    spoke: dict[str, float] = {}
    first: dict[str, float] = {}
    for u in utts:
        spoke[u.raw_speaker] = spoke.get(u.raw_speaker, 0.0) + (u.end - u.start)
        first.setdefault(u.raw_speaker, u.start)
        if u.vec is not None:
            by_label.setdefault(u.raw_speaker, []).append(u.vec)

    label_vec: dict[str, np.ndarray] = {}
    for lab, vecs in by_label.items():
        m = np.mean(vecs, axis=0)
        n = np.linalg.norm(m)
        label_vec[lab] = m / n if n else m

    # Фаза 1: кластеризуем только метки с надёжным вектором (говорили достаточно долго).
    # Короткие обрывки — особенно осколки от сплиттера — имеют шумный вектор и, попадая
    # в общую кластеризацию, плодят микрокластеры по 4-10 секунд вместо реальных людей.
    solid = [l for l in label_vec if spoke.get(l, 0) >= config.CLUSTER_MIN_SEC]
    weak = [l for l in label_vec if l not in solid]
    clusters: list[dict] = []
    for lab in sorted(solid, key=lambda l: -spoke.get(l, 0)):
        v = label_vec[lab]
        best_i, best_cos = -1, 0.0
        for i, c in enumerate(clusters):
            cen = c["sum"] / np.linalg.norm(c["sum"])
            cos = float(cen @ v)
            if cos > best_cos:
                best_i, best_cos = i, cos
        if best_i >= 0 and best_cos >= config.CLUSTER_JOIN:
            c = clusters[best_i]
            c["labels"].append(lab)
            c["sum"] = c["sum"] + v
            c["spoke"] += spoke.get(lab, 0.0)
            c["first"] = min(c["first"], first.get(lab, 0.0))
        else:
            clusters.append({"labels": [lab], "sum": v.copy(),
                             "spoke": spoke.get(lab, 0.0), "first": first.get(lab, 0.0)})

    # Фаза 2: короткие метки приписываем к ближайшему сформированному кластеру
    # (без порога — просто ближайший); низкая уверенность отметится на уровне реплик.
    for lab in sorted(weak, key=lambda l: -spoke.get(l, 0)):
        v = label_vec[lab]
        if not clusters:
            clusters.append({"labels": [lab], "sum": v.copy(),
                             "spoke": spoke.get(lab, 0.0), "first": first.get(lab, 0.0)})
            continue
        best_i = max(range(len(clusters)),
                     key=lambda i: float((clusters[i]["sum"] / np.linalg.norm(clusters[i]["sum"])) @ v))
        c = clusters[best_i]
        c["labels"].append(lab)
        c["spoke"] += spoke.get(lab, 0.0)     # вектор кластера НЕ обновляем шумным обрывком
        c["first"] = min(c["first"], first.get(lab, 0.0))

    # 3) имя кластеру: матч центроида против базы голосов
    label_to_name: dict[str, str] = {}
    cluster_ref: dict[str, np.ndarray] = {}
    summary = []
    anon = 0
    for c in sorted(clusters, key=lambda c: c["first"]):
        cen = c["sum"] / np.linalg.norm(c["sum"])
        ranked = sorted(((n, float(v @ cen)) for n, v in base.items()), key=lambda kv: -kv[1])
        if ranked and ranked[0][1] >= config.CONF_OK:
            name, score = ranked[0]
        else:
            anon += 1
            name, score = f"S{anon}", (ranked[0][1] if ranked else 0.0)
        for lab in c["labels"]:
            label_to_name[lab] = name
        cluster_ref[name] = cen
        summary.append({"name": name, "spoke_sec": round(c["spoke"]),
                        "labels": c["labels"], "base_cos": round(score, 3),
                        "candidates": [{"name": n, "cos": round(s, 3)} for n, s in ranked[:3]]})

    # 4) per-реплика: confidence + разворот только у спорных
    ref = dict(base)
    ref.update({k: v for k, v in cluster_ref.items() if k.startswith("S")})
    for u in utts:
        cluster_name = label_to_name.get(u.raw_speaker, "S?")
        if u.vec is None:
            u.speaker, u.confidence, u.inherited = cluster_name, None, True
            continue
        ranked = sorted(((n, float(v @ u.vec)) for n, v in ref.items()), key=lambda kv: -kv[1])
        top = [{"name": n, "cos": round(s, 3)} for n, s in ranked[:3]]
        cl_cos = next((s for n, s in ranked if n == cluster_name), 0.0)
        u.speaker = cluster_name
        u.confidence = round(cl_cos, 3)
        margin = (ranked[0][1] - ranked[1][1]) if len(ranked) > 1 else 1.0
        mismatch = bool(ranked) and ranked[0][0] != cluster_name
        if cl_cos < config.CONF_OK or margin < config.TOP2_MARGIN or mismatch:
            u.ambiguous = True
            u.detail = {"top": top,
                        "cluster_default": {"name": cluster_name, "cos": round(cl_cos, 3)},
                        "reason": ("cluster_mismatch" if mismatch else
                                   "top2_close" if margin < config.TOP2_MARGIN else "low_confidence")}
    return {"clusters": summary, "labels": label_to_name, "splits": split_count}


JUDGE_PROMPT = """Диалог, транскрипт с автоматически определёнными спикерами. У одной реплики
определение спикера ненадёжно. Определи по контексту, кто её произнёс.

Контекст (реплики до и после, с их спикерами):
{context}

Спорная реплика [{start}]: "{text}"
Кандидаты по голосу: {candidates}

Опирайся на: кто говорит о себе в первом лице, к кому обращаются по имени, чья роль в
разговоре (ведущий/гость), логика чередования. Верни JSON:
{{"speaker": "<имя из кандидатов>", "confidence": <0..1>, "why": "<кратко>"}}
Если по тексту определить нельзя — {{"speaker": null, "confidence": 0, "why": "..."}}"""


def judge_conflicts(utts: list[Utterance], limit: int = 25) -> tuple[int, float]:
    """Эшелон 2: текст-судья по спорным репликам. (сколько закрыто, потрачено $)."""
    idx = [i for i, u in enumerate(utts) if u.ambiguous]
    total_cost, closed = 0.0, 0
    for i in idx[:limit]:
        u = utts[i]
        ctx = []
        for j in range(max(0, i - 3), min(len(utts), i + 4)):
            if j == i:
                continue
            ctx.append(f"[{utts[j].speaker}] {utts[j].text[:160]}")
        cands = ", ".join(f"{c['name']} ({c['cos']})" for c in (u.detail or {}).get("top", []))
        try:
            verdict, cost = clients.judge(JUDGE_PROMPT.format(
                context="\n".join(ctx), start=f"{u.start:.0f}s", text=u.text[:400], candidates=cands))
        except Exception:
            continue
        total_cost += cost
        u.detail = dict(u.detail or {}, judge=verdict)
        name = verdict.get("speaker")
        if name and float(verdict.get("confidence") or 0) >= 0.7:
            agrees = name == u.speaker
            u.detail["resolution"] = f"judge:{name} ({'подтвердил' if agrees else 'переназначил'})"
            u.speaker = name
            if agrees:
                u.ambiguous = False   # голос и текст сошлись — конфликт закрыт
                closed += 1
    return closed, total_cost
