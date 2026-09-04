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
    cluster: int = -1           # наш кластер голоса (не метка провайдера)
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


def _merge_closest(cents: list[np.ndarray], groups: list[list[int]], join: float) -> None:
    """Агломеративная склейка по среднему направлению, пока ближайшая пара выше порога."""
    while len(groups) > 1:
        C = np.array(cents)
        M = C @ C.T
        np.fill_diagonal(M, -1.0)
        i, j = np.unravel_index(int(M.argmax()), M.shape)
        if M[i, j] < join:
            break
        groups[i] += groups[j]
        m = np.mean([cents[i], cents[j]], axis=0)
        cents[i] = m / (np.linalg.norm(m) or 1.0)
        groups.pop(j)
        cents.pop(j)


def _cluster_utterances(utts: list[Utterance], join: float | None = None) -> list[dict]:
    """Кто есть кто — решаем по голосу самих реплик, а не по меткам провайдера.

    Метка Scribe (speaker_0/1) — его гипотеза, и она бывает смешанной: на записи #19
    в одну метку попали два человека, центроид получился химерой (0.85 к одному и
    0.65 к другому), и оба получали чужое имя. Голоса же разделяются чисто: те же
    реплики, собранные в кластеры по звучанию, дают 0.91 при отрыве 0.33.

    Порог здесь НАМНОГО ниже, чем при склейке меток: усреднение по метке поднимает
    косинус, а у одиночных реплик того же человека он около 0.5. Поэтому опорой
    берём только длинные реплики, а коротышей приписываем к готовым кластерам.
    """
    anchors = [i for i, u in enumerate(utts)
               if u.vec is not None and u.end - u.start >= config.ANCHOR_MIN_SEC]
    if not anchors:
        return []
    groups = [[i] for i in anchors]
    cents = [utts[i].vec.copy() for i in anchors]
    _merge_closest(cents, groups, config.UTTER_JOIN if join is None else join)

    sec = lambda g: sum(utts[i].end - utts[i].start for i in g)
    # Кластер из пары случайных реплик — чаще шум, чем человек. Такие распускаем:
    # их реплики пойдут общим порядком, приписыванием к устоявшимся кластерам.
    keep = [k for k in sorted(range(len(groups)), key=lambda k: -sec(groups[k]))
            if sec(groups[k]) >= config.CLUSTER_MIN_SEC]
    if not keep:                        # короткая запись — оставляем самый крупный
        keep = [max(range(len(groups)), key=lambda k: sec(groups[k]))]

    out = [{"members": list(groups[k]), "cent": cents[k], "spoke": sec(groups[k]),
            "first": min(utts[i].start for i in groups[k])} for k in keep]
    out.sort(key=lambda c: c["first"])   # порядок появления в разговоре
    taken = {i for c in out for i in c["members"]}

    # Остальные реплики (короткие, распущенные) — к ближайшему по голосу кластеру.
    # Без вектора судить не по чему: тогда опираемся на метку провайдера, внутри
    # неё голос обычно один — это ровно та подсказка, ради которой она и нужна.
    by_label: dict[str, list[int]] = {}
    for ci, c in enumerate(out):
        for i in c["members"]:
            by_label.setdefault(utts[i].raw_speaker, []).append(ci)
    for i, u in enumerate(utts):
        if i in taken:
            continue
        ci = None
        if u.vec is not None:
            best = max(range(len(out)), key=lambda k: float(out[k]["cent"] @ u.vec))
            # Просто «ближайший» — плохой критерий: реплика чужого человека всё равно
            # к кому-то ближе всех, и кластер собирал в себя посторонние голоса с
            # похожестью 0.26. Не дотянула до порога — судим не по голосу.
            if float(out[best]["cent"] @ u.vec) >= config.ATTACH_MIN_COS:
                ci = best
        elif by_label.get(u.raw_speaker):
            # Метка провайдера — запасной вариант ТОЛЬКО когда вектора нет вовсе.
            # Если вектор есть и он не дотянул до порога, метке доверять тем более
            # нельзя: именно так в кластер заезжали реплики с похожестью 0.13.
            hits = by_label[u.raw_speaker]
            ci = max(set(hits), key=hits.count)
        if ci is None:
            continue          # ни голос, ни метка не говорят — оставляем неопознанной
        out[ci]["members"].append(i)     # центроид НЕ трогаем: вектор коротыша шумный
        out[ci]["spoke"] += u.end - u.start
    for ci, c in enumerate(out):
        for i in c["members"]:
            utts[i].cluster = ci
    return out


def _pick_join(utts: list[Utterance], base: dict[str, np.ndarray]) -> tuple[float, dict]:
    """Порог склейки подбираем под запись, а не берём константой.

    Единого правильного порога не существует: он зависит от акустики, числа людей
    и того, насколько похожи их голоса. На трёх реальных записях оптимум оказался
    разным — 0.60, 0.625 и 0.675, и разница в опознанной речи была кратной.

    Критерий — сколько МИНУТ РЕЧИ удалось уверенно назвать по базе эталонов. Чистая
    геометрия (силуэт) для этого не годится: она всегда предпочитает меньше
    кластеров и на записи #19 выбирала порог, где два человека слиты в одного.
    Дробление же одного человека на два кластера безвредно — оба получат его имя.
    """
    if not base:
        return config.UTTER_JOIN, {"reason": "база голосов пуста"}
    names = sorted(base)
    B = np.array([base[n] for n in names])
    best = (config.UTTER_JOIN, -1.0, 0)
    tried = []
    for k in range(int((config.JOIN_MAX - config.JOIN_MIN) / config.JOIN_STEP) + 1):
        th = round(config.JOIN_MIN + k * config.JOIN_STEP, 3)
        clusters = _cluster_utterances(list(utts), th)
        named = 0.0
        for c in clusters:
            sim = B @ c["cent"]
            order = np.argsort(-sim)
            gap = sim[order[0]] - (sim[order[1]] if len(order) > 1 else 0.0)
            if sim[order[0]] >= config.CONF_OK and gap >= config.TOP2_MARGIN:
                named += c["spoke"]
        tried.append({"join": th, "clusters": len(clusters), "named_sec": round(named)})
        # При равном результате выигрывает БОЛЬШИЙ порог. Ошибки диаризации не
        # равноценны: слипшихся людей приходится расклеивать руками по репликам, а
        # лишнее дробление закрывается парой подтверждений — оба куска получают
        # одно имя. Поэтому при сомнении дробим.
        if named > best[1] + 1e-9 or (abs(named - best[1]) <= 1e-9 and th > best[0]):
            best = (th, named, len(clusters))
    return best[0], {"picked": best[0], "named_sec": round(best[1]), "tried": tried}


def assign_speakers(utts: list[Utterance], src: Path, base: dict[str, np.ndarray],
                    forced_join: float | None = None) -> dict:
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

    # 2) кластеризация РЕПЛИК по голосу (метки провайдера — только подсказка).
    # Порог подбираем под запись: единого правильного не существует.
    if forced_join:
        join, join_info = forced_join, {"picked": forced_join, "source": "выбран вручную"}
    else:
        join, join_info = _pick_join(utts, base)
    clusters = _cluster_utterances(utts, join)

    # 3) имя кластеру: матч центроида против базы голосов
    names: list[str] = []
    cluster_ref: dict[str, np.ndarray] = {}
    summary = []
    anon = 0
    for c in clusters:
        cen = c["cent"]
        ranked = sorted(((n, float(v @ cen)) for n, v in base.items()), key=lambda kv: -kv[1])
        if ranked and ranked[0][1] >= config.CONF_OK:
            name, score = ranked[0]
        else:
            anon += 1
            name, score = f"S{anon}", (ranked[0][1] if ranked else 0.0)
        names.append(name)
        cluster_ref[name] = cen
        summary.append({"name": name, "spoke_sec": round(c["spoke"]),
                        "utterances": len(c["members"]), "base_cos": round(score, 3),
                        "labels": sorted({utts[i].raw_speaker for i in c["members"]})[:8],
                        "candidates": [{"name": n, "cos": round(s, 3)} for n, s in ranked[:3]]})

    # 4) per-реплика: confidence + разворот только у спорных
    ref = dict(base)
    ref.update({k: v for k, v in cluster_ref.items() if k.startswith("S")})
    for u in utts:
        cluster_name = names[u.cluster] if 0 <= u.cluster < len(names) else "S?"
        if u.vec is None:
            u.speaker, u.confidence, u.inherited = cluster_name, None, True
            continue
        ranked = sorted(((n, float(v @ u.vec)) for n, v in ref.items()), key=lambda kv: -kv[1])
        top = [{"name": n, "cos": round(s, 3)} for n, s in ranked[:3]]
        cl_cos = next((s for n, s in ranked if n == cluster_name), 0.0)
        margin = (ranked[0][1] - ranked[1][1]) if len(ranked) > 1 else 1.0
        mismatch = bool(ranked) and ranked[0][0] != cluster_name

        # Провайдер иногда сваливает чужую реплику в чужую метку целиком (перехват,
        # шутка чужим голосом). Если голос реплики уверенно ближе к ДРУГОМУ человеку,
        # чем к дефолту её кластера — переназначаем реплику, а не тащим её за меткой.
        reassigned = None
        if (mismatch and ranked[0][1] >= config.REASSIGN_MIN_COS
                and ranked[0][1] - cl_cos >= config.REASSIGN_MIN_GAP):
            reassigned = {"from": cluster_name, "to": ranked[0][0],
                          "cos": round(ranked[0][1], 3), "cluster_cos": round(cl_cos, 3)}
            cluster_name = ranked[0][0]
            cl_cos = ranked[0][1]
            mismatch = False

        u.speaker = cluster_name
        u.confidence = round(cl_cos, 3)
        if cl_cos < config.UTT_CONF_OK or margin < config.UTT_TOP2_MARGIN or mismatch:
            u.ambiguous = True
            u.detail = {"top": top,
                        "cluster_default": {"name": cluster_name, "cos": round(cl_cos, 3)},
                        "reason": ("cluster_mismatch" if mismatch else
                                   "top2_close" if margin < config.UTT_TOP2_MARGIN
                                   else "low_confidence")}
        if reassigned:
            u.detail = dict(u.detail or {}, reassigned=reassigned, top=top)
    return {"clusters": summary, "splits": split_count, "join": join_info}


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
