"""Стенд: какая модель голоса и какой способ сравнения точнее на НАШИХ данных.

Истина — таблица benchmark_labels: фрагменты, отобранные заранее без подсказки и
размеченные человеком ушами. Эталоны — активные speaker_samples. Для каждой пары
«модель × способ» считаем, у какой доли фрагментов первый кандидат совпал с меткой.

Способы сравнения:
  centroid   косинус к среднему вектору человека (как в проде)
  knn        ближайший отдельный эталон, а не среднее (устойчивее к разнородной базе)
  centered   центроид, но из всех векторов вычтен средний вектор ИХ записи —
             компенсация микрофона/комнаты; проверяем, помогает ли на честном наборе

Запуск в контейнере воркера:
  python /app/bench_speakers.py                  # все модели из /models
  python /app/bench_speakers.py --models campplus_zh_en_advanced.onnx
"""
from __future__ import annotations

import argparse
import collections
import glob
import json
import os
import sys
import tempfile
import time
from pathlib import Path

import numpy as np
import sherpa_onnx

import audio as audio_mod
import db

# «?» — человек сам не разобрал, «[noise]» — не речь, «[skip]» — размечать не стал,
# «[stranger]» — незнакомый голос; последний меряется отдельно (ложные имена)
SKIP_LABELS = {"?", "[noise]", "[skip]"}


def load_truth():
    rows = db.q("""SELECT l.segment_id, l.speaker_name, l.condition, s.recording_id,
                          s.start_sec, s.end_sec, s.end_sec - s.start_sec AS dur, r.audio_path
                     FROM benchmark_labels l JOIN segments s ON s.id = l.segment_id
                     JOIN recordings r ON r.id = s.recording_id
                    WHERE r.audio_path IS NOT NULL""") or []
    return [r for r in rows if r[1] not in SKIP_LABELS]


def load_refs():
    rows = db.q("""SELECT sp.name, ss.path, ss.source, ss.duration_sec FROM speakers sp
                     JOIN speaker_samples ss ON ss.speaker_id = sp.id
                    WHERE ss.is_active AND ss.path IS NOT NULL""") or []
    return [r for r in rows if r[1] and Path(r[1]).is_file()]


def extractor(model_path: str):
    return sherpa_onnx.SpeakerEmbeddingExtractor(
        sherpa_onnx.SpeakerEmbeddingExtractorConfig(model=model_path, num_threads=4))


def embed_wav(ext, samples) -> np.ndarray:
    st = ext.create_stream()
    st.accept_waveform(16000, np.ascontiguousarray(samples, dtype=np.float32))
    st.input_finished()
    v = np.array(ext.compute(st), dtype=np.float64)
    return v / (np.linalg.norm(v) or 1.0)


def embed_span(ext, path: Path, start: float, dur: float) -> np.ndarray:
    with tempfile.TemporaryDirectory() as td:
        samples = audio_mod.to_wav16k(path, Path(td) / "x.wav", start, min(dur, 45.0))
    return embed_wav(ext, samples)


def embed_file(ext, path: Path) -> np.ndarray:
    with tempfile.TemporaryDirectory() as td:
        samples = audio_mod.to_wav16k(path, Path(td) / "x.wav")
    return embed_wav(ext, samples)


def evaluate(model_path: str, truth, refs, rec_mean_cache: dict):
    name = Path(model_path).name
    ext = extractor(model_path)
    t0 = time.time()

    # эталоны: по имени -> список векторов
    ref_vecs: dict[str, list] = collections.defaultdict(list)
    for who, path, _, _ in refs:
        try:
            ref_vecs[who].append(embed_file(ext, Path(path)))
        except Exception:
            pass
    names = sorted(ref_vecs)
    cent = np.array([np.mean(ref_vecs[n], axis=0) / (np.linalg.norm(np.mean(ref_vecs[n], axis=0)) or 1)
                     for n in names])
    all_refs = [(n, v) for n in names for v in ref_vecs[n]]

    # тестовые фрагменты
    test = []
    for sid, who, cond, rid, st, en, dur, ap in truth:
        if who not in ref_vecs:
            continue                      # эталонов у этого человека нет — сравнивать не с чем
        try:
            v = embed_span(ext, Path(ap), float(st), float(dur))
        except Exception:
            continue
        test.append((who, cond, float(dur), rid, v))

    # средний вектор записи для центрирования — по случайным 40 фрагментам записи
    def rec_mean(rid):
        key = (name, rid)
        if key not in rec_mean_cache:
            segs = db.q("""SELECT start_sec, end_sec FROM segments WHERE recording_id=%s
                            AND end_sec-start_sec >= 2 ORDER BY random() LIMIT 40""", rid) or []
            ap = db.q1("SELECT audio_path FROM recordings WHERE id=%s", rid)[0]
            vs = []
            for st, en in segs:
                try:
                    vs.append(embed_span(ext, Path(ap), float(st), float(en) - float(st)))
                except Exception:
                    pass
            rec_mean_cache[key] = np.mean(vs, axis=0) if vs else np.zeros_like(cent[0])
        return rec_mean_cache[key]

    def score(method):
        hits = collections.Counter(); tot = collections.Counter()
        for who, cond, dur, rid, v in test:
            if method == "centroid":
                guess = names[int(np.argmax(cent @ v))]
            elif method == "knn":
                guess = max(all_refs, key=lambda nv: float(nv[1] @ v))[0]
            else:  # centered
                m = rec_mean(rid)
                vv = v - m; vv /= (np.linalg.norm(vv) or 1)
                cc = cent - m; cc /= np.linalg.norm(cc, axis=1, keepdims=True)
                guess = names[int(np.argmax(cc @ vv))]
            ok = guess == who
            for key in ("всего", f"усл:{cond}", f"длина:{'кор' if dur < 3 else 'ср' if dur < 8 else 'длин'}"):
                tot[key] += 1; hits[key] += ok
        return {k: (hits[k], tot[k]) for k in tot}

    out = {m: score(m) for m in ("centroid", "knn", "centered")}
    return name, len(test), time.time() - t0, out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", nargs="*", help="имена .onnx в /models (по умолчанию все)")
    args = ap.parse_args()
    models = args.models or [Path(p).name for p in glob.glob("/models/*.onnx")
                             if "vad" not in Path(p).name.lower()]
    truth = load_truth(); refs = load_refs()
    print(f"размечено фрагментов: {len(truth)} · эталонов с аудио: {len(refs)}")
    if not truth:
        print("нечего мерить: разметьте фрагменты на вкладке «Бенчмарк»"); return
    cache: dict = {}
    for m in models:
        try:
            name, n, el, res = evaluate(f"/models/{m}", truth, refs, cache)
        except Exception as e:
            print(f"{m}: не запустилась — {str(e)[:120]}"); continue
        print(f"\n=== {name} · фрагментов {n} · {el:.0f}с")
        for method, r in res.items():
            line = " · ".join(f"{k} {h}/{t} ({h/t*100:.0f}%)" for k, (h, t) in r.items())
            print(f"  {method:9} {line}")


if __name__ == "__main__":
    main()
