"""Сравнение сегментации pyannote с нашим конвейером на одной записи.

Мерить надо против того, что человек подтвердил ушами, а не против ощущений.
Запись #49 для этого удобна: там разобраны Владимир, Яна и «алена психотерапевт»,
и известно, что они говорили примерно поровну.

Метрика — доля времени, на которой мы и pyannote согласны о том, ГДЕ ГРАНИЦА между
говорящими: для каждой секунды смотрим, кто говорит по нашей разметке и кто по
pyannote, и строим таблицу соответствия. Идеальная диаризация даёт однозначное
отображение «наш спикер -> их спикер»; смешение видно сразу.
"""
from __future__ import annotations

import collections
import sys
import tempfile
import time
from pathlib import Path

import numpy as np
import sherpa_onnx

import audio as audio_mod
import db

SEG_MODEL = "/models/sherpa-onnx-pyannote-segmentation-3-0/model.onnx"
EMB_MODEL = "/models/campplus_zh_en_advanced.onnx"


def diarize(samples, num_clusters: int = -1, threshold: float = 0.5):
    cfg = sherpa_onnx.OfflineSpeakerDiarizationConfig(
        segmentation=sherpa_onnx.OfflineSpeakerSegmentationModelConfig(
            pyannote=sherpa_onnx.OfflineSpeakerSegmentationPyannoteModelConfig(model=SEG_MODEL),
            num_threads=4),
        embedding=sherpa_onnx.SpeakerEmbeddingExtractorConfig(model=EMB_MODEL, num_threads=4),
        clustering=sherpa_onnx.FastClusteringConfig(num_clusters=num_clusters, threshold=threshold),
        min_duration_on=0.3, min_duration_off=0.5)
    sd = sherpa_onnx.OfflineSpeakerDiarization(cfg)
    t0 = time.time()
    res = sd.process(samples).sort_by_start_time()
    return res, time.time() - t0


def run(rec_id: int, num_clusters: int = -1):
    path = Path(db.q1("SELECT audio_path FROM recordings WHERE id=%s", rec_id)[0])
    dur = float(db.q1("SELECT duration_sec FROM recordings WHERE id=%s", rec_id)[0])
    with tempfile.TemporaryDirectory() as td:
        samples = audio_mod.to_wav16k(path, Path(td) / "x.wav")
    res, el = diarize(samples, num_clusters)
    print(f"pyannote: {dur/60:.0f} мин аудио за {el/60:.1f} мин = {dur/el:.1f}x реального времени")

    # раскладываем обе разметки по секундной сетке и сравниваем
    grid = int(dur)
    theirs = ["-"] * grid
    for s in res:
        for t in range(int(s.start), min(int(s.end), grid)):
            theirs[t] = f"P{s.speaker}"
    ours = ["-"] * grid
    for name, st, en in db.q("""SELECT speaker_name, start_sec, end_sec FROM segments
                                 WHERE recording_id=%s ORDER BY start_sec""", rec_id) or []:
        for t in range(int(float(st)), min(int(float(en)), grid)):
            ours[t] = name

    both = [(o, p) for o, p in zip(ours, theirs) if o != "-" and p != "-"]
    print(f"речь нашли оба: {len(both)/60:.1f} мин из {dur/60:.0f}")
    print(f"  только мы: {sum(1 for o,p in zip(ours,theirs) if o!='-' and p=='-')/60:.1f} мин"
          f" · только pyannote: {sum(1 for o,p in zip(ours,theirs) if o=='-' and p!='-')/60:.1f} мин")

    table = collections.Counter(both)
    our_names = sorted({o for o, _ in both}, key=lambda n: -sum(v for (o, _), v in table.items() if o == n))
    print()
    print("кто с кем совпал (секунды):")
    for n in our_names[:6]:
        row = sorted(((v, p) for (o, p), v in table.items() if o == n), reverse=True)[:3]
        total = sum(v for v, _ in row)
        line = " · ".join(f"{p} {v/60:.1f}м" for v, p in row)
        best = row[0][0] / max(sum(v for (o, _), v in table.items() if o == n), 1) * 100 if row else 0
        print(f"   {n:26} -> {line}   (в один их кластер попало {best:.0f}%)")


if __name__ == "__main__":
    run(int(sys.argv[1]) if len(sys.argv) > 1 else 49,
        int(sys.argv[2]) if len(sys.argv) > 2 else -1)
