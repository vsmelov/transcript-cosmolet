"""Детектор речи: где в записи люди действительно говорят.

Два локальных бесплатных слоя вместо платного черновика STT:
  1. Silero VAD — режет тишину. На ночной записи оставляет ~2.6% длительности.
  2. CED (AudioSet, 527 классов) — говорит, ЧТО в оставшемся куске: Speech,
     Conversation, Silence, Television, Rustling. VAD ловится на речеподобных
     шумах (диктофон в кармане), классификатор их отбраковывает.

Раньше эту роль играл whisper-черновик: он стоил денег, галлюцинировал на тишине
(«Спасибо», «Субтитры») и как детектор был заметно хуже — на 11 часах ночи он
пометил бы речью 86% записи вместо реальных 2.6%.
"""
from __future__ import annotations

import wave
from pathlib import Path

import numpy as np

import config

SR = 16000
# классы AudioSet, которые считаем человеческой речью
SPEECH_LABELS = {
    "Speech", "Conversation", "Narration, monologue", "Whispering",
    "Male speech, man speaking", "Female speech, woman speaking",
    "Child speech, kid speaking", "Speech synthesizer", "Shout", "Yell",
    "Chatter", "Babbling", "Crying, sobbing", "Laughter", "Singing",
}

_vad = None
_tagger = None


def _vad_engine():
    global _vad
    if _vad is None:
        import sherpa_onnx
        cfg = sherpa_onnx.VadModelConfig()
        cfg.silero_vad.model = config.VAD_MODEL
        cfg.silero_vad.threshold = config.VAD_THRESHOLD
        cfg.silero_vad.min_silence_duration = config.VAD_MIN_SILENCE
        cfg.silero_vad.min_speech_duration = config.VAD_MIN_SPEECH
        cfg.sample_rate = SR
        _vad = sherpa_onnx.VoiceActivityDetector(cfg, buffer_size_in_seconds=60)
    return _vad


def _tag_engine():
    global _tagger
    if _tagger is None:
        import sherpa_onnx
        _tagger = sherpa_onnx.AudioTagging(sherpa_onnx.AudioTaggingConfig(
            model=sherpa_onnx.AudioTaggingModelConfig(
                ced=f"{config.TAG_MODEL}/model.int8.onnx", num_threads=2),
            labels=f"{config.TAG_MODEL}/class_labels_indices.csv", top_k=5))
    return _tagger


def load_wav16k(path: Path) -> np.ndarray:
    with wave.open(str(path), "rb") as w:
        data = np.frombuffer(w.readframes(w.getnframes()), dtype=np.int16)
    return data.astype(np.float32) / 32768.0


def vad_spans(samples: np.ndarray) -> list[list[float]]:
    """Куски, которые Silero VAD считает речью."""
    vad = _vad_engine()
    vad.reset()
    out: list[list[float]] = []
    win = 512
    for i in range(0, len(samples) - win, win):
        vad.accept_waveform(samples[i:i + win])
        while not vad.empty():
            s = vad.front
            out.append([s.start / SR, (s.start + len(s.samples)) / SR])
            vad.pop()
    vad.flush()
    while not vad.empty():
        s = vad.front
        out.append([s.start / SR, (s.start + len(s.samples)) / SR])
        vad.pop()
    return out


def classify(samples: np.ndarray, start: float, end: float) -> tuple[float, list[dict]]:
    """(вероятность речи, топ-классы) для куска записи."""
    tagger = _tag_engine()
    a, b = int(start * SR), min(len(samples), int(end * SR))
    if b - a < int(0.4 * SR):
        return 0.0, []
    s = tagger.create_stream()
    s.accept_waveform(SR, samples[a:b])
    events = tagger.compute(s)
    tags = [{"name": e.name, "prob": round(float(e.prob), 3)} for e in events]
    speech = max((float(e.prob) for e in events if e.name in SPEECH_LABELS), default=0.0)
    return speech, tags


def detect(wav_path: Path, total_sec: float) -> dict:
    """Карта речи: регионы для дорогой модели + журнал решений для дебага в UI."""
    samples = load_wav16k(wav_path)
    spans = vad_spans(samples)

    decisions, kept = [], []
    for s, e in spans:
        speech, tags = classify(samples, s, e)
        ok = speech >= config.TAG_SPEECH_MIN
        decisions.append({"start": round(s, 2), "end": round(e, 2),
                          "speech": ok, "speech_prob": round(speech, 3),
                          "tags": tags[:3]})
        if ok:
            kept.append([s, e])

    # склейка близких кусков и поля, чтобы не срезать начало/конец фразы
    regions: list[list[float]] = []
    for s, e in kept:
        s, e = max(0.0, s - config.REGION_PAD_SEC), min(total_sec, e + config.REGION_PAD_SEC)
        if regions and s - regions[-1][1] <= config.REGION_MERGE_GAP_SEC:
            regions[-1][1] = max(regions[-1][1], e)
        else:
            regions.append([s, e])

    covered = sum(e - s for s, e in regions)
    vad_sec = sum(e - s for s, e in spans)
    return {
        "regions": regions,
        "decisions": decisions,
        "total_sec": total_sec,
        "vad_sec": round(vad_sec, 1),
        "speech_sec": round(covered, 1),
        "dropped_by_tagger_sec": round(vad_sec - sum(e - s for s, e in kept), 1),
        "dropped_sec": round(total_sec - covered, 1),
    }
