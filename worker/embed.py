"""Локальные голосовые эмбеддинги (sherpa-onnx CAM++, 192 float) и матчинг по базе."""
from __future__ import annotations

import json
import tempfile
from pathlib import Path

import numpy as np

import audio as audio_mod
import config
import db

_extractor = None


def extractor():
    global _extractor
    if _extractor is None:
        import sherpa_onnx
        _extractor = sherpa_onnx.SpeakerEmbeddingExtractor(
            sherpa_onnx.SpeakerEmbeddingExtractorConfig(model=config.EMBED_MODEL, num_threads=2))
    return _extractor


def embed_span(src: Path, start: float, dur: float) -> np.ndarray:
    """Нормированный вектор голоса для куска [start, start+dur] исходного файла."""
    ext = extractor()
    with tempfile.TemporaryDirectory() as td:
        samples = audio_mod.to_wav16k(src, Path(td) / "x.wav", start, min(dur, config.EMBED_CLIP_MAX_SEC))
    s = ext.create_stream()
    s.accept_waveform(16000, samples)
    s.input_finished()
    v = np.array(ext.compute(s), dtype=np.float64)
    n = np.linalg.norm(v)
    return v / n if n else v


def known_speakers() -> dict[str, np.ndarray]:
    """Эталон каждого человека: среднее нормированных векторов всех его сэмплов."""
    rows = db.q("""SELECT s.name, ss.embedding FROM speakers s
                   JOIN speaker_samples ss ON ss.speaker_id = s.id
                   WHERE ss.embedding IS NOT NULL""")
    acc: dict[str, list[np.ndarray]] = {}
    for name, emb in rows:
        acc.setdefault(name, []).append(np.array(json.loads(emb) if isinstance(emb, str) else emb))
    out = {}
    for name, vecs in acc.items():
        m = np.mean(vecs, axis=0)
        n = np.linalg.norm(m)
        out[name] = m / n if n else m
    return out
