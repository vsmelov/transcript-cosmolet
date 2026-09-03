"""Локальные голосовые эмбеддинги (sherpa-onnx CAM++, 192 float) и матчинг по базе."""
from __future__ import annotations

import json
import tempfile
import threading
import wave
from pathlib import Path

import numpy as np

import audio as audio_mod
import config
import db

# Экстрактор — по экземпляру на поток: обработчиков несколько, а create_stream()
# у общего объекта из разных потоков даёт гонку.
_local = threading.local()


def extractor():
    if getattr(_local, "ext", None) is None:
        import sherpa_onnx
        _local.ext = sherpa_onnx.SpeakerEmbeddingExtractor(
            sherpa_onnx.SpeakerEmbeddingExtractorConfig(model=config.EMBED_MODEL, num_threads=config.ONNX_THREADS))
    return _local.ext


def _vector(samples: np.ndarray) -> np.ndarray:
    ext = extractor()
    s = ext.create_stream()
    s.accept_waveform(16000, np.ascontiguousarray(samples, dtype=np.float32))
    s.input_finished()
    v = np.array(ext.compute(s), dtype=np.float64)
    n = np.linalg.norm(v)
    return v / n if n else v


def embed_span(src: Path, start: float, dur: float) -> np.ndarray:
    """Нормированный вектор голоса для куска [start, start+dur] исходного файла."""
    with tempfile.TemporaryDirectory() as td:
        samples = audio_mod.to_wav16k(src, Path(td) / "x.wav", start, min(dur, config.EMBED_CLIP_MAX_SEC))
    return _vector(samples)


class AudioCache:
    """Файл один раз конвертируется в wav16k, дальше куски читаются срезами.

    Отдельный ffmpeg на каждую реплику (их сотни) превращал этап опознания
    в десятки минут. При этом весь массив в памяти держать нельзя: десять часов
    аудио — это 2+ ГБ, а обработок идёт несколько параллельно. Поэтому wav лежит
    во временном файле, а доступ к нему — через memmap: ОС сама подгружает нужные
    страницы, потребление памяти остаётся низким и предсказуемым.
    """

    def __init__(self, src: Path):
        self.sr = 16000
        self._tmp = tempfile.TemporaryDirectory(prefix="audiocache_")
        wav = Path(self._tmp.name) / "full.wav"
        audio_mod.to_wav16k(src, wav)
        with wave.open(str(wav), "rb") as w:
            frames = w.getnframes()
        # 44 байта — стандартный заголовок WAV, который пишет ffmpeg для PCM 16-бит
        self._data = (np.memmap(wav, dtype=np.int16, mode="r", offset=44, shape=(frames,))
                      if frames else np.zeros(0, dtype=np.int16))

    @property
    def samples(self) -> np.ndarray:
        return self._data

    def _slice(self, a: int, b: int) -> np.ndarray:
        return np.asarray(self._data[a:b], dtype=np.float32) / 32768.0

    def close(self) -> None:
        self._data = np.zeros(0, dtype=np.int16)
        self._tmp.cleanup()

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()

    def embed(self, start: float, dur: float) -> np.ndarray:
        a = max(0, int(start * self.sr))
        b = min(len(self._data), a + int(min(dur, config.EMBED_CLIP_MAX_SEC) * self.sr))
        if b - a < int(0.5 * self.sr):      # слишком короткий кусок — вектор бессмысленен
            raise ValueError("кусок короче 0.5с")
        return _vector(self._slice(a, b))

    def embed_spans(self, spans: list[tuple[float, float]]) -> np.ndarray:
        """Вектор по СКЛЕЕННЫМ интервалам речи (паузы и фон между словами выброшены).

        Слепое окно фиксированной длины может попасть в паузу или шум; интервалы
        слов от Scribe дают чистую речь, из которой вектор получается устойчивее.
        """
        parts = []
        total = 0.0
        for s, e in spans:
            a, b = max(0, int(s * self.sr)), min(len(self._data), int(e * self.sr))
            if b > a:
                parts.append(self._slice(a, b))
                total += (b - a) / self.sr
                if total >= config.EMBED_CLIP_MAX_SEC:
                    break
        if total < 0.8:
            raise ValueError("речи меньше 0.8с")
        return _vector(np.concatenate(parts))


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
