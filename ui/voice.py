"""Голосовые эмбеддинги для UI: ffmpeg-вырезка + sherpa-onnx CAM++ (192), матчинг по базе.

Логика повторяет worker/embed.py и worker/audio.py: кусок аудио -> wav 16k mono ->
SpeakerEmbeddingExtractor -> нормированный вектор. Эталон человека = среднее
нормированных векторов его АКТИВНЫХ сэмплов.
"""
from __future__ import annotations

import json
import os
import re
import subprocess
import tempfile
import threading
import wave
from pathlib import Path

import numpy as np

EMBED_MODEL = os.environ.get("EMBED_MODEL", "/models/campplus_zh_en_advanced.onnx")
CLIP_MAX_SEC = 45.0        # длиннее не эмбеддим (как EMBED_CLIP_MAX_SEC воркера)
CLIP_MIN_SEC = 0.5

_ex = None
_ex_lock = threading.Lock()   # sherpa-стрим не шарим между потоками threadpool'а FastAPI


def extractor():
    global _ex
    if _ex is None:
        import sherpa_onnx
        _ex = sherpa_onnx.SpeakerEmbeddingExtractor(
            sherpa_onnx.SpeakerEmbeddingExtractorConfig(model=EMBED_MODEL, num_threads=2))
    return _ex


# --- ffmpeg ------------------------------------------------------------------

def to_wav16k(src: Path, out: Path, start: float | None = None, dur: float | None = None):
    cmd = ["ffmpeg", "-v", "error", "-y"]
    if start is not None:
        cmd += ["-ss", f"{start:.2f}"]
    if dur is not None:
        cmd += ["-t", f"{dur:.2f}"]
    cmd += ["-i", str(src), "-ar", "16000", "-ac", "1", str(out)]
    p = subprocess.run(cmd, capture_output=True, text=True)
    if p.returncode != 0:
        raise RuntimeError(f"ffmpeg wav: {p.stderr[:300]}")
    with wave.open(str(out), "rb") as w:
        data = np.frombuffer(w.readframes(w.getnframes()), dtype=np.int16)
    return data.astype(np.float32) / 32768.0


def cut(src: Path, start: float, dur: float, out: Path, reencode: bool = False) -> None:
    """Вырезка куска: stream copy, при провале — перекодирование в aac."""
    cmd = ["ffmpeg", "-v", "error", "-y", "-ss", f"{start:.2f}", "-t", f"{dur:.2f}", "-i", str(src)]
    cmd += ["-c:a", "aac", "-b:a", "96k"] if reencode else ["-c", "copy"]
    cmd.append(str(out))
    p = subprocess.run(cmd, capture_output=True, text=True)
    if p.returncode != 0 or not out.exists() or out.stat().st_size < 512:
        if reencode:
            raise RuntimeError(f"ffmpeg cut: {p.stderr[:300]}")
        cut(src, start, dur, out, reencode=True)


def duration(path: Path) -> float | None:
    try:
        out = subprocess.run(
            ["ffprobe", "-v", "error", "-show_entries", "format=duration",
             "-of", "default=noprint_wrappers=1:nokey=1", str(path)],
            capture_output=True, text=True, check=True).stdout.strip()
        return float(out)
    except Exception:
        return None


# --- эмбеддинги --------------------------------------------------------------

def embed_span(src: Path, start: float, dur: float) -> np.ndarray:
    """Нормированный вектор голоса для куска [start, start+dur]."""
    with _ex_lock:
        ext = extractor()
        with tempfile.TemporaryDirectory() as td:
            samples = to_wav16k(Path(src), Path(td) / "x.wav", start, min(dur, CLIP_MAX_SEC))
        if samples.size < 1600:                      # <0.1 c — эмбеддить нечего
            raise RuntimeError("clip too short after decode")
        s = ext.create_stream()
        s.accept_waveform(16000, samples)
        s.input_finished()
        v = np.array(ext.compute(s), dtype=np.float64)
    return unit(v)


def unit(v: np.ndarray) -> np.ndarray:
    n = float(np.linalg.norm(v))
    return v / n if n else v


def parse_vec(raw) -> np.ndarray | None:
    """Вектор из БД: pgvector отдаётся строкой '[...]', либо это уже список."""
    if raw is None:
        return None
    try:
        v = np.array(json.loads(raw) if isinstance(raw, str) else list(raw), dtype=np.float64)
        return unit(v) if v.size else None
    except Exception:
        return None


def vec_literal(v: np.ndarray) -> str:
    return "[" + ",".join(f"{float(x):.6f}" for x in v) + "]"


def centroid(vecs) -> np.ndarray | None:
    vecs = [v for v in vecs if v is not None]
    if not vecs:
        return None
    return unit(np.mean(vecs, axis=0))


def cos(a, b) -> float:
    if a is None or b is None:
        return 0.0
    return float(np.dot(unit(np.asarray(a)), unit(np.asarray(b))))


def top_matches(v, refs: dict, k: int = 3) -> list:
    """top-k людей по косинусу к эталонам."""
    out = [{"name": n, "cos": round(cos(v, r), 4)} for n, r in refs.items()]
    out.sort(key=lambda x: x["cos"], reverse=True)
    return out[:k]


def coherences(samples) -> dict:
    """samples: [(id, vec|None, is_active)]. Косинус сэмпла к центроиду ОСТАЛЬНЫХ активных."""
    active = [(sid, v) for sid, v, act in samples if act and v is not None]
    out = {}
    for sid, v, _act in samples:
        others = [w for i, w in active if i != sid]
        c = centroid(others)
        out[sid] = round(cos(v, c), 4) if (v is not None and c is not None) else None
    return out


# --- прочее ------------------------------------------------------------------

_TR = {'а':'a','б':'b','в':'v','г':'g','д':'d','е':'e','ё':'e','ж':'zh','з':'z','и':'i','й':'j',
       'к':'k','л':'l','м':'m','н':'n','о':'o','п':'p','р':'r','с':'s','т':'t','у':'u','ф':'f',
       'х':'h','ц':'c','ч':'ch','ш':'sh','щ':'sch','ъ':'','ы':'y','ь':'','э':'e','ю':'yu','я':'ya'}


def slug(name: str) -> str:
    s = "".join(_TR.get(ch, ch) for ch in (name or "").lower())
    s = re.sub(r"[^a-z0-9]+", "-", s).strip("-")
    return s or "speaker"
