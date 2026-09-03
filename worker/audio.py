"""ffmpeg-обвязка: длительность, нарезка, конвертация в wav16k для эмбеддингов."""
from __future__ import annotations

import subprocess
import wave
from pathlib import Path

import numpy as np


def duration(path: Path) -> float:
    out = subprocess.run(
        ["ffprobe", "-v", "error", "-show_entries", "format=duration",
         "-of", "default=noprint_wrappers=1:nokey=1", str(path)],
        capture_output=True, text=True, check=True).stdout.strip()
    return float(out)


def cut(src: Path, start: float, dur: float, out: Path, reencode: bool = False) -> None:
    """Вырезка куска. stream copy по умолчанию; reencode для форматов, где copy ломается.

    Результат обязательно проверяем. Stream copy умеет завершиться с кодом 0 и при
    этом положить огрызок: на записи #13 кусок в 9 секунд вышел размером 879 байт,
    и ElevenLabs отвечал на него «File is corrupted» — этап падал, а причина была
    не в API. Не сошлась длительность — перерезаем с перекодированием.
    """
    cmd = ["ffmpeg", "-v", "error", "-y", "-ss", f"{start:.2f}", "-t", f"{dur:.2f}", "-i", str(src)]
    cmd += ["-c:a", "aac", "-b:a", "96k"] if reencode else ["-c", "copy"]
    cmd.append(str(out))
    p = subprocess.run(cmd, capture_output=True, text=True)
    if p.returncode != 0 and not reencode:
        # некоторые контейнеры (opus->m4a и т.п.) не переживают copy — пробуем перекодировать
        cut(src, start, dur, out.with_suffix(".m4a"), reencode=True)
        if out.suffix != ".m4a":
            out.with_suffix(".m4a").rename(out)
        return
    if p.returncode != 0:
        raise RuntimeError(f"ffmpeg cut: {p.stderr[:300]}")
    if not reencode and not _looks_complete(out, dur):
        cut(src, start, dur, out, reencode=True)
        if not _looks_complete(out, dur):
            raise RuntimeError(f"нарезка {out.name}: кусок не читается даже после перекодирования")


def _looks_complete(path: Path, want_sec: float) -> bool:
    """Похож ли вырезанный кусок на целый: файл на месте и звучит нужное время."""
    try:
        if not path.is_file() or path.stat().st_size < 2048:
            return False
        # допуск щедрый: stream copy режет по границам кадров и метит края неточно
        return duration(path) >= min(want_sec, 1.0) * 0.6
    except Exception:
        return False


def to_wav16k(src: Path, out: Path, start: float | None = None, dur: float | None = None) -> np.ndarray:
    """Конвертация (куска) в mono 16k и загрузка в float32 [-1..1]."""
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
