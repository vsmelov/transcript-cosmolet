"""Эксперимент: сколько речи в ночной записи и чем её честно найти.

Проверяем Silero VAD (локально, бесплатно) на 11-часовой записи с диктофоном в кармане:
  - сколько времени он считает речью,
  - устойчивы ли эти куски по голосу (шорох ткани даёт нестабильный вектор),
  - похожи ли они на известных людей.

Запуск ВНУТРИ контейнера воркера:
    docker compose exec worker python //data/vad_experiment.py [путь] [минут]
"""
import json
import sys
import time
import wave
from pathlib import Path

sys.path.insert(0, "/app")
import numpy as np  # noqa: E402

import audio as audio_mod  # noqa: E402
import embed as embed_mod  # noqa: E402

SRC = Path(sys.argv[1] if len(sys.argv) > 1 else "/data/experiments/night.mp3")
LIMIT_MIN = float(sys.argv[2]) if len(sys.argv) > 2 else 0     # 0 = весь файл
VAD_MODEL = "/models/silero_vad.onnx"
SR = 16000

sys.stdout.reconfigure(encoding="utf-8", errors="replace")


def load_audio(path: Path, limit_min: float) -> np.ndarray:
    tmp = Path("/tmp/night16k.wav")
    print(f"[1/4] декодирую {path.name} в wav16k...", flush=True)
    t0 = time.time()
    audio_mod.to_wav16k(path, tmp, None, limit_min * 60 if limit_min else None)
    with wave.open(str(tmp), "rb") as w:
        data = np.frombuffer(w.readframes(w.getnframes()), dtype=np.int16)
    print(f"      готово за {time.time()-t0:.0f}с, {len(data)/SR/3600:.2f} ч аудио", flush=True)
    return data.astype(np.float32) / 32768.0


def run_vad(samples: np.ndarray) -> list[tuple[float, float]]:
    import sherpa_onnx
    cfg = sherpa_onnx.VadModelConfig()
    cfg.silero_vad.model = VAD_MODEL
    cfg.silero_vad.threshold = 0.5
    cfg.silero_vad.min_silence_duration = 0.8   # пауза, после которой считаем речь законченной
    cfg.silero_vad.min_speech_duration = 0.25
    cfg.sample_rate = SR
    vad = sherpa_onnx.VoiceActivityDetector(cfg, buffer_size_in_seconds=60)

    print("[2/4] гоняю Silero VAD...", flush=True)
    t0 = time.time()
    win, segs = 512, []
    for i in range(0, len(samples) - win, win):
        vad.accept_waveform(samples[i:i + win])
        while not vad.empty():
            s = vad.front
            segs.append((s.start / SR, (s.start + len(s.samples)) / SR))
            vad.pop()
    vad.flush()
    while not vad.empty():
        s = vad.front
        segs.append((s.start / SR, (s.start + len(s.samples)) / SR))
        vad.pop()
    print(f"      готово за {time.time()-t0:.0f}с, кусков: {len(segs)}", flush=True)
    return segs


def main() -> None:
    samples = load_audio(SRC, LIMIT_MIN)
    total = len(samples) / SR
    segs = run_vad(samples)
    speech = sum(e - s for s, e in segs)
    print()
    print(f"=== VAD: речь {speech:.0f}с из {total:.0f}с = {speech/total*100:.1f}% записи")
    print(f"    экономия на Scribe: ${(total-speech)/3600*0.22:.2f} "
          f"(весь файл стоил бы ${total/3600*0.22:.2f}, по VAD — ${speech/3600*0.22:.2f})")

    # длинные куски — кандидаты в реальную речь; проверяем их голосом
    print("\n[3/4] проверяю куски по голосу (эмбеддинги)...", flush=True)
    base = embed_mod.known_speakers()
    long_segs = sorted([x for x in segs if x[1] - x[0] >= 2.0],
                       key=lambda x: -(x[1] - x[0]))[:40]
    rows = []
    for s, e in long_segs:
        a, b = int(s * SR), int(min(e, s + 20) * SR)
        try:
            v = embed_mod._vector(samples[a:b])
        except Exception:
            continue
        # устойчивость голоса внутри куска: шорох даёт разъезжающиеся половины
        half = (a + b) // 2
        try:
            stab = float(embed_mod._vector(samples[a:half]) @ embed_mod._vector(samples[half:b]))
        except Exception:
            stab = float("nan")
        top = sorted(((n, float(x @ v)) for n, x in base.items()), key=lambda kv: -kv[1])[:1]
        rows.append((s, e - s, stab, top[0] if top else ("-", 0.0)))

    print(f"{'начало':>9} {'длит':>6} {'устойч':>7} {'ближайший из базы':>28}")
    for s, d, stab, (nm, cos) in rows[:25]:
        h, rem = divmod(int(s), 3600)
        m, sec = divmod(rem, 60)
        print(f"{h:02d}:{m:02d}:{sec:02d} {d:6.1f}с {stab:7.2f} {nm:>22} {cos:5.2f}")

    known = [r for r in rows if r[3][1] >= 0.5]
    stable = [r for r in rows if r[2] == r[2] and r[2] >= 0.6]
    print(f"\n[4/4] из {len(rows)} длинных кусков: похожи на известных {len(known)}, "
          f"устойчивы по голосу {len(stable)}")
    Path("/data/experiments/vad_result.json").write_text(json.dumps({
        "total_sec": total, "speech_sec": speech, "segments": len(segs),
        "checked": [{"start": r[0], "dur": r[1], "stability": r[2],
                     "top": r[3][0], "cos": r[3][1]} for r in rows],
    }, ensure_ascii=False, indent=2), encoding="utf-8")
    print("результат: /data/experiments/vad_result.json")


if __name__ == "__main__":
    main()
