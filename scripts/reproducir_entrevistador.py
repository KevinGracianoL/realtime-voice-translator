"""Simula al entrevistador remoto para la demo: reproduce el WAV en CABLE Input.

Así el flujo incoming lo captura de CABLE Output como si viniera de Meet/Zoom
(configuración real del ADR-015: Meet con SALIDA de audio en CABLE Input).

Reproduce `--veces` (default 3) con pausa entre turnos y TERMINA solo; imprime
progreso. Para repetir en bucle (video): `--veces 0` (infinito, Ctrl+C).

Uso: python scripts/reproducir_entrevistador.py [--veces 3]
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import soundfile as sf


def main(argv: list[str] | None = None) -> None:  # pragma: no cover - máquina
    parser = argparse.ArgumentParser()
    parser.add_argument("--veces", type=int, default=3, help="0 = bucle infinito")
    parser.add_argument(
        "--pausa",
        type=float,
        default=2.0,
        help="silencio entre turnos en segundos (el VAD de Silero une turnos "
        "con pausas < ~1 s en una sola locución y nunca cierra el turno)",
    )
    args = parser.parse_args(argv)
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", line_buffering=True)

    import numpy as np
    import pyaudio

    wav = Path(__file__).resolve().parent / "audio" / "entrevistador_en.wav"
    muestras, sr = sf.read(str(wav), dtype="float32")
    duracion_s = len(muestras) / sr

    pa = pyaudio.PyAudio()
    try:
        indice = next(
            i
            for i in range(pa.get_device_count())
            if "CABLE Input" in str(pa.get_device_info_by_index(i)["name"])
            and pa.get_device_info_by_index(i)["maxOutputChannels"] == 2
        )
    except StopIteration as exc:
        raise RuntimeError("VB-CABLE no disponible (CABLE Input)") from exc
    rate_cable = int(pa.get_device_info_by_index(indice)["defaultSampleRate"])
    ratio = rate_cable / sr
    n_salida = int(len(muestras) * ratio)
    pos = np.arange(n_salida, dtype=np.float32) / ratio
    i0 = pos.astype(np.int64)
    i1 = np.minimum(i0 + 1, len(muestras) - 1)
    alpha = (pos - i0).astype(np.float32)
    res = muestras[i0] * (1 - alpha) + muestras[i1] * alpha
    pcm = (np.clip(res, -1.0, 1.0) * 32767).astype(np.int16).tobytes()

    stream = pa.open(
        format=pyaudio.paInt16,
        channels=1,
        rate=rate_cable,
        output=True,
        output_device_index=indice,
    )
    print(f"Entrevistador en CABLE Input ({wav.name}, {duracion_s:.1f}s) x{args.veces or '∞'}")
    n = 0
    try:
        while args.veces == 0 or n < args.veces:
            stream.write(pcm)
            n += 1
            print(f"  turno {n} reproducido ({duracion_s:.1f}s)", flush=True)
            time.sleep(args.pausa)
    except KeyboardInterrupt:
        print("\nDetenido.")
    finally:
        stream.close()
        pa.terminate()
    print("Entrevistador terminó.")


if __name__ == "__main__":
    main()
