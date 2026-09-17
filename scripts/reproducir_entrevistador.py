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
        "--delay",
        type=float,
        default=0.0,
        help="segundos de espera antes del primer turno (para abrir OBS y "
        "empezar a grabar antes de que el entrevistador hable)",
    )
    parser.add_argument(
        "--pausa",
        type=float,
        default=8.0,
        help="silencio entre turnos en segundos (el WAV tiene ~20 s de "
        "preguntas: 8 s de pausa hacen un ciclo de ~28 s, natural para la "
        "demo; pausas < ~1 s hacen que el VAD una los turnos)",
    )
    args = parser.parse_args(argv)
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", line_buffering=True)

    import os

    import numpy as np
    import pyaudio

    from traductor.flujo.adaptadores import _buscar_device

    wav = Path(__file__).resolve().parent / "audio" / "entrevistador_en.wav"
    muestras, sr = sf.read(str(wav), dtype="float32")
    duracion_s = len(muestras) / sr

    pa = pyaudio.PyAudio()
    # _buscar_device prefiere el device MME (16 canales, 44100 nativo): el
    # duplicado WASAPI a 48000 pasa por el resampler defectuoso y pela el
    # audio (ver hallazgo del PR #33). Nombre configurable por env para
    # alinear con TRADUCTOR_DEVICE_INCOMING del flujo incoming.
    nombre = os.environ.get("TRADUCTOR_DEVICE_ENTREVISTADOR", "CABLE Input")
    indice = _buscar_device(pa, nombre, "maxOutputChannels", 2)
    if indice is None:
        pa.terminate()
        raise RuntimeError(f"El device de salida '{nombre}' no está disponible (VB-CABLE)")
    rate_cable = int(pa.get_device_info_by_index(indice)["defaultSampleRate"])
    # resample LINEAL a la tasa del cable + ESTÉREO (mismo patrón que
    # SalidaCable): escribir mono a un device estéreo deforma el audio
    # (sonaba acelerado/agudo — bug cazado en la demo del PR #26)
    ratio = rate_cable / sr
    n_salida = int(len(muestras) * ratio)
    pos = np.arange(n_salida, dtype=np.float32) / ratio
    i0 = pos.astype(np.int64)
    i1 = np.minimum(i0 + 1, len(muestras) - 1)
    alpha = (pos - i0).astype(np.float32)
    res = muestras[i0] * (1 - alpha) + muestras[i1] * alpha
    stereo = np.repeat(res, 2)
    pcm = (np.clip(stereo, -1.0, 1.0) * 32767).astype(np.int16).tobytes()

    stream = pa.open(
        format=pyaudio.paInt16,
        channels=2,
        rate=rate_cable,
        output=True,
        output_device_index=indice,
    )
    print(f"Entrevistador en CABLE Input ({wav.name}, {duracion_s:.1f}s) x{args.veces or '∞'}")
    if args.delay > 0:
        print(f"  esperando {args.delay:.0f}s antes del primer turno...", flush=True)
        time.sleep(args.delay)
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
