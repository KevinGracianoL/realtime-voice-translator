"""Endurance de 90 minutos del flujo outgoing_es_to_en (ADR-019).

Ejercita la CADENA REAL completa — faster-whisper es → Argos → worker XTTS
(proceso aparte, venv-tts) → VB-CABLE — alimentada con los segmentos VAD de
una grabación (el micrófono real no puede hablar 90 minutos; la entrada de
mic es la misma etapa ASR, con el mismo config del flujo).

Gates del ADR-019 que evalúa y reporta:
- endurance: completa 90 minutos sin cuelgues;
- sin OOM;
- sin crecimiento sostenido de memoria (RAM + VRAM + RSS del worker): se
  compara el primer tramo contra el último y la pendiente de la serie;
- sin respuestas atrasadas: p95 del cierre del turno del primer tramo vs el
  último (degradación) y conteo de turnos > 5 s;
- sin artefactos de palabras: ASR-de-retorno sobre una muestra del audio
  sintetizado (palabras faltantes/sobrantes contra el texto intencionado);
- recuperación: si el worker se traba, se reinicia y se cuenta (watchdog,
  ADR-015) — el flujo no se cae.

El contexto de RAM se anota al final (regla del ADR-014). La salida se pega
como evidencia en el ADR-019.
"""

from __future__ import annotations

import os
import sys
import tempfile
import time
from pathlib import Path
from typing import Any


def _duracion_voz(wav: Path) -> float:  # pragma: no cover - máquina
    import soundfile as sf

    audio, sr = sf.read(str(wav), dtype="float32")
    return float(len(audio)) / float(sr)


def _ventanas_vad(whisper: Any, muestras: Any, sr: int) -> list[Any]:  # pragma: no cover
    segmentos, _ = whisper.transcribe(muestras, language="es")
    ventanas = []
    for s in segmentos:
        a = int(s.start * sr)
        b = min(int(s.end * sr), len(muestras))
        ventanas.append(muestras[a:b])
    return ventanas


def main(argv: list[str] | None = None) -> None:  # pragma: no cover - máquina
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")

    import psutil
    import soundfile as sf
    import torch
    from faster_whisper import WhisperModel

    from traductor.flujo.adaptadores import (
        SalidaCable,
        TtsWorkerCliente,
        perfil_por_defecto,
        python_venv_tts,
    )
    from traductor.hardware.cuda import vram_ocupada_mib
    from traductor.traduccion.argos import traducir

    DURACION_MIN = float(os.environ.get("ENDURANCE_MIN", "90.0"))
    wav = Path(__file__).resolve().parents[1] / "scripts" / "audio" / "voz_kevin.wav"
    muestras, sr = sf.read(str(wav), dtype="float32")

    whisper = WhisperModel("tiny", device="cuda", compute_type="int8_float16")
    ventanas = _ventanas_vad(whisper, muestras, sr)
    if not ventanas:
        raise RuntimeError("sin segmentos VAD: instrumento roto")

    directorio_salida = Path(tempfile.mkdtemp(prefix="endurance-"))
    worker = TtsWorkerCliente(
        python=python_venv_tts(),
        directorio_salida=directorio_salida,
        perfil_id=perfil_por_defecto(),
    )
    worker.iniciar()
    cable = SalidaCable()
    cable.abrir()

    latencias: list[float] = []
    etapas: dict[str, list[float]] = {"asr": [], "traduccion": [], "worker": [], "cable": []}
    reinicios = 0
    oom = 0
    artefactos = {"turnos": 0, "faltantes": 0, "sobrantes": 0}
    muestras_ram: list[tuple[float, float]] = []  # (t_min, MiB)
    muestras_vram: list[tuple[float, float]] = []
    muestras_worker_rss: list[tuple[float, float]] = []
    atrasadas = 0
    turnos = 0
    t_inicio = time.perf_counter()
    limite = t_inicio + DURACION_MIN * 60

    def _normalizar(texto: str) -> str:
        return " ".join("".join(c for c in texto.lower() if c.isalnum() or c.isspace()).split())

    def _verificar_artefactos(audio: bytes, intencion: str) -> None:
        """ASR-de-retorno: palabras faltantes/sobrantes del audio sintetizado.

        Se pasa el WAV como BytesIO (revisión #23): faster-whisper con
        ndarray NO resamplea (asume 16 k) y oiría el audio a 2/3 de
        velocidad — los desvíos medidos eran en parte el instrumento.
        """
        import io

        segs, _ = whisper.transcribe(io.BytesIO(audio), language="en")
        retorno = _normalizar(" ".join(s.text for s in segs)).split()
        esperado = _normalizar(intencion).split()
        faltan = [w for w in esperado if w not in retorno]
        sobran = [w for w in retorno if w not in esperado]
        artefactos["turnos"] += 1
        artefactos["faltantes"] += len(faltan)
        artefactos["sobrantes"] += len(sobran)

    print(
        f"Endurance {DURACION_MIN} min iniciado: {len(ventanas)} ventanas VAD, "
        f"worker pid en curso, cable listo."
    )
    print("Muestreo de memoria cada 60 s; ASR-de-retorno cada 20 turnos.")
    siguiente_memoria = time.time() + 60
    while time.perf_counter() < limite:
        t_turno = time.perf_counter()
        ventana = ventanas[turnos % len(ventanas)]
        try:
            texto_es = " ".join(s.text for s in whisper.transcribe(ventana, language="es")[0])
            etapas["asr"].append((time.perf_counter() - t_turno) * 1000.0)
            texto_en = traducir(texto_es, "es", "en")
            etapas["traduccion"].append((time.perf_counter() - t_turno) * 1000.0)
            salida = worker.sintetizar(texto_en)
            if salida is None:
                reinicios += 1  # el worker se trabó/falló: escalera (watchdog)
                continue
            audio, duracion_s, _nombre = salida
            etapas["worker"].append((time.perf_counter() - t_turno) * 1000.0)
            cable.reproducir(audio, duracion_s, "xtts-kevin")
            etapas["cable"].append((time.perf_counter() - t_turno) * 1000.0)
            latencia = (time.perf_counter() - t_turno) * 1000.0
            latencias.append(latencia)
            turnos += 1
            if latencia > 5000.0:
                atrasadas += 1
            if turnos % 20 == 0:
                _verificar_artefactos(audio, texto_en)
        except Exception as exc:  # noqa: BLE001 - el bucle no muere por un turno
            if "out of memory" in str(exc).lower():
                oom += 1
            print(f"turno {turnos}: error capturado: {exc}")
            continue
        if time.time() >= siguiente_memoria:
            minuto = (time.perf_counter() - t_inicio) / 60
            muestras_ram.append((minuto, psutil.virtual_memory().used / 1024**2))
            vram = vram_ocupada_mib(torch.cuda)
            muestras_vram.append((minuto, vram if vram is not None else 0.0))
            pid_worker = worker.pid
            rss = (
                psutil.Process(pid_worker).memory_info().rss / 1024**2
                if pid_worker is not None
                else 0.0
            )
            muestras_worker_rss.append((minuto, rss))
            siguiente_memoria = time.time() + 60

    cable.cerrar()
    worker.cerrar()
    duracion = (time.perf_counter() - t_inicio) / 60
    _reportar(
        duracion,
        latencias,
        etapas,
        turnos,
        atrasadas,
        reinicios,
        oom,
        artefactos,
        muestras_ram,
        muestras_vram,
        muestras_worker_rss,
    )


def _pendiente(puntos: list[tuple[float, float]]) -> float:
    """Pendiente de la serie (MiB por minuto) por mínimos cuadrados."""
    if len(puntos) < 2:
        return 0.0
    xs = [p[0] for p in puntos]
    ys = [p[1] for p in puntos]
    n = len(xs)
    mx = sum(xs) / n
    my = sum(ys) / n
    num = sum((x - mx) * (y - my) for x, y in zip(xs, ys, strict=True))
    den = sum((x - mx) ** 2 for x in xs)
    return num / den if den else 0.0


def _reportar(  # pragma: no cover - máquina
    duracion: float,
    latencias: list[float],
    etapas: dict[str, list[float]],
    turnos: int,
    atrasadas: int,
    reinicios: int,
    oom: int,
    artefactos: dict[str, int],
    muestras_ram: list[tuple[float, float]],
    muestras_vram: list[tuple[float, float]],
    muestras_worker_rss: list[tuple[float, float]],
) -> None:
    import statistics

    def _p95(v: list[float]) -> float:
        ordenados = sorted(v)
        return ordenados[int(0.95 * len(ordenados)) - 1] if v else 0.0

    p50 = statistics.median(latencias) if latencias else 0.0
    mitad = len(latencias) // 2
    p95_1 = _p95(latencias[:mitad]) if mitad else 0.0
    p95_2 = _p95(latencias[mitad:]) if mitad else 0.0
    pend_ram = _pendiente(muestras_ram)
    pend_vram = _pendiente(muestras_vram)
    pend_worker = _pendiente(muestras_worker_rss)
    print()
    print("=" * 70)
    print("ENDURANCE 90 MIN — ADR-019")
    print("=" * 70)
    print(
        f"duración: {duracion:.1f} min | turnos completos: {turnos} "
        f"| cierre del turno p95: {_p95(latencias):.0f} ms | p50: {p50:.0f} ms"
    )
    print("  por etapa (p95, acumulado desde el inicio del turno):")
    for etapa, valores in etapas.items():
        print(f"    {etapa:12s} {_p95(valores):.0f} ms")
    print(
        f"  p95 primer tramo: {p95_1:.0f} ms | p95 último tramo: {p95_2:.0f} ms "
        f"(degradación: {p95_2 - p95_1:+.0f} ms)"
    )
    print(
        f"respuestas atrasadas (>5 s): {atrasadas} | reinicios de worker: {reinicios} | OOM: {oom}"
    )
    print(f"artefactos (ASR-de-retorno): {artefactos}")
    print(
        f"memoria: RAM pendiente {pend_ram:+.2f} MiB/min | VRAM {pend_vram:+.2f} "
        f"MiB/min | worker RSS {pend_worker:+.2f} MiB/min"
    )
    if muestras_ram:
        print(f"  RAM: inicio {muestras_ram[0][1]:.0f} MiB -> fin {muestras_ram[-1][1]:.0f} MiB")
    if muestras_vram:
        print(f"  VRAM: inicio {muestras_vram[0][1]:.0f} MiB -> fin {muestras_vram[-1][1]:.0f} MiB")
    if muestras_worker_rss:
        inicio_rss = muestras_worker_rss[0][1]
        fin_rss = muestras_worker_rss[-1][1]
        print(f"  worker RSS: inicio {inicio_rss:.0f} MiB -> fin {fin_rss:.0f} MiB")
    print()
    print(
        "(anotar el contexto de RAM de la máquina al pegar esta evidencia: qué "
        "más estaba abierto — regla del ADR-014)"
    )
    print("=" * 70)


if __name__ == "__main__":
    main()
