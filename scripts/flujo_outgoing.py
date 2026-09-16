"""Flujo outgoing_es_to_en (ADR-015) — wiring de la máquina objetivo.

Cablea el core puro con los adaptadores reales y corre hasta Ctrl+C:
1. Valida el arranque OFFLINE de la traducción (precarga mwt, bloqueante).
2. Lanza el worker TTS (proceso aparte, venv-tts).
3. Abre el micrófono (RealtimeSTT es): parciales a pantalla, finales al flujo.
4. El flujo traduce, muestra en el teleprompter, sintetiza (escalera) y
   escribe el audio en VB-CABLE.

Requisitos de la máquina: venv-tts (coqui-tts), VB-CABLE, el UI del
teleprompter corriendo y el perfil enrolado (TRADUCTOR_PERFIL_ID).

Uso:
    $env:PYTHONPATH = "src"
    python scripts/flujo_outgoing.py
"""

from __future__ import annotations

import os
import sys
import tempfile
from pathlib import Path


def main(argv: list[str] | None = None) -> None:  # pragma: no cover - máquina
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")

    from traductor.flujo.adaptadores import (
        AsrRealtime,
        AsrRetorno,
        SalidaCable,
        TeleprompterHttp,
        TtsWorkerCliente,
        perfil_por_defecto,
        python_venv_tts,
        validar_arranque_real,
    )
    from traductor.flujo.outgoing import FlujoOutgoing
    from traductor.traduccion.argos import traducir

    if not validar_arranque_real():
        return  # bloqueante (ADR-014/015): sin traducción offline, no arranca

    directorio_salida = Path(tempfile.mkdtemp(prefix="flujo-outgoing-"))
    worker_tts = TtsWorkerCliente(
        python=python_venv_tts(),
        directorio_salida=directorio_salida,
        perfil_id=perfil_por_defecto(),
    )
    worker_tts.iniciar()
    cable = SalidaCable()
    cable.abrir()  # el stream se mantiene abierto entre turnos (revisión #23)

    flujo = FlujoOutgoing(
        traducir=lambda es: traducir(es, "es", "en"),
        tts_primario=worker_tts,  # escalera: si el worker falla...
        tts_fallback=None,  # ...la voz genérica llega en un PR posterior
        teleprompter=TeleprompterHttp(),
        salida_audio=cable,
        # ADR-019 fix 3: validación de artefactos en vivo (ASR-de-retorno)
        # contra el texto TRADUCIDO. Umbral 10 = calibración medida (audio
        # limpio máx 8 sobrantes, audio corrupto mín 11, n=6 turnos).
        verificar_artefactos=AsrRetorno().transcribir,
        max_sobrantes=10,
    )
    try:
        AsrRealtime(
            flujo,
            # 1.5 s de silencio cierra el turno: el párrafo entero es UN turno
            # (evita que cada frase cancele el TTS de la anterior). Ajustable
            # por env sin tocar código.
            post_speech_silence_duration=float(os.environ.get("TRADUCTOR_SILENCIO_TURNO_S", "1.5")),
        ).correr()
    except KeyboardInterrupt:
        print("\nFlujo detenido.")
    finally:
        cable.cerrar()
        worker_tts.cerrar()


if __name__ == "__main__":
    main()
