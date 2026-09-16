"""Flujo incoming_en_to_es (ADR-015) — wiring de la máquina objetivo.

Escuchar al entrevistador: captura el audio REMOTO (el entrevistador llega por
Meet/Zoom; su audio se enruta al cable virtual), lo transcribe en inglés
(ASR del cable), lo traduce EN→ES (Argos) y lo muestra como SUBTÍTULOS en el
teleprompter. El clon dinámico de la voz del entrevistador es OPCIONAL
(ADR-015: subtítulos primero) — no se implementa aquí.

Configuración de Meet/Zoom (una vez por sesión):
- Dispositivo de SALIDA de audio: "CABLE Input (VB-Audio Virtual Cable)".
  Así el audio remoto entra al cable y este flujo lo captura de "CABLE Output".

Requisitos de la máquina: VB-CABLE, el UI del teleprompter corriendo, el
modelo argos en→es instalado (traducir("hello", "en", "es") funciona offline).

Uso:
    $env:PYTHONPATH = "src"
    python scripts/flujo_incoming.py
"""

from __future__ import annotations

import sys


def main(argv: list[str] | None = None) -> None:  # pragma: no cover - máquina
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")

    from traductor.flujo.adaptadores import AsrCable, TeleprompterHttp, indice_cable_output
    from traductor.flujo.incoming import FlujoIncoming, validar_arranque_en_es
    from traductor.traduccion.argos import traducir

    def traducir_en_es(en: str) -> str:
        return traducir(en, "en", "es")

    salud = validar_arranque_en_es(traducir_en_es)
    if not salud.disponible:
        print(f"Arranque BLOQUEADO: {salud.detalle}")
        return

    flujo = FlujoIncoming(
        traducir=traducir_en_es,
        teleprompter=TeleprompterHttp(),
    )
    try:
        AsrCable(flujo, indice_cable=indice_cable_output()).correr()
    except KeyboardInterrupt:
        print("\nFlujo detenido.")


if __name__ == "__main__":
    main()


if __name__ == "__main__":
    main()
