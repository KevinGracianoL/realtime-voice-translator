# ADR-009 - Dirección de traducción fijada por la fuente de audio

- **Estado:** Aceptado (2026-08-20, paso 2) — **reconstruido el 2026-09-18** desde el commit que documentó la decisión (el README de la época tenía una sección dedicada) y el código.
- **Contexto:** el sistema traduce en dos direcciones (tu voz ES→EN; el entrevistador EN→ES). ¿Cómo decide cada momento la dirección: un **detector de idioma en vivo**, o algo fijo?
- **Decisión:** la dirección se fija por la **fuente de audio**, no por detección de idioma: **micrófono = ES→EN**; **audio virtual del entrevistador = EN→ES**. Determinista, sin detector.
- **Por qué:** un detector de idioma titubea justo donde el usuario hace **code-switching** (mezcla español e inglés: nombres técnicos, siglas, frases hechas) — y ese titubeo llegaría en mitad de una respuesta. La fuente es un dato seguro: el micrófono siempre es tuyo, el cable siempre es del entrevistador.
- **Consecuencias:**
  - La arquitectura se separa en **dos flujos** (outgoing/incoming, ADR-015) con devices explícitos y configurables por env (`TRADUCTOR_DEVICE_INCOMING`, `TRADUCTOR_DEVICE_OUTGOING`).
  - `audio/virtual.py` (`seleccionar_ruta`) elige la ruta determinista por nombre de dispositivo, coherente con ADR-002.
  - Cero milisegundos de detección y cero falsos cambios de dirección.
- **Trazabilidad:** commit `700d2c4` (paso 2): añadió al README la sección «ADR-009 — Dirección de traducción fijada por fuente de audio, no por detección de idioma» y el mensaje del commit la cita. Código vigente: `audio/virtual.py:33` («Elige ruta determinista por nombre (ADR-009)»), `flujo/incoming.py`/`outgoing.py`. Tests: `tests/test_virtual.py` (`seleccionar_ruta_*`).
