# ADR-006 - Sobre RealtimeSTT: el VAD/ASR es commodity, nosotros orquestamos

- **Estado:** Aceptado (2026-08-13, primer commit; matizado el 2026-09-15 con el ADR-012) — **reconstruido el 2026-09-18** desde el código y los ADRs posteriores.
- **Contexto:** la captura de micrófono necesita VAD + segmentación + ASR en tiempo real. ¿Escribirlo desde cero o apoyarse en una librería madura?
- **Decisión:** **usar RealtimeSTT** para el flujo de micrófono (VAD + streaming + ASR commodity) y dedicar el esfuerzo propio a lo que sí es diferencial: la orquestación del flujo (cola de tamaño 1, cancelación, escalera, validación de artefactos). El **incoming no usa RealtimeSTT**: lee del cable virtual y usa un VAD propio + faster-whisper (`AsrCable`), porque el cable solo acepta 48000 y el stack de RealtimeSTT imponía su device.
- **Por qué:** el VAD de micrófono es un problema resuelto por la industria; el valor del proyecto está en el pipeline medido y en sus reglas, no en reimplementar un detector de voz. Cuando el ADR-012 midió motores ASR, RealtimeSTT **no se borra** hasta que un ADR fije el ganador (decisión explícita de trazabilidad).
- **Consecuencias:**
  - `audio/captura.py` usa `AudioToTextRecorder` (tiny, int8 — ADR-004) para el micrófono.
  - `flujo/adaptadores.py` (`AsrRealtime` para el outgoing, `AsrCable` para el incoming) mantiene el motor intercambiable por env (`TRADUCTOR_MODELO_ASR`, `TRADUCTOR_ASR_DEVICE`).
  - La sustitución del incoming se cerró con evidencia en el PR #26 (ADR-015): menos dependencias y VAD propio sobre el cable.
- **Trazabilidad:** commits `60d0371` (primer commit, con RealtimeSTT), `05385ac`, `fe68a84` (incoming con `AsrCable`, PR #26). Código: `audio/captura.py`, `flujo/adaptadores.py` (`AsrRealtime`/`AsrCable`), `requirements.txt` (`realtimestt==1.0.2`). Referencias: `asr/__init__.py`, ADR-012.
