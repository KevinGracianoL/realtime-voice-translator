# ADR-001 - Pipeline en cascada, no end-to-end

- **Estado:** Aceptado (2026-08-20, paso 2) — **reconstruido el 2026-09-18** desde la tabla del README, los commits y el código (la era inicial no escribió el archivo).
- **Contexto:** la primera decisión del proyecto fue la forma de la arquitectura: un modelo *end-to-end* de audio→audio, o una **cascada** de etapas explícitas (VAD → ASR → traducción → TTS) con texto intermedio.
- **Decisión:** cascada. Cada etapa es una función separada y medible, y el **texto intermedio se muestra en pantalla** (teleprompter, ADR-007) para que el usuario pueda verificar la traducción **antes** de responder.
- **Por qué:**
  - En una entrevista, un error de traducción no es un bug: es la respuesta equivocada. El texto verificable vale más que unos milisegundos de latencia de un modelo end-to-end.
  - Cada etapa se mide por separado y el presupuesto se **deriva** de lo medido (ADR-003/014), en vez de declarar un número de caja negra.
  - Un fallo se aísla a su etapa y degrada con la escalera (ADR-008), cosa que un end-to-end monolítico no permite.
- **Consecuencias:**
  - Dos flujos explícitos (outgoing/incoming, ADR-015) sobre las mismas etapas.
  - `RegistroEtapas` atribuye la latencia al 100 % por fronteras (entrada → ASR → traducción → TTS → entrega → audible) con `raise` ante residuo sin atribuir.
  - La voz sintética (TTS) es la última etapa y es opcional: sin ella quedan los subtítulos (ADR-007/008).
- **Trazabilidad:** commits `700d2c4` (paso 2, traducción), `20d683e` (paso 3, medidor), `1413e4c`/`e3bc802` (flujos, Fase 3). Código: `src/traductor/flujo/outgoing.py`, `flujo/incoming.py`, `latencia/presupuesto.py`, `latencia/medidor.py`, `tts/harness.py` (`RegistroEtapas`). Tests: `tests/test_presupuesto.py`, `tests/test_medidor.py`. Título original del README de la era: «Pipeline en cascada, no end-to-end» (`7e191b4`).
