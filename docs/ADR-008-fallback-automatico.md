# ADR-008 - Fallback automático: una entrevista no es un log

- **Estado:** Aceptado (2026-09-10, Fase 3) — **reconstruido el 2026-09-18** desde el código y los tests.
- **Contexto:** durante una entrevista real, cualquier etapa puede fallar (el worker del TTS se cae, el motor devuelve audio sospechoso, el teleprompter no responde). En un script de batch un fallo es un log y sigues; en una llamada en vivo, **el usuario no puede reiniciar la entrevista**.
- **Decisión:** **degradación automática en escalera, sin reiniciar la llamada**: voz clonada + subtítulos → voz genérica + subtítulos → solo subtítulos. Ningún fallo de una etapa corta el flujo; el teleprompter caído tampoco (el flujo sigue, la UI pierde la vista).
- **Por qué:** «una entrevista no es un log»: el valor está en que la conversación continúe; la peor degradación aceptable (subtítulos solos) sigue siendo útil, y es preferible a un silencio muerto.
- **Consecuencias:**
  - `flujo/outgoing.py` implementa los niveles (constantes `NIVEL_CLONADO`/`NIVEL_VOZ_GENERICA`/`NIVEL_SUBTITULOS`) y la escalera al fallar el TTS primario/fallback.
  - `flujo/adaptadores.py` envuelve el teleprompter en `suppress(Exception)` con el comentario de este ADR: el UI caído no corta el flujo.
  - La escalera se profundiza después: validación de artefactos en vivo (ADR-015) y degradación sin reiniciar con telemetría (ADR-019).
  - 6+ tests cubren la escalera: primario falla → genérica; ambos fallan → subtítulos; sin fallback → directo a subtítulos; worker que falla tras reinicio; stream que falla; audio con artefactos que no se reproduce.
- **Trazabilidad:** commits `1413e4c`/`e3bc802` (Fase 3, PR #22), reforzado en `68141c9` (PR #24). Código: `flujo/outgoing.py`, `flujo/adaptadores.py` (cita «ADR-008»). Tests: `tests/test_flujo_outgoing.py`.
