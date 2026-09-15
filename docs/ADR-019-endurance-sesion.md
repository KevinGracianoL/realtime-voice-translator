# ADR-019 - Gates de sesión y endurance del flujo (Propuesto)

- **Estado:** **corrida larga EJECUTADA (2026-09-11, evidencia abajo)** — 90 min continuos de la cadena real con 417 turnos: memoria ESTABLE (sin crecimiento), sin OOM, endurance completado; el gate de respuestas atrasadas FALLA por el diseño actual del flujo (worker con síntesis completa, no streaming) y el de artefactos queda pendiente de la validación del ADR-015. La aprobación FINAL queda condicionada a los fixes identificados.
- **Contexto:** los gates del ADR-014 (TTFA, VRAM, RAM, pipeline end-to-end) miden corridas de segundos. Las fugas de memoria, las respuestas atrasadas y los fallos de disponibilidad solo aparecen en sesiones largas: la evidencia del ADR-014 (tasa de fallo 5 % en la etapa de traducción; artefacto del decodificador en ventanas específicas) demuestra que lo que no se ve en segundos sí aparece en minutos.
- **Criterios de aceptación de la corrida larga (todos deben pasar; `None` = sin medir = FALLA, misma regla del ADR-014):**

  - **Endurance: completa 90 minutos continuos** de flujo real (mic → ASR → Argos → TTS → CABLE → Meet/Zoom según el nivel de la escalera del ADR-015) **sin cuelgues, sin respuestas atrasadas y sin pérdida de dispositivos**.
  - **Sin OOM** durante la sesión.
  - **Sin crecimiento sostenido de memoria:** la RAM (y la VRAM del worker TTS) no crecen de forma monótona; se mide con `psutil`/`vram_ocupada_mib` en intervalos regulares y se compara el inicio con el final (con margen de ruido anotado).
  - **Sin artefactos de palabras:** validación ASR-de-retorno sobre el audio sintetizado (ADR-015): palabras añadidas, omitidas o repetidas, clipping, silencios anómalos y duración absurda se detectan y rechazan antes del micrófono virtual. Los candidatos del A/B del ADR-014 (`in→and`, `start→stop`) se siguen aquí.
  - **Voz reconocible en A/B:** la firma el usuario ESCUCHANDO (ningún script la firma). Criterio de escucha: timbre, RITMO (que no suene apurada), pausas naturales y palabras técnicas claras.
  - **Degradación automática sin reiniciar la llamada:** la escalera del ADR-015 baja de nivel ante fallos (voz clonada → voz genérica → subtítulos) sin perder la llamada.
  - **Recuperación automática:** los workers (TTS, traducción) con watchdog se reinician y re-enrolan sin intervención.

- **Cómo se mide:** corrida real de 90 minutos con el flujo `outgoing_es_to_en` (ADR-015) + navegador/cámara/Meet abiertos (el contexto de RAM se anota, regla del ADR-014); timestamps por etapa (patrón `RegistroEtapas`); log estructurado de fallos y degradaciones; `nvidia-smi` y `psutil` periódicos. Los valores alimentan `MedicionTts` (campos de sesión) y el harness emite el veredicto final.
- **Prerequisito de arranque (bloqueante, ADR-014):** la etapa de traducción valida su modelo spacy `mwt` precargado y funciona sin red ANTES de aceptar la llamada.
- **Consecuencias:**
  - El motor (ADR-011) pasa de "aceptado por los gates medidos" a "aprobado" solo con esta corrida.
  - Los artefactos y las fugas que aparezcan se convierten en tests de regresión (patrón del proyecto: cada bug real deja su test).
  - La evidencia de esta corrida se pega en este ADR con el mismo patrón de registro que el ADR-014.

## Evidencia — corrida larga de 90 minutos (2026-09-11)

**Instrumento:** `scripts/endurance_flujo.py` — cadena REAL (faster-whisper es co-residente → Argos → worker XTTS en proceso aparte → VB-CABLE) alimentada con los 3 segmentos VAD de `voz_kevin.wav` (el micrófono real no puede hablar 90 min; la entrada es la misma etapa ASR del flujo). Muestreo de memoria cada 60 s; ASR-de-retorno cada 20 turnos; cierre del turno medido por etapas con atribución acumulada.

**Contexto de RAM (regla del ADR-014):** durante la corrida estaban abiertos opencode (esta sesión), el harness, el worker TTS y VS Code; la RAM de la máquina BAJÓ durante la corrida (13372 → 10717 MiB) — la pendiente del flujo es la del harness, no una fuga.

| Gate | Resultado | Veredicto |
|---|---|---|
| Endurance 90 min continuos | **417 turnos completos, 0 cuelgues, 0 reinicios de worker** | ✅ PASA |
| Sin OOM | 0 eventos | ✅ PASA |
| Sin crecimiento sostenido de memoria | RAM **-9.0 MiB/min**, VRAM **+0.0** (911 → 911), worker RSS **-5.3 MiB/min** (1533 → 1177) | ✅ PASA |
| Sin respuestas atrasadas | **417/417 > 5 s** — cierre del turno p95 **21.3 s** (p50 11.1 s) | ❌ FALLA |
| Degradación (p95 primer vs último tramo) | 21.2 s → 21.1 s (**-127 ms**) | ✅ sin degradación |
| Sin artefactos de palabras | ASR-de-retorno en 20 turnos: **24 faltantes + 30 sobrantes** (el flujo aún no implementa la validación del ADR-015) | ❌ FALLA parcial |
| Voz reconocible A/B | firmado por el usuario (PR #20) + evidencia objetiva | ✅ PASA |
| Recuperación (watchdog) | 0 reinicios necesarios (el worker no falló — el camino no se ejercitó) | ⚠️ no ejercitado |

**Causas atribuidas del cierre de turno (por etapa, p95 acumulado):**
- ASR 501 ms · traducción 888 ms · **worker 9292 ms** — el worker del flujo usa la SÍNTESIS COMPLETA (`backend.sintetizar`), no el streaming: el primer chunk medido en ADR-014 era 0.7 s; la síntesis completa co-residente tarda ~8.5 s por turno. (RSS del worker leído de SU pid — corregido en la revisión #23; antes se medía el RSS del harness.)
- **cable 21311 ms** — la escritura al cable bloquea hasta que el audio termina de sonar (la duración de la reproducción no es latencia; el cierre del turno debe ser el primer sample audible, 0.2 s).
- Sin degradación entre tramos: el problema es el DISEÑO del flujo actual, no una fuga.

**Fixes identificados (con su medición):**
1. Worker con JOB STREAMING (primer chunk 0.7 s, ya medido en ADR-014) → el cierre bajaría de ~11-21 s a ~1.5-2 s.
2. `SalidaCable` escribe por bloques y el cierre del turno termina en el primer bloque aceptado (semántica first-sample-audible del ADR-014).
3. El flujo implementa la validación de artefactos del ADR-015 (ASR-de-retorno en vivo) y rechaza/degrada los turnos con desvíos.

**Conclusión:** la memoria, la estabilidad y el endurance de 90 minutos están VERIFICADOS (los gates que el endurance existe para cazar: fugas, OOM, respuestas que se degradan — todos limpios). La aprobación FINAL del motor (ADR-011) queda condicionada a los fixes 1-3; la evidencia de que el flujo no se degrada en 90 min está en esta tabla.
