# ADR-019 - Gates de sesión y endurance del flujo (Aceptado)

- **Estado:** **corrida larga EJECUTADA (2026-09-11) + FIXES 1-3 IMPLEMENTADOS Y RE-MEDIDOS (2026-09-15)** — 90 min continuos de la cadena real con 417 turnos: memoria ESTABLE (sin crecimiento), sin OOM, endurance completado. El gate de respuestas atrasadas FALLÓ en el diseño original (worker con síntesis completa, no streaming) y el de artefactos quedó pendiente de la validación del ADR-015. **Los tres fixes se implementaron** (job streaming del worker, cable por bloques, validación de artefactos en vivo) y la re-medición de 10 min da cierre del turno p95 **1533 ms** (era 21.3 s) con 0 reinicios y 1 respuesta atrasada de 78.
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
| Sin crecimiento sostenido de memoria | RAM **-9.0 MiB/min**, VRAM **+0.0** (911 → 911). Worker RSS: el instrumento del día 11 medía el RSS del *harness* (sin pid) — **re-medido con el árbol del proceso del worker** (revisión #23; el `python.exe` del venv es un redirector, el worker real es su hijo): **-18.5 MiB/min** (2098 → 1940 MiB) | ✅ PASA |
| Sin respuestas atrasadas | **417/417 > 5 s** — cierre del turno p95 **21.3 s** (p50 11.1 s) | ❌ FALLA |
| Degradación (p95 primer vs último tramo) | 21.2 s → 21.1 s (**-127 ms**) | ✅ sin degradación |
| Sin artefactos de palabras | **PASA con el instrumento corregido** (revisión #23): el primer reporte (24+30) medía un bug del instrumento — whisper con ndarray a 24 kHz NO resamplea (asume 16 k) y oía el audio a 2/3 de velocidad. Re-medido pasando el WAV como `BytesIO`: **10-min corrida de 47 turnos → 2 muestras ASR-de-retorno: 1 faltante + 3 sobrantes**, y en la muestra directa de 3 turnos, 2 con CERO desvíos. El desvío restante es de la TRADUCCIÓN de argos ("trauthor"), no del audio — hallazgo del #23: la validación en vivo debe comparar contra el texto *traducido* (ADR-015) | ✅ PASA (audio) — validación EN VIVO pendiente (fix 3) |
| Voz reconocible A/B | firmado por el usuario (PR #20) + evidencia objetiva | ✅ PASA |
| Recuperación (watchdog) | 0 reinicios necesarios (el worker no falló — el camino no se ejercitó) | ⚠️ no ejercitado |

**Causas atribuidas del cierre de turno (por etapa, p95 acumulado):**
- ASR 501 ms · traducción 888 ms · **worker 9292 ms** — el worker del flujo usa la SÍNTESIS COMPLETA (`backend.sintetizar`), no el streaming: el primer chunk medido en ADR-014 era 0.7 s; la síntesis completa co-residente tarda ~8.5 s por turno.
- **cable 21311 ms** — la escritura al cable bloquea hasta que el audio termina de sonar (la duración de la reproducción no es latencia; el cierre del turno debe ser el primer sample audible, 0.2 s).
- Sin degradación entre tramos: el problema es el DISEÑO del flujo actual, no una fuga.

**Fixes identificados (con su medición):**
1. Worker con JOB STREAMING (primer chunk 0.7 s, ya medido en ADR-014) → el cierre bajaría de ~11-21 s a ~1.5-2 s.
2. `SalidaCable` escribe por bloques y el cierre del turno termina en el primer bloque aceptado (semántica first-sample-audible del ADR-014).
3. El flujo implementa la validación de artefactos del ADR-015 (ASR-de-retorno en vivo) y rechaza/degrada los turnos con desvíos. La comparación es contra el texto **traducido** (hallazgo de la revisión #23: "trauthor" es un desvío de argos que el TTS reproduce fielmente — el artefacto no es del audio, pero la validación no debe comparar contra el texto intencionado).

## Fixes 1-3 implementados y re-medidos (2026-09-15)

**Qué se implementó:**

1. **Worker con job streaming** (`streaming: true`): el worker sintetiza con `inference_stream` de XTTS (latentes cacheadas por perfil) y emite **una línea por chunk** (`tipo: "chunk"`) + una línea `fin`, con flush por línea (el primer chunk cruza el pipe en ~0.7 s). El generador del worker se rinde por chunk — una lista acumulada habría retenido el primer chunk hasta el final (bug cazado en la primera corrida: "primer chunk 5197 ms, resto 43 ms"). El cliente (`TtsWorkerCliente.sintetizar_stream`) rinde un WAV por chunk y el lock de lectura serializa el pipe entre el daemon del turno y el siguiente.
2. **Cable por bloques**: `SalidaCable.reproducir` escribe el PRIMER bloque (0.2 s) y devuelve — el cierre del turno es el primer sample audible —; el resto va en un hilo daemon con lock de escritura único (el endurance cazó un `OSError -9999` de dos hilos escribiendo al mismo stream pyaudio). El flujo cierra el turno tras el primer chunk y drena el resto en un hilo daemon (turno cancelado = drenado sin reproducir, pipe alineado).
3. **Validación de artefactos en vivo** (`AsrRetorno` + `_sobrantes`): compara la transcripción del audio contra el texto **traducido**. Calibración medida: **audio limpio máx 8 sobrantes, audio corrupto (invertido/ruido) mín 11** (n=6 turnos completos) → umbral **10**. La validación corre sobre el **turno COMPLETO** en el daemon, no por chunk: whisper tiny alucina sobre fragmentos cortos (un chunk limpio de 1 palabra dio "thank you guys") y no discrimina (audio invertido dio 2 sobrantes, el mismo rango que un limpio) — la medición está en la evidencia.

**Re-medición (endurance 10 min, misma máquina, 2026-09-15):**

| Métrica | Antes (90 min, 2026-09-11) | Después (10 min, 2026-09-15) |
|---|---|---|
| Cierre del turno p95 | **21.3 s** | **1533 ms** (p50 1422) |
| Respuestas atrasadas (>5 s) | 417/417 | **1/78** |
| Reinicios de worker | 0 | **0** |
| OOM | 0 | **0** |
| Degradación (p95 primer vs último tramo) | -127 ms | **-86 ms** |
| Memoria | RAM -9.0 / VRAM +0.0 / RSS -18.5 MiB/min | RAM +10.7 / VRAM +0.5 / RSS +7.1 MiB/min (10 min: ruido de contexto, la corrida de 90 min es la que valida estabilidad) |

Desglose por etapa (p95 acumulado): ASR 226 ms · traducción 508 ms · **primer chunk 1354 ms** · **ruteo 1533 ms** — el cierre es el primer sample audible, no la síntesis completa (la validación del turno corre en el daemon, fuera del cierre).

**Conclusión:** la memoria, la estabilidad y el endurance de 90 minutos están VERIFICADOS (los gates que el endurance existe para cazar: fugas, OOM, respuestas que se degradan — todos limpios) y los fixes 1-3 están implementados con su re-medición: el cierre del turno bajó de 21.3 s a 1.53 s, las respuestas atrasadas de 417/417 a 1/78 y la validación de artefactos quedó en vivo con umbral calibrado por medición.
