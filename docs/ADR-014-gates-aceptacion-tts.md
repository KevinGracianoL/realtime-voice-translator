# ADR-014 - Gates de aceptación del motor TTS

- **Estado:** **XTTS-v2 ACEPTADO por los gates MEDIDOS del ADR-014** (pipeline end-to-end 1237.0-1469.3 ms < 2000 ms con 100 % atribuido (protocolo vigente: segmentos VAD completos; 0 de 4 corridas sobre el límite); TTFA derivado PASA; VRAM/RAM PASA; 2026-09-09, quinta capa abajo). Aprobacion FINAL condicionada a los gates de sesion (ADR-019). Candidato B RECHAZADO (arquitectural). El modulo gates.py y el harness quedan como gate de regresion.
- **Contexto:** para que ADR-011 pase de "Propuesto" a un motor concreto, hace falta un **criterio de aceptación objetivo**. Sin él, la elección del motor sería una opinión más. Las mediciones se hacen en la GPU real (GTX 1650 Ti 4 GB, presupuesto ADR-003), con el ASR co-residente — mismo criterio que ADR-010 y ADR-012.
- **Cadena de decisión:** se prueba primero **XTTS-v2** (fork `coqui-tts`, pesos CPML — aceptable por uso personal no comercial, declarado en ADR-011). **Orden práctico: correr SOLO el gate de VRAM primero** (smoke de ~5 min, con el ASR co-residente) antes de montar el venv completo — es el gate que más probablemente tumbe al candidato en esta GPU de 4 GB (la objeción de VRAM del descarte anterior, convertida en gate, no borrada). **Si falla cualquiera de los bloqueantes, se rechaza** (sin cuantización agresiva que empeore la voz) y se prueba **Supertonic 3 CPU + OpenVoice V2** (candidato B). El que XTTS se instale y se mida no lo declara aceptado: la aceptación es el resultado de los gates.
- **Criterios de aceptación (todos deben pasar; `None` = sin medir → FALLA):**

  - **TTFA (time to first audio) = PRIMER CHUNK reproducible, caliente p95 < presupuesto DERIVADO:** `2000 ms − (ASR p95 + traducción p95 + ruteo p95)`, cada etapa MEDIDA en el hardware con n≥20 (corrección 2026-09-09: el "< 400 ms" literal era un sub-presupuesto inventado). Etapa sin medir → presupuesto no derivable → FALLA. **Nota metodológica (2026-09-09, review #19):** la suma de p95 de etapas NO es el p95 de la cadena (los peores casos no coinciden): el presupuesto derivado es una cota CONSERVADORA, no la aritmética exacta del flujo — el VEREDICTO lo da la corrida encadenada (pipeline end-to-end). El TTFA derivado es un gate DIAGNÓSTICO: si se vuelve rojo por el error de OTRA etapa (como ocurrió con el ruteo inflado a 1003.5 ms), el problema es de la definición de esa etapa, no del motor.
  - **VRAM total (ASR + motor) < 3.2 GB** (3276.8 MiB). El "~1 GB" del ASR es un **supuesto a medir en la primera corrida**, no una cifra verificada.
  - **RAM total < 18 GB.** Definición: `psutil.virtual_memory().used` de TODA la máquina — el veredicto depende de qué más esté abierto; anotar el contexto al correr (como con `nvidia-smi`).
  - **Pipeline warm p95 < 2 s** (cierre del turno: ASR ≤300 ms + traducción ≤200 ms + primer fragmento ≤350 ms + ruteo ≤100 ms → ~1.3 s, dentro del objetivo de 1.5–2 s del ADR-003).
  - **Sin OOM.**
  - **Sin crecimiento sostenido de memoria** durante la sesión.
  - **Sin palabras añadidas, omitidas, repetidas ni sonidos metálicos** en el corpus de aceptación.
  - **La voz del usuario es reconocible** en una prueba A/B.
  - **Endurance: completa 90 minutos continuos** (ADR-019) sin cuelgues ni respuestas atrasadas.

- **Cómo se mide:** `scripts/medir_gates_tts.py` — carga el motor candidato y **faster-whisper co-residente**, hace `warm-up`, `n≥20` síntesis calientes con reloj inyectable y evalúa `evaluar_gates`. La VRAM se mide con **`vram_ocupada_mib()`** (`mem_get_info` a nivel driver, una sola llamada sobre el dispositivo actual): incluye lo que CTranslate2 reserva fuera del allocator de torch. **Whisper se calienta con su PRIMERA `transcribe()`** (CT2 reserva workspace/KV/beam ahí) y el generador se drena con `list()`. **Auto-verificación del instrumento:** la foto de Whisper se toma justo después de su warm-up (delta aislado, impreso siempre, `raise` si < 50 MiB); la foto de co-residencia después de las síntesis TTFA; `--warmup-audio` obligatorio y validado en argparse (falla en el primer segundo); 0 segmentos = `raise`. Los gates de sesión (OOM, memoria, artefactos, A/B, endurance) los mide la corrida larga del ADR-019 y se alimentan como valores al módulo. Prueba de cierre: VRAM vs `nvidia-smi` ±100 MiB. Un gate sin medir (`None`) **falla**: un criterio que no se puede evaluar no pasa en silencio.
- **Consecuencias:**
  - Un candidato que pase todos los gates se propone como decisión de ADR-011 (motor concreto) con su propio PR.
  - Un candidato que no pase se descarta con la evidencia pegada aquí (mismo patrón que ADR-010) y se prueba el siguiente de la cadena (B: Supertonic + OpenVoice V2).
  - El módulo `gates.py` queda como gate de regresión: si un futuro cambio empeora alguna métrica, se detecta re-corriendo el harness.

## Evidencia — XTTS-v2 rechazado por TTFA (2026-09-08)

**Entorno:** Ryzen 5 4600H / GTX 1650 Ti 4 GB / Windows. Motor: XTTS-v2 vía `coqui-tts` 0.27.5 (fork idiap), torch 2.14.0+cu132, en venv propio (venv-tts). ASR co-residente: faster-whisper tiny int8. Referencia de voz: `scripts/audio/voz_kevin.wav` (13 s, 16 kHz mono). Harness con `--warmup-audio`; `warm-up: 3 segmentos`, `delta VRAM (Whisper) = 110 MiB`.

| Gate | Medido | Límite | Estado |
|---|---|---|---|
| TTFA síntesis completa (contrato no-streaming, p95 n=20) | **3574.6 ms** | < 400 ms | FALLA |
| TTFA primer chunk (streaming real `inference_stream`, n=5) | rango **655–889 ms** (p95 no válido con n<20) | < 400 ms | FALLA |
| VRAM co-residente (XTTS + Whisper) | 3010.9 MiB | < 3276.8 MiB | PASA |
| RAM total (máquina) | 11804.9 MiB | < 18432 MiB | PASA |

**Lectura honesta del TTFA:** el gate define TTFA = time to first audio; con el contrato no-streaming se mide la síntesis completa (3574 ms). XTTS soporta streaming (`inference_stream`, 6 chunks por frase) y el **primer chunk más rápido observado (655 ms) ya supera el límite por 60 %** — el rechazo no depende del `n` (no se reporta p95 con n=5; la regla del ADR exige n≥20). Las latents de condicionamiento (756 ms) se calculan una vez y no entran al presupuesto por turno (ADR-011).

**Nota (2026-09-18, review PR #34):** los números de streaming de esta ronda se midieron con `inference_stream`; el camino declarado de XTTS es hoy el **batch por frase** con latentes cacheadas (medición en ADR-019: latentes 747–807 ms una vez por perfil, primer chunk ~2.0 s, RTF sostenido 0.76–0.98) y el harness `scripts/medir_gates_tts.py` mide ese camino. Las latentes siguen fuera del presupuesto por turno, como fija este ADR.

**Contexto de RAM (nota):** el valor 11804.9 MiB es el uso de TODA la máquina (`psutil.virtual_memory().used`) y **no se anotó qué había abierto** durante la corrida. Hoy sobra holgura (~6.6 GB), pero el candidato B puede quedar al filo: **las corridas futuras deben anotar el contexto** (navegador, Meet, etc.) como se hace con `nvidia-smi`.

**Veredicto:** un bloqueante basta → **XTTS-v2 RECHAZADO** (sin cuantización agresiva). Sigue **Supertonic 3 CPU + OpenVoice V2 (candidato B)** — este ADR queda como gate de regresión y registro del go/no-go.

## Evidencia — candidato B (Supertonic 3 CPU + OpenVoice V2) rechazado por TTFA (2026-09-09)

**Entorno:** Ryzen 5 4600H / GTX 1650 Ti 4 GB / Windows. Motor: Supertonic 3 (SDK `supertonic` 1.3.1, ONNX CPU, voz M1) + OpenVoice V2 (converter, CPU, watermark off) vía `BackendB` (`src/traductor/tts/backend_b.py`), torch 2.14.0+cu132 en venv-tts. ASR co-residente: faster-whisper tiny int8 (CUDA). Referencia: `scripts/audio/voz_kevin.wav` (13 s). Harness: `--motor b --warmup-audio voz_kevin.wav --referencia voz_kevin.wav`; texto: "Thank you for the question. My experience with distributed systems is one of my strongest points." (fragmento real del flujo ES→EN). `warm-up: 3 segmentos`, `delta VRAM (Whisper) = 110 MiB`. El `target_se` (timbre de Kevin) se extrae una vez por perfil (~2 s con VAD silero, en el warm-up, no entra al turno); el `src_se` (timbre de la voz base M1) también se cachea.

| Gate | Medido | Límite | Estado |
|---|---|---|---|
| TTFA caliente p95 (n=20, pipeline completo por turno) | **7885.8 ms** | < 400 ms | FALLA |
| VRAM co-residente (Whisper sola: el motor es CPU) | 910.9 MiB | < 3276.8 MiB | PASA |
| RAM total (máquina) | 11531.8 MiB | < 18432 MiB | PASA |

**Contexto de RAM (anotado, regla del ADR):** durante la corrida estaban abiertos opencode (~1 GB), VSCode (×2), WhatsApp, Edge webview, Windows Defender y el explorador — el 11531.8 MiB incluye esa máquina normal de trabajo; el harness en sí añade ~2 GB sobre la línea base (~9.5 GB sin él).

**Lectura honesta del TTFA:** el pipeline B es **no-streaming por construcción**: Supertonic sintetiza el fragmento completo (~1.75 s caliente para 6.4 s de audio) y OpenVoice V2 convierte el timbre del fragmento completo (~5-6 s). No existe "primer chunk" en ninguna etapa — el TTFA es la síntesis entera. **7885.8 ms es 20× el límite** y el número es estable (n=20, p95). La latencia no depende del streaming: es el costo del convertidor sobre el audio completo.

**Veredicto:** un bloqueante basta → **candidato B RECHAZADO**. Con XTTS y B descartados, la cadena del ADR-011 está agotada (Pocket ya estaba descartado). La escalera del ADR-015 apunta al nivel 3 (voz inglesa **genérica** Supertonic/Piper + subtítulos) — sin clonación de timbre, la etapa de conversión desaparece y queda solo el synth (~1.75 s para 6.4 s de audio, aún por encima del TTFA de 400 ms para fragmentos completos; la decisión de flujo decide cómo partir el audio y qué gate aplica). La clonación de voz (Fase 3) queda condicionada a un motor que cumpla los gates: la evidencia de este ADR es el criterio, no la opinión.

## Corrección — el sub-gate de TTFA era inválido y se reemplaza por uno DERIVADO (2026-09-09)

**Qué se corrige:** el gate "TTFA < 400 ms" de la evidencia de 2026-09-08 era un **sub-presupuesto inventado** (ASR 300 + traducción 200 + primer fragmento 350 + ruteo 100 = 950 ms de etapas "presupuestadas"), nunca validado contra el total. Con el propio número medido de XTTS streaming (primer chunk 889 ms peor caso, n=5), la suma contra el objetivo end-to-end de ADR-003 cabe: **300 + 200 + 889 + 100 = 1489 ms < 2000 ms**. La evidencia anterior NO se borra (registro del error); este apartado corrige el criterio y pega la medición nueva. Regla del usuario que dirige el proyecto: **nunca fijar un número a ojo** — el presupuesto se deriva de etapas medidas.

**Criterio nuevo (implementado en `gates.py` como cálculo, no literal):**

```
presupuesto_ttfa = 2000 ms − (ASR p95 + traducción p95 + ruteo p95)
```

Cada etapa debe estar MEDIDA en el hardware (n≥20, p95, reloj inyectable del harness); una etapa sin medir hace el presupuesto no derivable y el gate FALLA. TTFA se redefine como **primer chunk de audio reproducible** (streaming caliente para XTTS).

**Etapas medidas (2026-09-09, máquina objetivo, venv-tts, faster-whisper co-residente, `warm-up: 3 segmentos`):**

| Etapa | Supuesto anterior | Medido ahora (p95, n≥20) | Veredicto |
|---|---|---|---|
| ASR (faster-whisper tiny int8, es) | 300 ms | **768.5 ms** | el supuesto subestimaba 2.5× |
| Traducción (Argos ES→EN, 20 frases distintas) | 200 ms | **176.6 ms** | supuesto razonable (argos 1.11 cachea el mismo texto: se miden frases distintas como los turnos reales) |
| Ruteo a VB-CABLE | 100 ms | **sin medir — VB-CABLE NO instalado** (instrumento listo: drenado real hasta que el dispositivo consume el chunk + auto-verificación del piso físico `frames/sample_rate` con `raise` si el p95 sale por debajo — patrón delta VRAM; tests en `test_harness_gates.py`) | gate FALLA por regla (sin medir no pasa) |
| TTFA 1er chunk (XTTS `inference_stream`, n≥20, caliente) | 655–889 ms (n=5) | **691.1 ms** | el n=5 de la corrida anterior no cumplía la regla n≥20; el p95 n≥20 confirma el orden de magnitud |
| Pipeline end-to-end (audio → 1er audio en el micrófono virtual) | inferido sumando supuestos (nunca medido) | **sin medir — VB-CABLE ausente** (instrumento listo: UNA corrida encadenada con cortes DISTINTOS de audio por iteración — los turnos reales difieren y argos cachea texto idéntico —, primer chunk + ruteo con drenado) | gate FALLA por regla; la inferencia es exactamente lo que produjo el rechazo equivocado |
| VRAM co-residente | — | **2906.9 MiB** | PASA (< 3276.8) |
| RAM total (máquina, contexto anotado) | — | **13198.6 MiB** | PASA (< 18432) |

**Presupuesto derivado con lo medido:** 2000 − 768.5 − 176.6 − ruteo → **1054.9 ms con ruteo = 0** (o 954.9 ms con el supuesto de 100 ms). El TTFA medido (691.1 ms) cabe en cualquiera de los dos — pero el gate FALLA hoy porque **el ruteo no se puede medir** (VB-CABLE no está instalado; instrumento listo y verificado: drenado real + auto-verificación del piso físico con `raise`, pipeline end-to-end por cortes distintos) y el pipeline end-to-end sigue sin medir. XTTS **NO queda aprobado**: la corrección invalida el rechazo anterior, no fabrica un pase — la aprobación requiere instalar VB-CABLE y cerrar estas dos mediciones + los gates de sesión (OOM, memoria, artefactos, A/B — lo firma el usuario escuchando —, endurance 90 min). Próximo paso: instalar VB-CABLE y re-correr el harness (`--motor xtts --warmup-audio voz_kevin.wav --referencia voz_kevin.wav`); el instrumento cierra las dos mediciones pendientes con su auto-verificación.

**Hallazgos de la medición:**
- `ARGOS_COMPUTE_TYPE` sin "default" hace que argos es→en produzca basura ("mainstream" en bucle); el wrapper `traductor.traduccion.argos` ya lo fija — el harness exige salida correcta (sanity) antes de medir la etapa.
- Argos cachea el mismo texto (0.0 ms): medir con frases distintas, como los turnos reales.
- torchaudio/torchcodec necesita las DLL compartidas de FFmpeg en `PATH` (build BtbN, mismo requisito que la evidencia XTTS).

## Evidencia final — XTTS-v2 RECHAZADO por el pipeline end-to-end medido (2026-09-09, VB-CABLE instalado)

**Entorno:** Ryzen 5 4600H / GTX 1650 Ti 4 GB / Windows / VB-CABLE instalado. Motor: XTTS-v2 (fork coqui-tts 0.27.5) con el perfil de Kevin, streaming `inference_stream` caliente, faster-whisper tiny int8 co-residente, Argos ES→EN (el wrapper fija `ARGOS_COMPUTE_TYPE=default`), FFmpeg DLLs en PATH. Harness: `--motor xtts --warmup-audio voz_kevin.wav --referencia voz_kevin.wav`; `warm-up: 3 segmentos`, `delta VRAM (Whisper) = 110 MiB`.

**Tabla completa: supuesto anterior | medido ahora | veredicto (n=20 iteraciones por corrida, p95 intra-corrida; entre corridas se reporta el PEOR caso — 4 corridas no dan p95 inter-corrida):**

| Etapa | Supuesto | Medido | Veredicto |
|---|---|---|---|
| ASR (faster-whisper tiny int8 es) | 300 ms | **762.0 ms** | el supuesto subestimaba 2.5× |
| Traducción (Argos ES→EN, 20 frases distintas) | 200 ms | **206.0 ms** | supuesto razonable (argos cachea texto idéntico → frases distintas) |
| Ruteo a VB-CABLE (drenado real, auto-verificación del piso físico) | 100 ms | **1003.5 ms** | el supuesto fallaba por 10×: un chunk de 1 s no puede ser reproducible en menos de 1 s (piso 1000 ms, verificación PASA) |
| TTFA 1er chunk (XTTS streaming, n≥20, caliente) | 655–889 ms (n=5, no válido) | **727.0 ms** | confirmado con n≥20 |
| **Pipeline end-to-end (audio → 1er audio en el CABLE, UNA corrida encadenada)** | inferido en 1489 ms (suma de supuestos) | **2756.4 ms** | **> 2000 ms → FALLA** |
| VRAM co-residente | — | 2906.9 MiB | PASA (< 3276.8) |
| RAM total (contexto anotado) | — | 13304.4 MiB | PASA (< 18432) |

**Presupuesto TTFA derivado:** 2000 − 762.0 − 206.0 − 1003.5 = **28.5 ms** → TTFA 727.0 ms FALLA. Nota: con la semántica de drenado, el ruteo ≈ duración del chunk (el audio no se reproduce más rápido que su propia duración); el gate DECISIVO es el pipeline end-to-end medido (2756.4 ms), que falla por 756 ms.

**Quién consume el presupuesto (análisis del pipeline 2756.4 ms):** ruteo-drenado 1003.5 ms (36 %, inherente a la definición "reproducible" para un chunk de ~1 s), ASR 762.0 ms (28 % — 2.5× su supuesto: primer candidato a optimizar, p. ej. modelo menor o `compute_type` distinto, NO el TTS), TTFA 727.0 ms (26 %), traducción 206.0 ms (7 %). Sin el drenado (ruteo = 0) la cadena sería 762 + 206 + 727 = 1695 ms < 2000 ms: el cuello es ASR + chunk largo, no el sintetizador.

**Veredicto: XTTS-v2 RECHAZADO con fundamento medido.** Un bloqueante basta (pipeline end-to-end > 2000 ms); los gates de sesión (OOM, memoria, artefactos, A/B — lo firma el usuario escuchando —, endurance 90 min) quedan PENDIENTES y no cambian el veredicto. La optimización propuesta (no ejecutada, regla del usuario): **el ASR primero** (762 ms vs 300 ms supuestos), luego chunks de síntesis más cortos. La cadena del ADR-011 queda agotada: XTTS ✗ (medido), B ✗ (arquitectural, 7885.8 ms), Pocket ✗ (descartado) → el flujo arranca por la escalera del ADR-015 (nivel 3: voz genérica Supertonic + subtítulos, o nivel 4).

## Cuarta capa - la metrica de ruteo era invalida y se corrige (2026-09-09, segundo error del mismo tipo)

**Que se corrige:** la medicion de ruteo anterior (1003.5 ms con un chunk de ~1 s) NO era latencia: era la DURACION del audio. El instrumento median hasta que el chunk TERMINABA de reproducirse (drenado), cuando lo que importa es cuando EMPIEZA - el interlocutor oye el primer sample, no espera a que el chunk acabe. La duracion del audio no es latencia (si lo fuera, hablar mas largo te haria mas lento). Agravante: la auto-verificacion pedida ("p95 < piso de drenado -> raise") FORZO el error - blindo el bug en vez de atraparlo. El guard se INVIERTE.

**Definicion nueva (implementada y verificada):** ruteo = TIME-TO-FIRST-SAMPLE-AUDIBLE: desde que el chunk se entrega al dispositivo hasta que el primer sample es reproducible en CABLE Input. NO incluye la duracion del chunk. El instrumento usa el LOOPBACK del CABLE: escribe a CABLE Input y DETECTA el primer sample audible (>= 5 muestras consecutivas sobre 0.2 % de full scale) leyendo CABLE Output - el dispositivo real entregandolo.

**Guard invertido (atrapa el error conocido; test obligatorio: medir el drenado DEBE fallar):**
- p95 >= duracion del chunk -> raise (la medicion incluyo el drenado otra vez).
- p95 <= 0 (o None) -> raise (no se midio nada).
- Imprime siempre: p95 medido, duracion del chunk y ambos limites.

## Evidencia final corregida - XTTS-v2 RECHAZADO por el pipeline end-to-end medido (2026-09-09, VB-CABLE instalado)

**Entorno:** idem capas anteriores (XTTS-v2 fork coqui-tts, perfil de Kevin, streaming caliente, faster-whisper co-residente, Argos con ARGOS_COMPUTE_TYPE=default, VB-CABLE, FFmpeg DLLs). Harness: `--motor xtts --warmup-audio voz_kevin.wav --referencia voz_kevin.wav`; `warm-up: 3 segmentos`, `delta VRAM (Whisper) = 110 MiB`.

**Tabla completa: supuesto | medido antes (metrica rota) | medido ahora | veredicto (n>=20, p95):**

| Etapa | Supuesto | Antes (roto) | Ahora | Veredicto |
|---|---|---|---|---|
| ASR (faster-whisper tiny int8 es) | 300 ms | 768.5 ms | **773.1 ms** | 2.5x el supuesto - primer candidato a optimizar |
| Traduccion (Argos ES->EN, 20 frases distintas) | 200 ms | 176.6 ms | **175.0 ms** | supuesto razonable |
| Ruteo a VB-CABLE (primer sample audible, loopback) | 100 ms | 1003.5 ms (duracion del chunk, invalido) | **221.0 ms** | latencia real del dispositivo (guard invertido PASA: 0 < 221 < 1000) |
| TTFA 1er chunk (XTTS streaming, n>=20) | 655-889 (n=5) | 691.1 ms | **686.0 ms** | confirmado con n>=20 |
| **Pipeline end-to-end (audio -> primer sample audible en el CABLE, UNA corrida encadenada)** | inferido 1489 ms | 2756.4 ms (ruteo inflado) | **2532.4 ms** | **> 2000 ms -> FALLA** |
| VRAM co-residente | - | 2906.9 | **2938.9 MiB** | PASA (< 3276.8) |
| RAM total (contexto anotado) | - | 13198.6 | **13506.7 MiB** | PASA (< 18432) |

**Presupuesto TTFA derivado:** 2000 - 773.1 - 175.0 - 221.0 = **830.9 ms** -> TTFA 686.0 ms **PASA** (primera vez que el gate TTFA pasa con la metrica correcta).

**Margen contra los 2000 ms:** el pipeline mide **2532.4 ms** - FALLA por 532 ms. La suma de etapas (773.1 + 175.0 + 686.0 + 221.0 = 1855.1 ms) cabe, pero la corrida ENCADENADA no: hay 677 ms de diferencia entre la suma y la medicion real - exactamente el tipo de inferencia que produjo los rechazos erroneos anteriores; por eso el veredicto es la corrida encadenada, no la suma.

**Veredicto: XTTS-v2 RECHAZADO con fundamento medido.** Un bloqueante basta (pipeline end-to-end 2532.4 ms > 2000 ms). Margen negativo -> se PROPONE (no se ejecuta, regla del usuario) la optimizacion del ASR: 773.1 ms = 45 % de la suma de etapas y 2.5x su supuesto original; opciones a evaluar: modelo menor, compute_type distinto, ajuste de beam size. Los gates de sesion (OOM, memoria, artefactos, A/B - lo firma el usuario escuchando -, endurance 90 min) quedan PENDIENTES y no cambian el veredicto. Cadena del ADR-011 agotada: XTTS ✗ (medido), B ✗ (arquitectural, 7885.8 ms), Pocket ✗ (descartado). Licencia recordada: los pesos de XTTS-v2 son Coqui Public Model License (no comercial) - valido para uso personal, no para servicio a terceros (ya declarado en ADR-011).
## Quinta capa - atribucion del pipeline y veredicto final (2026-09-09)

**Por que se instrumento:** la corrida encadenada (2920.5 ms) dejaba 677 ms sin atribuir frente a la suma de etapas aisladas. **Hallazgo adicional (tercera capa de correccion de instrumento):** el ASR encadenado oscilaba 768-1829 ms entre corridas con los mismos cortes; el diagnostico dirigido (misma rebanada 20x, con y sin XTTS residente y con streams abiertos) aíslo la causa: DOS ventanas de 3 s de esta grabacion que arrancan en los segundos 8.5 y 9.0 hacen que el decodificador tarde 700-3100 ms (artefacto del contenido acustico, verificable: las ventanas adyacentes hacen 139-288 ms). El VAD del flujo real segmenta en 0, 4.66 y 10.04 - esos inicios no existen en la entrada real. **RIESGO ABIERTO (no cerrado):** el artefacto del decodificador (700-3100 ms en ventanas con inicios 8.5/9.0 s, adyacentes a 139-288 ms) queda documentado como riesgo del ASR con este modelo/contenido; no desaparece por no estar en el conjunto nuevo - si el VAD produjera un inicio en esa region, el flujo pagaria esa latencia. CORRECCION DEL CRITERIO (revision del usuario): las ventanas deben DERIVARSE de la segmentacion VAD real (inicios 0 / 4.66 / 10.04), NO elegirse por su latencia - el conjunto anterior era circular (se excluian las lentas y luego se reportaba que no habia lentas). El conjunto actual sale de los inicios VAD del corpus; NINGUNA ventana se descarta por su resultado. Las ventanas patologicas (inicios 8.5/9.0) NO estan en el conjunto porque el VAD real no segmenta ahi - no porque se excluyeran. Regla: no se emite veredicto con una fraccion sin atribuir. Se marco cada frontera (entrada de audio -> ASR -> traduccion -> primer chunk TTS -> entrega al dispositivo -> primer sample audible) con reloj inyectable y cierre EXACTO por iteracion (`RegistroEtapas`, guard verificado: si la suma no cierra el total, raise).

**Residuo encontrado y corregido (cuarta correccion de instrumento):** el bucle de escritura BLOQUEANTE del chunk hacia entrar el drenado del dispositivo dentro de la etapa "entrega" (527.7 ms del desglose 2920.5). El primer sample audible llega DURANTE la escritura; la medicion esperaba a escribir todo el chunk. Correccion: detectar el primer sample audible despues de CADA bloque escrito - la medicion termina cuando el interlocutor oye, no cuando el chunk termina de escribirse. Con esto "entrega" paso de 527.7 ms a 19.4 ms y la etapa quedo limpia de contenido de audio.

**Desglose medido (n=20, p95, cierre exacto por iteracion, guard verificado):**

| Frontera | p95 | vs aislado |
|---|---|---|
| ASR (segmentos VAD COMPLETOS, protocolo vigente) | 234.4-256.7 ms | aislado 771-812 ms (wav completo 13 s) - consistente con ~55-65 ms/s de habla; sin contencion GPU (el desglose anterior 183-195 ms era del protocolo de ventanas de 3 s) |
| Traduccion | 278.8-322.0 ms | aislado 177.6-182.0 ms (los segmentos completos dan textos mas largos; y el p95 cae en traducciones reales, no cacheadas, en este protocolo) |
| TTS primer chunk | 736.6-831.4 ms | aislado 721.0-813.8 ms - consistente |
| Entrega y primer sample audible (UN solo evento) | 15.4-19.7 ms | ruteo aislado 219.9-221.6 ms (offset estereo corregido, indice//2; el aislado paga la apertura del dispositivo; en cadena el stream ya esta caliente) |

**Tabla final: supuesto | medido | veredicto (n>=20, p95):**

| Gate | Supuesto | Medido | Veredicto |
|---|---|---|---|
| TTFA 1er chunk (derivado: 2000 - 771-812 - 182-197 - 221-224 = 767-826 ms) | < 400 literal (invalido) | 723.3-754.9 ms | **PASA** |
| Pipeline end-to-end (audio -> primer sample audible, UNA corrida encadenada, 100 % atribuido; 4 corridas del protocolo vigente, n=20 iteraciones por corrida) | inferido 1489 ms | **peor caso 1469.3 ms (rango 1237.0-1469.3)** | **< 2000 ms -> PASA (margen 531 ms)** |
| VRAM co-residente | < 3276.8 | 2906.9-2938.9 MiB | PASA |
| RAM total (contexto anotado) | < 18432 | 13079-13859 MiB | PASA |
| Sin OOM / memoria estable / artefactos / A/B / endurance 90 min | - | sin medir | PENDIENTES (no son aprobados) |

**Veredicto: XTTS-v2 ACEPTADO por los gates MEDIDOS del ADR-014** (pipeline **peor caso medido 1469.3 ms** < 2000 ms, rango 1237.0-1469.3 en 4 corridas con el protocolo vigente (segmentos VAD completos); 0 de 4 sobre 2000 ms; margen 531 ms sobre el peor caso; n=4 corridas es insuficiente para p95 entre corridas — se reporta el peor caso, no p50). **Estabilidad:** 19 corridas con el protocolo de ventanas de 3 s (p50 1107.1 / p95 1158.2, rango 1059.9-1171.5, 0 sobre el limite) y 4 corridas con el protocolo vigente de segmentos VAD completos (rango 1237.0-1469.3, peor caso 1469.3 ms, 0 sobre el limite). El veredicto usa el protocolo vigente y su peor caso. **LIMITACION CONOCIDA (declarada, no escondida):** el corpus tiene 3 segmentos VAD unicos, asi que n=20 son 3 entradas distintas CICLADAS; argos cachea los textos repetidos y el p95 del pipeline resulta optimista en <= 177 ms (p95 aislado de traduccion). Con margen de 531 ms (peor caso) no cambia el veredicto, pero el flujo real con turnos distintos pagara esa traduccion en cada turno. **TASA DE FALLO DECLARADA (dato aparte de la latencia):** 1 de las 20 CORRIDAS de estabilidad (5 %) MURIO: la corrida 13 cayó en su etapa de TRADUCCION (argos intento descargar el modelo spacy `mwt` on-demand y la red fallo). No es latencia: es un fallo de DISPONIBILIDAD - en una entrevista real el traductor muere a media pregunta. El p95 se reporta sobre las 19 corridas validas (correcto para latencia), pero la tasa de fallo 5 % (1 corrida de 20) se declara aparte y NO desaparece. **Requisito BLOQUEANTE del flujo outgoing_es_to_en (ADR-015): precargar el modelo spacy `mwt` en el arranque del worker de traduccion y validar la traduccion sin red antes de aceptar una llamada** (no es una nota, es una condicion de entrada del flujo). La aprobacion FINAL queda condicionada a los gates de sesion (ADR-019): OOM, crecimiento de memoria, artefactos, voz reconocible en A/B (lo firma el usuario escuchando) y endurance de 90 minutos - pendiente no es aprobado. Margen saludable (531 ms sobre el peor caso medido): la optimizacion del ASR queda descartada de momento (la etapa mayor sigue siendo el TTS, 736-831 ms; el ASR encadenado subio de 183-195 a 234-256 ms con los segmentos completos, pero el margen se mantiene).

## Regla del guard de instrumento (leccion transferible de estas cinco capas)

Cada correccion de instrumento de este ADR sigue la misma regla, y la regla es el activo que se lleva a cualquier medicion futura:

1. **El guard debe FALLAR ante el error conocido.** Si el test que reproduce el error no hace reventar el guard, el guard no sirve: `verificar_ruteo_primer_sample` tiene el test que mide el drenado completo (1003.5 ms >= 1000 ms) y exige que `raise`; `verificar_resolucion_audible` tiene el test del 0.1 ms sub-resolucion.
2. **Un guard debe anclarse a una referencia INDEPENDIENTE de la definicion que verifica** (un loopback fisico, un segundo instrumento, un valor conocido externo), nunca a un derivado de la propia definicion: un guard cuyo umbral se deriva de la MISMA definicion que valida no puede detectar que la definicion es incorrecta — solo puede confirmarla. `piso_ruteo_ms(frames, sr) = frames/sr` era el piso de DRENADO: el guard exigia `p95 >= 1000 ms` para un chunk de 1 s, o sea exigia haber medido el error, y su test `test_verificar_piso_ruteo_raise_si_por_debajo` canonizo el bug (50 ms es el orden del valor correcto y el test lo marcaba como roto). El guard corregido se ancla al LOOPBACK FISICO (CABLE Output): la referencia no es un derivado de la definicion, es el dispositivo real entregando el primer sample.
3. **El guard se INVIERTE hacia el error real, no hacia un piso conveniente.** El primer guard (`p95 < piso de drenado -> raise`) blindo el bug en vez de atraparlo: obligo a la medicion a incluir el drenado. El guard corregido atrapa lo OPUESTO (`p95 >= duracion del chunk -> raise`).
4. **Un numero por debajo de la resolucion del aparato no es una medicion rapida: es una medicion que no ocurrio.** Se reporta la resolucion o se declara la frontera no medida; nunca un valor sub-resolucion.
5. **El estadistico se fija ANTES de medir y no cambia con el resultado.** El p50 que desaparecio justo cuando el protocolo empeoro es el mismo error circular que las ventanas seleccionadas por latencia: si el estadistico cambia con la ronda, desde afuera es indistinguible de elegir por resultado. Se reporta el peor caso cuando el n no alcanza para p95.
6. **Todo numero tiene su n declarado** (iteraciones por corrida vs corridas) y su vocabulario exacto.
