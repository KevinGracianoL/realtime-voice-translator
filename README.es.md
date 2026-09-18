<div align="center">

# Traductor de voz ES ↔ EN para entrevistas

**Todo se ejecuta en tu computadora. El audio no sale de tu equipo.**

[English](README.md) · [Español](README.es.md)

[![CI](https://github.com/KevinGracianoL/traductor-voz-entrevistas/actions/workflows/ci.yml/badge.svg)](https://github.com/KevinGracianoL/traductor-voz-entrevistas/actions)
![Python 3.11+](https://img.shields.io/badge/Python-3.11%2B-3776AB?style=flat-square&logo=python&logoColor=white)
![PyTorch CUDA](https://img.shields.io/badge/PyTorch-CUDA%2013.2-EE4C2C?style=flat-square&logo=pytorch)
![mypy strict](https://img.shields.io/badge/mypy-strict-2A6DB5?style=flat-square)
![399 tests](https://img.shields.io/badge/tests-399-2A6DB5?style=flat-square)
![coverage 100%](https://img.shields.io/badge/coverage-100%25-brightgreen?style=flat-square)
![License MIT](https://img.shields.io/badge/license-MIT-yellow?style=flat-square)

**Por [Kevin Graciano](https://github.com/KevinGracianoL)** — clona el repositorio y ejecútalo en tu equipo (ver Instalación y uso).

</div>

---

## Qué es

Este programa te ayuda en una entrevista de trabajo en inglés. Escucha al entrevistador desde el audio de la llamada y muestra la traducción al español en pantalla. Cuando respondes en español por tu micrófono, transcribe tu respuesta, la traduce al inglés y la convierte en voz con un timbre clonado del tuyo, que entra a la llamada por un micrófono virtual. Todo funciona en tu propia computadora: no se envía audio ni texto a servicios en la nube.

El propósito es que puedas verificar por texto lo que entendiste y lo que vas a decir antes de decirlo, sin depender de una conexión a internet ni de una API externa.

---

## Video de demostración

[`docs/demo/demo_portafolio.mp4`](docs/demo/demo_portafolio.mp4) (1 min 4 s) — grabación real en la máquina de referencia, con el audio del sistema:

- El entrevistador habla en inglés y en pantalla aparece su texto con la traducción al español.
- Al terminar de responder en español por el micrófono, se muestra el texto de tu respuesta.
- Tu voz en inglés sale por el micrófono virtual hacia la llamada.

**Limitación que se ve en el video:** entre el final de tu respuesta y el momento en que se escucha tu voz en inglés pasa una espera de alrededor de 14 segundos en esa grabación. De ese tiempo, 2,5 s son la confirmación de que terminaste de hablar y el resto es la traducción y la generación de la voz (4,2 a 11,5 s medidos en la corrida, según la frase). La voz traducida no es instantánea.

---

## Qué puede hacer y cuáles son sus límites

**Capacidades verificadas**

- Transcribir al entrevistador en inglés y mostrar subtítulos en español en `localhost`.
- Transcribir tu voz en español y mostrar el texto en pantalla.
- Generar tu respuesta en inglés con una voz clonada a partir de muestras tuyas, y enrutarla al micrófono virtual de la llamada.
- Funcionar con Meet, Zoom o Teams a través de dispositivos de audio virtuales, sin plugins ni integración con la plataforma.
- Seguir funcionando si una etapa falla: si la voz clonada no está disponible, usa una voz genérica y, como último recurso, deja solo los subtítulos. El flujo no se detiene.

**Límites actuales**

- La voz en inglés no es inmediata: el cierre del turno (el momento en que empieza a escucharse) tardó entre 4,2 y 11,5 segundos en la grabación de demostración, y unos 2 segundos para el primer audio con la GPU libre.
- Si el entrevistador habla mientras el sistema genera tu voz, la generación se ralentiza (ver Resultados técnicos) y el final del audio puede escucharse con pausas. En la grabación de demostración el entrevistador no habla mientras se genera la voz.
- Requiere Windows, una GPU NVIDIA compatible (la referencia es una GTX 1650 Ti de 4 GB) y los drivers virtuales [VB-CABLE](https://vb-audio.com/Cable/) y [VoiceMeeter](https://vb-audio.com/Voicemeeter/).
- Los pesos del modelo de voz XTTS-v2 usan la licencia Coqui Personal Model License: **uso personal no comercial**. El código de este repositorio es MIT.
- Esta versión trabaja con un perfil de voz y un idioma de salida (inglés).

---

## Cómo funciona

El programa separa dos tareas que corren a la vez:

- **Escuchar al entrevistador:** toma el audio de la llamada, lo transcribe en inglés, lo traduce al español y lo muestra en pantalla.
- **Traducir tu respuesta:** toma tu micrófono, transcribe tu español, lo traduce al inglés y lo convierte en voz con tu timbre, que sale por un micrófono virtual hacia la llamada.

Cada tarea avanza por etapas separadas (detección de voz, transcripción, traducción y voz), conectadas por colas de un solo elemento: una respuesta nueva cancela la anterior y nada se procesa dos veces. Las tareas escuchan y hablan por dispositivos de audio distintos, de modo que el sistema no se escucha a sí mismo.

En el diagrama: **VAD** es la detección de actividad de voz (cuándo hay alguien hablando), **ASR** es el reconocimiento de voz (transcripción) y **TTS** es la síntesis de voz (generar audio).

```mermaid
flowchart LR
    subgraph OUT["outgoing_es_to_en (hablar)"]
        M1[Micrófono] --> VAD[VAD]
        VAD --> ASR[ASR español<br/>faster-whisper int8]
        ASR --> TR[Argos ES→EN]
        TR --> TP[Teleprompter ES+EN]
        TR --> TTS[XTTS-v2<br/>tu voz en inglés]
        TTS --> CABLE[VB-CABLE]
        CABLE --> MEET[Meet/Zoom]
    end
    subgraph IN["incoming_en_to_es (escuchar)"]
        REM[Audio remoto] --> ASREN[ASR inglés]
        ASREN --> TR2[Argos EN→ES]
        TR2 --> SUB[Subtítulos locales]
    end
```

Antes de que un audio defectuoso llegue al micrófono virtual, el sistema lo transcribe de vuelta (loopback: escribe en el cable y lo lee) y compara con el texto traducido; si encuentra diferencias grandes, no reproduce ese audio y pasa al siguiente nivel de la lista de capacidades.

Las piezas por etapa:

| Etapa | Herramienta | Nota |
|---|---|---|
| Transcripción | `faster-whisper` en `int8` | La GPU de referencia no tiene Tensor Cores, así que se usa INT8 en lugar de FP16 |
| Traducción | `argos-translate` + `ctranslate2` | Sin conexión, en CPU; `ARGOS_COMPUTE_TYPE=default` es obligatorio |
| Voz | XTTS-v2 (fork `coqui-tts`) | Genera frase por frase con las latentes del perfil en caché |
| Audio virtual | VB-CABLE + VoiceMeeter | Dos tubos: entrevistador y tu voz en inglés |
| Interfaz | FastAPI | Teleprompter en `localhost:8000` |
| Medición | `time.perf_counter` y registros por etapa | Los números de este README salen de ahí |

El detalle de la metodología de medición, las rondas de corrección del banco de pruebas y los incidentes que dejaron tests de regresión están en [ADR-014](docs/ADR-014-gates-aceptacion-tts.md) y [ADR-019](docs/ADR-019-endurance-sesion.md).

---

## Instalación y uso

Requisitos: Windows 10/11, Python 3.11 o superior, CUDA 13.2, un micrófono, y los drivers gratuitos VB-CABLE y VoiceMeeter. La máquina de referencia es un Ryzen 5 4600H con una GTX 1650 Ti de 4 GB y 24 GB de RAM.

```powershell
# 1. Entorno
python -m venv venv; .\venv\Scripts\Activate.ps1

# 2. Dependencias (PyTorch con CUDA)
pip install -r requirements.txt --extra-index-url https://download.pytorch.org/whl/cu132
python setup_dlls.py

# 3. Comprobar que el entorno local coincide con el CI
ruff check .; ruff format --check .; mypy .; pytest
```

```powershell
$env:PYTHONPATH = "src"
python scripts/verificar_hardware.py  # → CUDA: True | GTX 1650 Ti | VRAM 3.2/4 GB | mic → texto
python scripts/demo_traduccion.py     # → "Tell me about a hard bug..." ↔ "Háblame de un bug..."
```

Para repetir la medición de aceptación del motor en tu hardware (el banco de pruebas del ADR-014, con comprobaciones propias):

```powershell
$env:PYTHONPATH = "src"
python scripts/medir_gates_tts.py --motor xtts --warmup-audio tu_voz.wav --referencia tu_voz.wav
```

**Ejecutar la demostración completa (los dos flujos):**

```powershell
$env:PYTHONPATH = "src"
# 0. El WAV del entrevistador no está en el repositorio (*.wav): regéneralo con
#    el python del venv del TTS (coqui):
venv-tts\Scripts\python.exe scripts/audio/generar_entrevistador.py
# 1. Teleprompter (subtítulos ES+EN, con la fuente de cada voz): http://localhost:8000
python -m uvicorn traductor.ui.app:app --host 127.0.0.1 --port 8000
# 2. Flujo entrante: entrevistador → subtítulos en español (lee de TRADUCTOR_DEVICE_INCOMING)
python scripts/flujo_incoming.py
# 3. Flujo saliente: tu voz ES → inglés al micrófono virtual (escribe en TRADUCTOR_DEVICE_OUTGOING)
python scripts/flujo_outgoing.py
# 4. Simulador del entrevistador (una pasada; escribe al VB-CABLE)
python scripts/reproducir_entrevistador.py --veces 1 --delay 2
```

**Dispositivos de audio (variables de entorno):**

```powershell
# De dónde lee el flujo entrante (el entrevistador llega por el VB-CABLE):
$env:TRADUCTOR_DEVICE_INCOMING = "CABLE Output"       # valor por defecto
# A dónde escribe la voz del flujo saliente (micrófono virtual de Meet/OBS):
$env:TRADUCTOR_DEVICE_OUTGOING = "VoiceMeeter Input"  # por defecto: "CABLE Input"
# Micrófono del flujo saliente (índice de pyaudio; la demo usa el del equipo, vía MME):
$env:TRADUCTOR_MIC_INDEX = "1"
```

**Ajustes de turno y de transcripción:**

- `TRADUCTOR_SILENCIO_TURNO_S` (1,5 s): cuánto silencio cierra tu turno. Con este valor, un párrafo completo es un solo turno. Con micrófono de equipo conviene subirlo a 2,5 s para que las pausas naturales entre frases no cierren el turno antes de tiempo.
- `TRADUCTOR_FRAGMENTO_MAX_S` (4 s): tamaño máximo de cada fragmento del flujo entrante. Fragmentos más largos hacen que el modelo de transcripción invente texto.
- `TRADUCTOR_UMBRAL_RMS` (300): nivel de voz que se considera actividad en el cable.
- `TRADUCTOR_MODELO_ASR` (`small`) y `TRADUCTOR_ASR_DEVICE` (`cuda` o `cpu`): modelo y dispositivo de la transcripción del flujo saliente. El modelo `tiny` confundía "un bug" con "a walk"; en GPUs de 4 GB conviene `cpu` para dejar memoria a la voz y al flujo entrante. Medido en CPU sobre 3 s de voz real: `small` tarda 1,9 s y `base` 0,7 s. Un valor mal escrito se rechaza al arrancar con un mensaje claro.

**Nota sobre los dispositivos MME y WASAPI:** el motor de VB-Audio funciona internamente a 44100 Hz. El mismo dispositivo expuesto por WASAPI a 48000 Hz pasa por un remuestreador que inserta saltos de fase cada ~20 ms; se oyen como clics apenas perceptibles, pero rompen la transcripción (el modelo oía "hard bug you solved" como palabras distintas). Por eso el código elige los dispositivos MME (tasa nativa de 44100). Un tono de prueba de 440 Hz sale a 440,0 Hz exactos por MME y a 522 Hz con saltos por WASAPI.

---

## Resultados técnicos

| Métrica | Valor medido |
|---|---|
| Flujo completo (audio de entrada → primera muestra audible en el cable) | peor caso 1469,3 ms en 4 corridas (rango 1237,0–1469,3), límite 2000 ms |
| Primer audio de la voz (GPU libre, primera frase) | ~2,0 s |
| Cierre del turno en la grabación de demostración | 4,2–11,5 s, más 2,5 s de confirmación de fin de turno |
| VRAM con el motor de voz y el modelo de transcripción del flujo entrante | 2906,9–2946,9 MiB (límite 3276,8 MiB) |
| RAM del sistema en la corrida de referencia | 13,0–13,9 GB |
| Velocidad de generación de voz (RTF: segundos de cómputo por segundo de audio) | p95 0,85 con los tres modelos cargados (n=20, ninguna corrida por encima de 1); ~2,8 si el flujo entrante transcribe en paralelo |
| Latentes del perfil de voz | 747–807 ms, una vez por perfil |

**Calidad del código**

| Comprobación | Herramienta | Resultado |
|---|---|---|
| Formato y estilo | ruff | sin avisos |
| Tipos | mypy --strict | sin errores |
| Comportamiento | pytest | 399 tests |
| Cobertura de líneas | coverage | 100 % (1259 líneas) |
| Calidad de los tests | mutmut | 0 mutantes supervivientes |

La metodología completa de medición (cómo se deriva el presupuesto de latencia, las cinco rondas de corrección del banco de pruebas, la validación por loopback y los p50/p95) está en [ADR-014](docs/ADR-014-gates-aceptacion-tts.md). La corrida de resistencia de 90 minutos, los cierres de turno y su re-medición están en [ADR-019](docs/ADR-019-endurance-sesion.md).

---

## Documentación técnica

```
├── src/traductor/
│   ├── hardware/cuda.py        # verifica GPU/VRAM
│   ├── audio/captura.py        # micrófono → texto (RealtimeSTT)
│   ├── audio/virtual.py        # ruta determinista por nombre (VB-CABLE)
│   ├── traduccion/argos.py     # EN↔ES sin conexión (ARGOS_COMPUTE_TYPE fijado)
│   ├── asr/                    # comparativa de transcripción (ADR-012)
│   ├── tts/                    # contratos y motor de voz elegido (ADR-011/014)
│   │   ├── worker.py           # worker aislado: trabajos JSON-line (ADR-013)
│   │   ├── backend_xtts.py     # XTTS-v2 vía fork coqui-tts (motor elegido)
│   │   └── harness.py          # banco de pruebas: registro de etapas y validaciones
│   ├── flujo/                  # núcleo y adaptadores de hardware (ADR-015)
│   │   ├── outgoing.py         # ES→EN: escalera, artefactos, cierre por primer fragmento
│   │   └── incoming.py         # EN→ES: subtítulos en tiempo real
│   ├── ui/                     # teleprompter FastAPI + WebSocket (ADR-007)
│   └── latencia/               # presupuesto y medidor (p50/p95, n≥20)
├── scripts/                    # hardware, traducción, banco de pruebas
├── setup_dlls.py               # convivencia CUDA 12/13 (Windows, bloqueos de antivirus)
├── docs/                       # 16 ADRs con su evidencia + demo/ (video)
├── tests/                      # 399 tests, mutación en CI
└── .github/workflows/ci.yml    # los 5 controles de integración
```

**ADRs (decisiones con su porqué y su evidencia):**

| # | Decisión | Por qué |
|---|---|---|
| 001 | **Cascada, no extremo a extremo** | [ADR-001](docs/ADR-001-cascada-no-end-to-end.md): el texto intermedio es verificable |
| 002 | **Audio virtual a nivel del sistema** | [ADR-002](docs/ADR-002-audio-virtual-nivel-so.md): funciona con cualquier Meet/Zoom, sin API por plataforma |
| 003 | **Techo de 1,5–2 s** | [ADR-003](docs/ADR-003-techo-presupuesto-latencia.md): se recorta calidad antes que latencia; presupuesto derivado de lo medido |
| 004 | **INT8, no FP16** | [ADR-004](docs/ADR-004-int8-no-fp16.md): la GPU de referencia no tiene Tensor Cores y FP16 se emula |
| 005 | **Local, no remoto** | [ADR-005](docs/ADR-005-local-no-remoto.md): sin conexión ni API; quien lo quiera usar clona el repositorio |
| 006 | **Sobre RealtimeSTT** | [ADR-006](docs/ADR-006-sobre-realtimestt.md): la detección de voz es común; el valor está en la orquestación |
| 007 | **Teleprompter primero** | [ADR-007](docs/ADR-007-teleprompter-primero.md): el texto es útil desde el primer momento; la voz llegó después |
| 008 | **Degradación automática** | [ADR-008](docs/ADR-008-fallback-automatico.md): una entrevista no se puede reiniciar |
| 009 | **Dirección de traducción por fuente de audio** | [ADR-009](docs/ADR-009-direccion-por-fuente.md): determinista, sin detector de idioma que falle al mezclar idiomas |
| 010 | **Chatterbox rechazado** | [ADR-010](docs/ADR-010-chatterbox-rechazado.md): 17,4 s en caliente y 3,6 GB medidos en esta GPU |
| 011 | **Motor elegido: XTTS-v2** | [ADR-011](docs/ADR-011-contratos-neutrales-tts.md): aprobado por los criterios medidos; licencia CPML declarada; Supertonic+OpenVoice como alternativa rechazada |
| 012 | **Comparativa de transcripción** | [ADR-012](docs/ADR-012-benchmark-asr.md): Moonshine Small en CPU contra faster-whisper Small en GPU, con WER normalizado |
| 013 | **Worker de voz aislado y enrolamiento** | [ADR-013](docs/ADR-013-worker-tts-enrolamiento.md): trabajos JSON-line, frontera de entrada y almacén local |
| 014 | **Criterios de aceptación del motor** | [ADR-014](docs/ADR-014-gates-aceptacion-tts.md): cinco rondas de corrección del instrumento; pipeline 1469,3 ms en el peor caso |
| 015 | **Arquitectura por flujos y escalera** | [ADR-015](docs/ADR-015-arquitectura-flujos-escalera.md): dos flujos, colas de tamaño 1 y validación de artefactos |
| 019 | **Sesión y resistencia** | [ADR-019](docs/ADR-019-endurance-sesion.md): 90 minutos seguidos sin fallos de memoria, y la aprobación final |

La numeración no es contigua: los números 016 a 018 quedaron como reserva sin usar; los ADRs reales son 001–015 y 019. Los documentos 001 a 009 se reconstruyeron el 2026-09-18 a partir de los commits, el código y los tests de esa etapa.

**Estado del desarrollo**

- [x] Pasos 1–5: hardware, traducción, medición, audio virtual y teleprompter.
- [x] Fases 2a–2e: DLLs de CUDA, contratos de voz, comparativa de transcripción, worker y criterios.
- [x] Fase 2f: elección del motor de voz con las mediciones del ADR-014.
- [x] Fase 2g: aprobación final respaldada por el ADR-019.
- [x] Fase 3: flujo saliente completo, con degradación por niveles.
- [x] Fase 4: resistencia de 90 minutos y correcciones del cierre de turno.
- [x] Fase 5: flujo entrante con subtítulos en español.

---

## Licencias y atribución

Código de este repositorio: MIT.

- Pesos de XTTS-v2: licencia Coqui Personal Model License, uso personal no comercial.
- OpenVoice V2: MIT.
- Supertonic 3: OpenRAIL-M.

Hecho por [Kevin Graciano](https://github.com/KevinGracianoL). Aprendiendo en público y midiendo en mi propio equipo. En el historial del repositorio no hay audios de entrevistas reales ni credenciales.
