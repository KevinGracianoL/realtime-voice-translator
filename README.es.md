<div align="center">

# 🎙️ Traductor de Voz en Tiempo Real

### ES ↔ EN para entrevistas de trabajo — **100 % local, privado y medido en mi propio hardware**

> `audio → VAD → ASR → traducción → TTS → audio` — el texto siempre está en pantalla, así detectas un error de traducción **antes** de responder.

**🌐 [English](README.md) · [Español](README.es.md)**

[![CI](https://github.com/KevinGracianoL/traductor-voz-entrevistas/actions/workflows/ci.yml/badge.svg)](https://github.com/KevinGracianoL/traductor-voz-entrevistas/actions)
![Python 3.11+](https://img.shields.io/badge/Python-3.11%2B-3776AB?style=flat-square&logo=python&logoColor=white)
![PyTorch CUDA](https://img.shields.io/badge/PyTorch-CUDA%2013.2-EE4C2C?style=flat-square&logo=pytorch)
![mypy strict](https://img.shields.io/badge/mypy-strict-2A6DB5?style=flat-square)
![coverage 100%](https://img.shields.io/badge/coverage-100%25-brightgreen?style=flat-square)
![mutantes 0 supervivientes](https://img.shields.io/badge/mutantes-0%20supervivientes-brightgreen?style=flat-square)
![399 tests](https://img.shields.io/badge/tests-399-2A6DB5?style=flat-square)
![25 ADRs](https://img.shields.io/badge/decisiones-25%20ADRs-2A6DB5?style=flat-square)
![License MIT](https://img.shields.io/badge/license-MIT-yellow?style=flat-square)

**Demo en vivo → [traductor-demo.kevingraciano.dev](https://traductor-demo.kevingraciano.dev)** · **Por [Kevin Graciano](https://github.com/KevinGracianoL)**

*Un traductor pensado para entrevistas reales — no para demos. Cada decisión tiene su ADR, cada ADR tiene sus números, y los números se tomaron en la máquina que va a sostener la entrevista.*

</div>

---

## 🏆 El veredicto (y cómo se ganó)

Después de **tres rondas de medición en hardware real** —y de que las dos primeras métricas que definí resultaran estar mal especificadas—, el motor elegido pasó el go/no-go con evidencia atribuida al 100 %:

| Gate | Medido | Límite | Veredicto |
|---|---|---|---|
| **Pipeline end-to-end** (audio → primer sample audible en VB-CABLE, una corrida encadenada) | **peor caso 1469.3 ms** (rango 1237.0–1469.3, 4 corridas) | < 2000 ms | ✅ **PASA** (margen 531 ms) |
| TTFA primer chunk (presupuesto **derivado**: 2000 − ASR − traducción − ruteo) | 723–755 ms < 767–826 ms | derivado | ✅ PASA |
| VRAM co-residente (XTTS + Whisper) | 2906.9–2946.9 MiB | < 3276.8 | ✅ PASA |
| RAM total (contexto anotado) | 13.0–13.9 GB | < 18 GB | ✅ PASA |
| Tasa de fallo (disponibilidad) | **5 % (1 de 20 corridas)** — la etapa de traducción murió por una descarga on-demand de spacy | 0 % | ⚠️ Declarada aparte → precarga bloqueante del flujo |
| A/B de voz (tu oído + evidencia objetiva) | timbre, ritmo 96–150 wpm, palabras técnicas claras | firma humana | ✅ Firmado |

**¿Por qué importa?** El techo de 1,5–2 s del ADR-003 **no se inventó ni se copió de un benchmark**: se midió la cadena completa *en esta GPU de 4 GB*, con el ASR co-residente y el micrófono virtual real. La historia completa de las dos métricas mal definidas que detecté yo mismo, corregí y convertí en reglas, vive en [ADR-014](docs/ADR-014-gates-aceptacion-tts.md) — cinco capas de evidencia, ninguna borrada.

**Nota de streaming (medida en la misma GPU):** el `inference_stream` de XTTS rinde **RTF ≈ 1.8** en la GTX 1650 Ti —más lento que tiempo real—: el stream de reproducción se queda sin datos y la voz sale con pausas ("una frase bien, después palabra por palabra"). El backend sintetiza **por frase en batch**: `inference()` sobre cada frase completa con las latentes del perfil **cacheadas** (una vez por perfil, **747–807 ms** re-medidos) y los settings de generación del config del modelo — **primer chunk ~2.0 s** y **RTF sostenido p95 0.85** (n=20 con la co-residencia de la demo: XTTS + whisper `small` + `tiny` en GPU; 0/20 corridas cruzan 1). Con el whisper del incoming transcribiendo **en paralelo** (el entrevistador hablando encima del TTS) el RTF sube a ~2.8: el resto del turno puede salir con pausas — límite declarado de la GPU de 4 GB que la demo no ejerce. El cómputo es estable; varía la duración del audio muestreado. El review del PR #34 cazó que el primer intento llamaba `tts.tts(speaker_wav=…)`, que **recalcula las latentes en cada llamada** (~100 ms por frase, instrumentado): el camino declarado las conserva. En hardware sin Tensor Cores el streaming continuo del motor queda fuera de presupuesto; en GPUs con RTF < 1 sostenido el streaming en vivo queda disponible con el mismo contrato.

---

## 🎬 La demo (video)

**[`docs/demo/demo_portafolio.mp4`](docs/demo/demo_portafolio.mp4)** — una corrida completa grabada en esta máquina (1 min 4 s, audio real del sistema):

1. El **entrevistador** habla en inglés (llega por el VB-CABLE) y el teleprompter muestra su texto con la traducción en español.
2. **Respondes en español** por tu micrófono; la respuesta aparece en pantalla al cerrar el turno.
3. Tu **voz clonada en inglés** sale por el VAIO hacia el micrófono virtual, **fluida** — es el batch por frase de arriba (con `inference_stream` el buffer se agotaba y se oía entrecortada).

La pausa de ~14 s entre tu respuesta y la voz en inglés es la espera declarada, medida en los logs de esta misma corrida: 2.5 s de detección de fin de turno + cierre de 4.2–11.5 s (traducción 0.2–0.7 s + TTS 4.2–10.7 s + ruteo < 0.03 s). El primer chunk suena en ~2 s con la GPU libre; en la demo la comparte con los ASR del flujo.

---

## 🎯 Qué resuelve

En una entrevista en inglés, un error de traducción no es un bug — **es la respuesta equivocada**.

Este proyecto prioriza **texto verificable** sobre voz sintética indistinguible, y **latencia medida en tu hardware** sobre benchmarks de RTX 4090 que no se cumplen en tu laptop.

1. **Privacidad real** — el audio nunca sale de la máquina. Todo corre local (CPU + GPU propia), sin nube, sin APIs.
2. **Cero magia** — cada etapa es una función pura y testeada: micrófono → VAD → ASR → traducción → teleprompter → voz.
3. **Evidencia sobre opinión** — 25 ADRs, cada decisión con su porqué medido. El motor de voz se rechazó **dos veces** con números antes de aceptarse con números.

---

## 🏗️ Arquitectura

Dos flujos explícitos (ADR-015) con workers persistentes, colas de tamaño 1, cancelación y degradación automática sin reiniciar la llamada:

```mermaid
flowchart LR
    subgraph OUT["outgoing_es_to_en (hablar)"]
        M1[🎤 Micrófono] --> VAD[VAD]
        VAD --> ASR[ASR español<br/>faster-whisper int8]
        ASR --> TR[Argos ES→EN]
        TR --> TP[📺 Teleprompter ES+EN]
        TR --> TTS[XTTS-v2<br/>tu voz en inglés]
        TTS --> CABLE[🔌 VB-CABLE]
        CABLE --> MEET[Meet/Zoom]
    end
    subgraph IN["incoming_en_to_es (escuchar)"]
        REM[🔊 Audio remoto] --> ASREN[ASR inglés]
        ASREN --> TR2[Argos EN→ES]
        TR2 --> SUB[📝 Subtítulos locales]
    end
```

**Escalera de degradación (sin reiniciar la entrevista):** voz clonada + subtítulos → voz genérica + subtítulos → solo subtítulos. Y antes de que un audio defectuoso salga al micrófono virtual: **validación ASR-de-retorno** (palabras añadidas/omitidas/repetidas, clipping, silencios anómalos) — no se reproduce audio sospechoso solo para mantener la clonación (en el modo sin streaming la validación es previa al ruteo; en streaming protege el resto del turno y deja telemetría, ver ADR-019).

---

## 🧪 Cómo se midió (lo que nadie copia de un README)

El go/no-go del motor no fue una tabla en un doc: fue **un harness con auto-verificación** (`scripts/medir_gates_tts.py`) que se corrigió a sí mismo cinco veces. Cada corrección quedó como **regla escrita** en el [ADR-014](docs/ADR-014-gates-aceptacion-tts.md), con sus números:

1. **El gate no se declara, se deriva** — `presupuesto TTFA = 2000 ms − (ASR + traducción + ruteo medidos)`. El `< 400 ms` literal era un sub-presupuesto inventado; hoy es un cálculo en `gates.py` (en esta máquina: **767–826 ms** derivados de un pipeline de 1237–1469 ms), no una constante.
2. **El guard debe fallar ante el error conocido** — el primer guard exigía haber medido el *drenado* (la duración del chunk, no la latencia); su test canonizó el bug. El guard corregido se ancla a una **referencia independiente** (el loopback físico del CABLE): un guard cuyo umbral se deriva de la misma definición que valida solo puede confirmarla.
3. **Un número por debajo de la resolución del aparato no es una medición rápida — es una medición que no ocurrió.**
4. **El estadístico se fija antes de medir** — y el peor caso se reporta cuando el n no alcanza para p95.
5. **La latencia se atribuye al 100 %** — `RegistroEtapas` marca cada frontera (entrada → ASR → traducción → TTS → entrega → audible) con cierre exacto por iteración; un residuo sin atribuir hace `raise`, no un print.

**Instrumentos clave:** reloj inyectable para todo (`p95` con `n≥20`, `math.ceil`, nada de `n<20` reportado como p95) · detección del primer sample audible por **loopback** (escribo a CABLE Input y leo CABLE Output) con guard de resolución · **ventanas derivadas del VAD real**, nunca seleccionadas por su latencia.

---

## ✅ Calidad — 5 gates, 1 contrato

| Pregunta | Herramienta | Config |
|---|---|---|
| ¿Legible y sin bugs? | **ruff** | `select = ["E","F","B","SIM","UP","I","S"]` |
| ¿Los tipos encajan? | **mypy --strict** | errores de tipo = CI rojo |
| ¿Hace lo que dice? | **pytest** | `--cov-fail-under=90` |
| ¿Qué no probé? | **coverage** | **100 %** (1259 stmts, 0 sin cubrir) |
| ¿Detectaría un bug? | **mutmut** | **0 supervivientes** — el gate CI falla si `survived > 0` |

> `mutmut` muta tu código a propósito (cambia `<=`→`<`, `*1000`→`/1000`, borra branches…) y exige que **alguien** lo detecte. El gate se verificó rompiendo el código a propósito y viendo a `mutmut results` rechazarlo — y varias veces encontró mutantes *equivalentes* que hubo que eliminar por reestructura, no por pragma.
>
> **399 tests** cubren el happy path **y** los modos de fallo: locks de antivirus, escrituras truncadas, `.tmp` huérfanos, mutantes que se contaminan entre sí por un WAV residual, rutas Windows con backslash/apóstrofo.

---

## 🔥 Lo que los bugs enseñaron (y que quedó como test)

Este proyecto se desarrolló con un revisor estricto a lo largo de **muchas rondas de review**. Cada bug real dejó una regresión test, no un parche:

- **Coexistencia CUDA 12/13 en una misma máquina.** torch 2.13 trae `cudart64_13`, pero `ctranslate2` necesita `cublas/cudart 12` → `RuntimeError: Library cublas64_12.dll is not found`. La solución (`setup_dlls.py`) copia **exactamente 3 DLLs** y registra los dirs de búsqueda vía `.pth` + `os.add_dll_directory`. Verificado en hardware real: `docs/smoke-windows.txt`.
- **Locks del antivirus en Windows.** Sobrescribir/borrar un `.dll` recién escrito falla mientras el AV lo escanea; *renombrarlo sí funciona*. `copiar_dlls` toma un backup inmutable, reintenta con backoff y **nunca deja el venv sin DLL ni con una DLL truncada** (3 invariantes de rollback).
- **100 % de cobertura ≠ cobertura de modos de fallo.** El bug de r5 solo aparecía con un *doble sucio* (escribe basura y luego revienta); los dobles limpios `raise`-y-ya lo dejaban pasar. Ese doble es hoy un test.
- **Un namespace package fantasma.** `makedirs` fabricaba un `ctranslate2/` vacío que enmascaraba una instalación rota. Ahora `dir_ct2()` deriva del paquete real y falla claro.
- **Un guard que blindaba el bug que debía atrapar.** El primer guard del ruteo exigía haber medido el drenado completo del chunk (1003.5 ms para 1 s de audio = la duración, no la latencia) y su test canonizó el error: 50 ms — el orden del valor correcto — se marcaba como "instrumento roto". La regla quedó escrita: *el guard se ancla a una referencia independiente de la definición que valida*.
- **El estadístico cambió justo cuando los números empeoraron.** El p50 apareció cuando el p95 hubiera sido más alto; desde afuera es indistinguible de elegir el estadístico por su resultado. Hoy se reporta el peor caso medido con su n declarado.
- **`ARGOS_COMPUTE_TYPE` sin "default" produce basura** en argos es→en ("mainstream" en bucle), y **argos cachea texto idéntico** (0.0 ms): el harness exige salida correcta antes de medir y mide frases distintas, como los turnos reales.

Cada uno de estos escenarios tiene su test **RED → GREEN**: se escribió el test, se vio fallar contra el código roto, y luego se arregló.

---

## 🛠️ Stack

| Capa | Tech | Nota |
|---|---|---|
| **ASR** | `faster-whisper` `int8` | TU117 sin Tensor Cores → FP16 emulado, INT8 en cores enteros |
| **Traducción** | `argos-translate` + `ctranslate2` | Offline, CPU, gratis; `ARGOS_COMPUTE_TYPE=default` obligatorio |
| **TTS** | **XTTS-v2** (fork `coqui-tts`) | Tu voz en inglés, batch por frase (`inference` + latentes cacheadas), perfil pre-enrolado |
| **Salida de audio** | VB-CABLE + VoiceMeeter | Dos tubos virtuales (entrevistador / tu voz EN), ruta por nombre y env |
| **Medición** | `time.perf_counter` inyectable + `RegistroEtapas` | Atribución al 100 %, p95 honesto (n≥20) |
| **UI** | `FastAPI` + teleprompter ES+EN | `localhost:8000`, deploy Caddy |
| **Calidad** | `ruff` · `mypy --strict` · `pytest` · `mutmut` | 5 gates, CI en GitHub Actions |

**Licencias declaradas (una por una):** código MIT · pesos XTTS-v2 **Coqui Public Model License** (uso personal no comercial — el proyecto declara la restricción, no la silencia) · OpenVoice V2 MIT · Supertonic 3 OpenRAIL-M. *CosyVoice 3 (Apache-2.0) queda anotado como candidato futuro para quitar el techo no-comercial.*

---

## 💻 Requisitos

- **Hardware de referencia:** Ryzen 5 4600H / GTX 1650 Ti 4 GB (TU117) / 24 GB RAM — 0 ms de red
- Python 3.11+, CUDA 13.2, Windows 10/11, micrófono, [VB-CABLE](https://vb-audio.com/Cable/) y [VoiceMeeter](https://vb-audio.com/Voicemeeter/) (drivers gratuitos) — un solo cable funciona, dos tubos evitan la retroalimentación entre flujos

---

## 🚀 Instalación

```powershell
# 1. Entorno
python -m venv venv; .\venv\Scripts\Activate.ps1

# 2. Deps (PyTorch con CUDA)
pip install -r requirements.txt --extra-index-url https://download.pytorch.org/whl/cu132
python setup_dlls.py

# 3. Verificar que local == CI
ruff check .; ruff format --check .; mypy .; pytest
```

## ▶️ Uso

```powershell
$env:PYTHONPATH = "src"
python scripts/verificar_hardware.py  # → CUDA: True | GTX 1650 Ti | VRAM 3.2/4 GB | mic → texto
python scripts/demo_traduccion.py     # → "Tell me about a hard bug..." ↔ "Háblame de un bug..."
```

**Re-medir el go/no-go del motor en tu hardware** (el harness del ADR-014, con auto-verificación incluida):

```powershell
$env:PYTHONPATH = "src"
python scripts/medir_gates_tts.py --motor xtts --warmup-audio tu_voz.wav --referencia tu_voz.wav
```

```python
from traductor.latencia.presupuesto import cabe_en_presupuesto
from traductor.latencia.medidor import medir_tiempo
from traductor.traduccion.argos import traducir
import time

# ¿Cabe en 1 s?
cabe_en_presupuesto({"asr": 500, "traduccion": 150, "tts": 300}, 1000)  # True

# Medir (los tests usan reloj falso, prod usa perf_counter)
texto, ms = medir_tiempo(lambda: traducir("hello", "en", "es"), clock=time.perf_counter)
```

---

## 🔊 Dos tubos de audio: VB-CABLE + VoiceMeeter (coste cero)

Con un solo cable virtual, tu voz en inglés (TTS) y la del entrevistador entran al mismo
tubo: el flujo incoming oiría **tu propio TTS** y lo transcribiría como si fuera el
entrevistador (retroalimentación física del tubo, no un bug de software). La solución es
separar los tubos — gratis, con [VoiceMeeter](https://vb-audio.com/Voicemeeter/) (del
mismo fabricante que VB-CABLE, donationware sin mínimo):

| Tubo | Device | Lleva |
|---|---|---|
| **VB-CABLE** (el que ya tienes) | `CABLE Input` → `CABLE Output` | La voz del **entrevistador** (simulador, o Meet/Zoom con salida de audio al cable) |
| **VoiceMeeter** (instalar + reiniciar) | `VoiceMeeter Input` (VAIO) → `VoiceMeeter Out` | **Tu voz en inglés** (TTS) — es el micrófono virtual que Meet/OBS ven |

No hace falta abrir la consola de VoiceMeeter: el par VAIO funciona como un cable de paso.
Oyes al entrevistador por el monitor de `CABLE Output` (Windows: propiedades del
dispositivo → "Escuchar este dispositivo").

Los dispositivos son configurables por env (defaults retrocompatibles con un solo cable):

```powershell
# De dónde LEE el incoming (el entrevistador llega por el VB-CABLE limpio):
$env:TRADUCTOR_DEVICE_INCOMING = "CABLE Output"       # default
# A dónde ESCRIBE el TTS del outgoing (mic virtual de Meet/OBS):
$env:TRADUCTOR_DEVICE_OUTGOING = "VoiceMeeter Input"  # default: "CABLE Input"
# Micrófono del outgoing (índice de pyaudio; la demo usa el de la laptop vía MME):
$env:TRADUCTOR_MIC_INDEX = "1"
```

**Correr los flujos (demo del video):**

```powershell
$env:PYTHONPATH = "src"
# 0. El WAV del entrevistador va gitignored (los *.wav nunca suben al repo):
#    regéneralo con el python del venv-tts (coqui):
venv-tts\Scripts\python.exe scripts/audio/generar_entrevistador.py
# 1. Teleprompter (subtítulos ES+EN, con la fuente de cada voz): http://localhost:8000
python -m uvicorn traductor.ui.app:app --host 127.0.0.1 --port 8000
# 2. Incoming: entrevistador → subtítulos en español (lee de TRADUCTOR_DEVICE_INCOMING)
python scripts/flujo_incoming.py
# 3. Outgoing: tu voz ES → inglés al mic virtual (escribe a TRADUCTOR_DEVICE_OUTGOING)
python scripts/flujo_outgoing.py
# 4. Simulador del entrevistador (una pasada; escribe al VB-CABLE)
python scripts/reproducir_entrevistador.py --veces 1 --delay 2
```

Ajustes de turno por env (sin tocar código): `TRADUCTOR_SILENCIO_TURNO_S` (1.5 s de
silencio cierra tu turno → el párrafo completo es UN turno, sin entrecortes),
`TRADUCTOR_FRAGMENTO_MAX_S` (4 s máx. por fragmento del incoming → whisper `small`
no alucina; el cambio de `tiny` a `small` costó ~550-860 ms medidos por fragmento
de 3 s, y cortó las alucinaciones del corpus) y `TRADUCTOR_UMBRAL_RMS` (300,
actividad de voz del cable).

Otros ajustes por env (validados al arrancar: un typo revienta con mensaje del
proyecto, no dentro de CTranslate2 a mitad de la grabación):
`TRADUCTOR_MODELO_ASR` (default `small` para la voz ES del outgoing; `tiny`
confundía "un bug" con "a walk") y `TRADUCTOR_ASR_DEVICE` (`cuda` o `cpu`): en
GPUs de 4 GB conviene **`cpu`** para el ASR del outgoing y dejar la GPU al TTS y
al whisper del incoming (margen de VRAM: 3.3/4.0 GiB con tres modelos
co-residentes). En CPU, medido por fragmento de 3 s de la voz ES real: `small`
**1.9 s** (RTF 0.63) y `base` **0.7 s** (RTF 0.23). Con micrófono de laptop, el
`TRADUCTOR_SILENCIO_TURNO_S` recomendado sube a **2.5 s** (las pausas naturales
entre frases de un párrafo no deben cerrar el turno).

**Nota técnica — por qué MME y no WASAPI (bug cazado con un tono puro de 440 Hz):**
el motor de VB-Audio corre internamente a **44100**. El mismo device expuesto por
WASAPI a 48000 pasa por un resampler que en Windows inserta **saltos de fase cada
~20 ms**: audio con clics inaudibles al oído pero que rompen la transcripción
(whisper oía "hard bug you solved" como palabras distintas). El código **prefiere
los devices MME** (tasa nativa 44100) automáticamente: el tono de prueba sale
**440.0 Hz exactos por MME** y 522 Hz con saltos por WASAPI. Si eliges devices a
mano, usa los MME.

---

## 📁 Estructura

```
├── src/traductor/
│   ├── hardware/cuda.py        # verifica GPU/VRAM
│   ├── audio/captura.py        # mic → texto (RealtimeSTT)
│   ├── audio/virtual.py        # ruta determinista por nombre (VB-CABLE)
│   ├── traduccion/argos.py     # EN↔ES offline (ARGOS_COMPUTE_TYPE fijado)
│   ├── asr/                    # benchmark ASR bidireccional (ADR-012)
│   │   └── ...                 # WER puro, manifest, medición, agregación
│   ├── tts/                    # contratos neutrales + motor elegido (ADR-011/014)
│   │   ├── modelos.py          # VoiceProfile, AudioResult, Salud
│   │   ├── backend.py          # TTSBackend (Protocol)
│   │   ├── tienda_json.py      # store real: un JSON por perfil (ADR-013)
│   │   ├── enrolamiento.py     # muestras → VoiceProfile validado (ADR-013)
│   │   ├── worker.py           # worker aislado: jobs JSON-line (ADR-013)
│   │   ├── gates.py            # go/no-go: presupuesto TTFA DERIVADO (ADR-014)
│   │   ├── backend_xtts.py     # XTTS-v2 vía fork coqui-tts (motor elegido)
│   │   ├── backend_b.py        # candidato B (Supertonic+OpenVoice): rechazado, evidencia conservada
│   │   └── harness.py          # parser + RegistroEtapas + guards de instrumento
│   ├── flujo/                  # core puro + adaptadores de hardware (ADR-015)
│   │   ├── outgoing.py         # ES→EN: escalera, artefactos, cierre por chunk
│   │   ├── incoming.py         # EN→ES: subtítulos en tiempo real
│   │   └── adaptadores.py      # micrófono, worker TTS, cable, teleprompter
│   ├── ui/                     # teleprompter FastAPI + WebSocket (ADR-007)
│   └── latencia/
│       ├── presupuesto.py      # ¿cabe? ¿quién es más lento?
│       └── medidor.py          # reloj inyectable, p50/p95 honesto (n≥20)
├── scripts/                    # hardware, traducción, harness de gates
├── setup_dlls.py               # CUDA 12/13 coexistiendo (Windows, locks AV)
├── docs/                       # 25 ADRs con evidencia medida + demo/ (video)
├── tests/                      # 399 tests, 100 % cov, mutantes en CI
└── .github/workflows/ci.yml    # 5 gates que fallan el PR si algo se rompe
```

---

## 📋 ADRs — decisiones con evidencia

| # | Decisión | Por qué |
|---|---|---|
| 001 | **Cascada, no end-to-end** | [ADR-001](docs/ADR-001-cascada-no-end-to-end.md): texto verificable > latencia mínima |
| 002 | **Audio virtual a nivel SO** | [ADR-002](docs/ADR-002-audio-virtual-nivel-so.md): funciona con cualquier Meet/Zoom sin API |
| 003 | **Techo 1,5–2 s** | [ADR-003](docs/ADR-003-techo-presupuesto-latencia.md): recorta calidad, nunca latencia; presupuesto TTFA derivado |
| 004 | **INT8, no FP16** | [ADR-004](docs/ADR-004-int8-no-fp16.md): TU117 sin Tensor Cores, FP16 emulado |
| 005 | **Local, no remoto** | [ADR-005](docs/ADR-005-local-no-remoto.md): AVX2+CUDA+0 ms gana a geografía |
| 006 | **Sobre RealtimeSTT** | [ADR-006](docs/ADR-006-sobre-realtimestt.md): VAD/ASR commodity, nosotros orquestamos |
| 007 | **Teleprompter primero** | [ADR-007](docs/ADR-007-teleprompter-primero.md): semanas vs meses, honestidad en entrevista |
| 008 | **Fallback automático** | [ADR-008](docs/ADR-008-fallback-automatico.md): una entrevista no es un log |
| 009 | **Dirección por fuente** | [ADR-009](docs/ADR-009-direccion-por-fuente.md): determinista, 0 ms, sin detector que falle en code-switching |
| 010 | **Chatterbox rechazado como TTS** | [ADR-010](docs/ADR-010-chatterbox-rechazado.md): 17,4 s warm / 3,6 GB **medidos** en esta GPU, sin co-residencia con Whisper |
| 011 | **Motor elegido: XTTS-v2** | [ADR-011](docs/ADR-011-contratos-neutrales-tts.md): aceptado por los gates medidos; CPML declarado (uso personal); Supertonic+OpenVoice como B (rechazado); Pocket descartado |
| 012 | **Benchmark ASR bidireccional** | [ADR-012](docs/ADR-012-benchmark-asr.md): Moonshine Small CPU vs faster-whisper Small GPU; WER normalizado + p50/p95 |
| 013 | **Worker TTS aislado + enrolamiento** | [ADR-013](docs/ADR-013-worker-tts-enrolamiento.md): jobs JSON-line, frontera de entrada, tienda local |
| 014 | **Gates de aceptación del motor** | [ADR-014](docs/ADR-014-gates-aceptacion-tts.md): **cinco capas de corrección de instrumento** + presupuesto TTFA derivado; pipeline end-to-end 1469.3 ms (peor caso) < 2000 ms |
| 015 | **Arquitectura por flujos y escalera** | [ADR-015](docs/ADR-015-arquitectura-flujos-escalera.md): outgoing/incoming, colas de tamaño 1, validación de artefactos, 4 niveles |
| 019 | **Gates de sesión y endurance** | [ADR-019](docs/ADR-019-endurance-sesion.md): 90 min continuos, sin OOM, memoria estable, artefactos, A/B firmado por el usuario — la aprobación final |

> La numeración no es contigua: **016–018 quedaron como números de reserva nunca usados** (sin documento ni referencia en el historial del repo); los ADRs reales son **001–015 + 019**. Los documentos 001–009 se reconstruyeron el 2026-09-18 desde los commits, el código y los tests originales (la era inicial no escribió los archivos).

---

## 🗺️ Roadmap

- [x] **Pasos 1–5** — hardware, traducción, medición, audio virtual, teleprompter
- [x] **Fases 2a–2e** — DLLs CUDA, contratos TTS, benchmark ASR, worker + enrolamiento, gates
- [x] **Fase 2f — Go/no-go del motor** — **XTTS-v2 aceptado por los gates medidos** (pipeline peor caso 1469.3 ms < 2000 ms, margen 531 ms; candidato B rechazado por TTFA arquitectural 7885.8 ms) — evidencia en [ADR-014](docs/ADR-014-gates-aceptacion-tts.md)
- [x] **Fase 2g — Motor elegido** — XTTS-v2; aprobación final respaldada por el [ADR-019](docs/ADR-019-endurance-sesion.md)
- [x] **Fase 3 — El flujo `outgoing_es_to_en`** — mic → VAD → ASR es → Argos → teleprompter → XTTS → VB-CABLE, con la escalera de degradación (ADR-015)
- [x] **Fase 4 — Endurance 90 min + fixes del cierre** — la corrida larga del ADR-019 (417 turnos, memoria estable) + job streaming del worker, cable por bloques y validación de artefactos en vivo: re-corrida de 90 min con **692 turnos**, cierre del turno p95 **21.3 s → 1.60 s**, atrasadas 417/417 → **1/692**
- [x] **Fase 5 — Flujo `incoming_en_to_es`** — el audio del entrevistador (Meet/Zoom por el cable virtual) → ASR en → Argos EN→ES → **subtítulos en español** en el teleprompter (ADR-015)
- [ ] **Clon remoto opcional (extensión)** — clon dinámico de la voz del entrevistador: subtítulos primero; el clon solo si las muestras recogidas pasan los controles del ADR-015

---

<div align="center">

**Hecho por [Kevin Graciano](https://github.com/KevinGracianoL)** — aprendiendo en público, midiendo en mi propio hardware.

*Privacidad por diseño: cero audios de entrevistas reales y cero credenciales en el historial del repo.*

</div>
