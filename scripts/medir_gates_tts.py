"""Mide los gates de aceptación del motor TTS en la máquina objetivo (ADR-014).

Carga el motor candidato (primario: XTTS-v2 vía fork coqui-tts, ADR-011) Y
faster-whisper co-residente. Mide con el medidor honesto (n>=20, math.ceil,
None si falta):
- las etapas que antes eran supuestos: ASR p95, traducción Argos p95, ruteo a
  VB-CABLE p95 — el presupuesto de TTFA se DERIVA de ellas (2000 ms − etapas),
  nunca se declara;
- el TTFA como PRIMER CHUNK reproducible (batch por frase caliente para XTTS,
  review PR #34: la primera frase completa de `sintetizar_stream`;
  síntesis completa para motores sin streaming, documentado);
- el PIPELINE COMPLETO end-to-end (audio de entrada -> primer audio en el
  micrófono virtual) con cable si VB-CABLE está instalado;
- la VRAM TOTAL a nivel driver (`vram_ocupada_mib`, incluye CTranslate2) y la
  RAM del sistema (psutil). La foto de Whisper se toma justo tras su warm-up
  (delta aislado); la de co-residencia, después de las síntesis.

Los gates de SESIÓN (pipeline p95, OOM, memoria, artefactos, A/B, endurance)
los mide la corrida larga del ADR-019 y se pasan por flags: sin ellos quedan
en None = FALLA. Un gate sin medición (p. ej. ruteo sin VB-CABLE instalado)
es un FALLO, nunca un pase silencioso.

Uso:
    $env:PYTHONPATH = "src"
    python scripts/medir_gates_tts.py --motor xtts --warmup-audio tu_voz.wav
    # + flags de sesión tras la corrida larga: --pipeline-p95 1500 --no-oom
    #   --memoria-estable --no-artefactos --voz-reconocible-ab --endurance-90min

El CI NO lo ejecuta: requiere GPU + modelo + Whisper + Argos. La salida se
pega como evidencia en el ADR-014.
"""

from __future__ import annotations

import sys
import time
from collections.abc import Callable
from functools import partial
from pathlib import Path
from typing import Any

import numpy as np

from traductor.hardware.cuda import vram_ocupada_mib
from traductor.latencia.medidor import agregar_medicion, medir_tiempo, resumen_estadisticas
from traductor.tts.gates import cabe_en_gates, evaluar_gates, resumen_gates
from traductor.tts.harness import (
    RegistroEtapas,
    componer_medicion,
    medir_ram_mib,
    parser_harness,
    perfil_benchmark,
)
from traductor.tts.modelos import VoiceProfile

N_REPETICIONES = 20
# Piso para la auto-verificación de Whisper. tiny int8 son decenas de MB de
# pesos; el bulto del contexto CUDA ya existe en vram_base. 50 es conservador,
# NO 500 (estimación de memoria citada sin verificar, r4).
DELTA_WHISPER_MIN_MIB = 50.0
# Candidato B (ADR-011): pesos de OpenVoice V2. Sobre-escribible por env para
# reproducir la medición en otra máquina sin tocar el código.
DIR_CHECKPOINTS_B = r"C:\Users\Kevin\deps\OpenVoice\checkpoints_v2"
TEXTO_TRADUCCION = "Mi experiencia mas fuerte es con sistemas distribuidos."
# argos 1.11 cachea el MISMO texto (la misma frase da 0.0 ms): la medición usa
# frases DISTINTAS, como los turnos reales del flujo.
FRASES_TRADUCCION = (
    "Mi experiencia mas fuerte es con sistemas distribuidos.",
    "Trabaje con bases de datos relacionales durante cinco anios.",
    "El proyecto mas grande que lideramos duro ocho meses.",
    "Aprendi python en mi primer trabajo como desarrollador.",
    "La comunicacion con el equipo fue clave para el exito.",
    "Prefiero escribir pruebas antes que el codigo de produccion.",
    "Nuestro equipo redujo el tiempo de despliegue a la mitad.",
    "El cliente pidio una interfaz mas simple para los usuarios.",
    "La documentacion tecnica debe actualizarse cada semana.",
    "Resolvimos el problema de rendimiento con un indice nuevo.",
    "Me gusta revisar el codigo de mis companeros con cuidado.",
    "La entrevista de hoy es para un puesto de ingeniero senior.",
    "El salario no es mi principal motivacion en este momento.",
    "Estoy acostumbrado a trabajar con equipos remotos.",
    "La empresa busca alguien con experiencia en la nube.",
    "Mi ultimo proyecto consistio en migrar servicios antiguos.",
    "Las reuniones diarias duran quince minutos en nuestro equipo.",
    "Considero que el feedback honesto mejora la calidad del trabajo.",
    "El despliegue continuo redujo los errores en produccion.",
    "Estoy disponible para comenzar la proxima semana.",
)
SR_FLUJO = 24000  # sample rate del chunk del flujo (XTTS) para el ruteo
UMBRAL_AUDIBLE = 64  # 0.2 % de full scale: el CABLE es passthrough digital (ruido ~0)


def _normalizar(texto: str) -> str:
    """Minúsculas, sin puntuación, espacios colapsados — para comparar salidas."""
    return " ".join("".join(c for c in texto.lower() if c.isalnum() or c.isspace()).split())


def _cargar_motor(motor: str) -> Any:
    """Carga el candidato pedido por `--motor` (xtts | b).

    `xtts`: primario (ADR-011), vía fork coqui-tts — requiere el venv propio
    del TTS y el modelo descargado. `b`: Supertonic 3 CPU + OpenVoice V2
    (ADR-011/014) — requiere supertonic + el repo OpenVoice instalado y los
    pesos en `DIR_CHECKPOINTS_B`.
    """
    if motor == "xtts":
        from traductor.tts.backend_xtts import BackendXtts

        return BackendXtts(idioma_salida="en")
    import os

    from traductor.tts.backend_b import BackendB

    return BackendB(dir_checkpoints=os.environ.get("OPENVOICE_CHECKPOINTS_V2", DIR_CHECKPOINTS_B))


def _cargar_whisper() -> Any:
    """faster-whisper tiny int8 en CUDA: carga el ASR co-residente (ADR-014)."""
    from faster_whisper import WhisperModel

    return WhisperModel("tiny", device="cuda", compute_type="int8_float16")


def _calentar_whisper_y_foto(whisper: Any, audio: Path) -> float | None:
    """Calienta la PRIMERA inferencia de Whisper y devuelve la VRAM tras ella.

    CTranslate2 reserva workspace/KV/beam en la primera `transcribe()`, no en
    `__init__` (r1 del PR #15). `transcribe()` devuelve un generador: se drena
    con `list()` o la inferencia no ocurre y el warm-up es decorativo.

    La foto se toma INMEDIATAMENTE después del warm-up (r4): así el delta
    contra la línea base aísla la VRAM de Whisper, sin mezclarla con las
    reservas perezosas que el motor haga en sus propias síntesis.

    Conteo de segmentos (r5/r6): el umbral de 50 detecta "Whisper ausente", no
    "presente pero frío" — los pesos se reservan en `__init__`, así que si
    `no_speech_threshold` corta el silencio, el delta igual pasa. 0 segmentos
    = el decoder no corrió y la VRAM subestimaría: `raise` con la salida
    (`--warmup-audio` con voz real), no un print que se pierde en la corrida.
    """
    import torch

    segmentos, _ = whisper.transcribe(str(audio), language="es")
    lista = list(segmentos)
    if not lista:
        raise RuntimeError(
            "el warm-up dio 0 segmentos: el decoder no se ejercitó y la VRAM "
            "subestimaría. El WAV pasado no produjo segmentos (¿silencio "
            "cortado por no_speech?). Usa --warmup-audio con un WAV de voz "
            "real; el instrumento no mide con warm-up frío."
        )
    print(f"warm-up: {len(lista)} segmentos")
    return vram_ocupada_mib(torch.cuda)


def _medir_p95(func: Callable[[], object], n: int, etapa: str) -> float | None:
    """n repeticiones calientes con el medidor honesto -> p95 (None si n<20)."""
    func()  # warm-up
    registro: dict[str, list[float]] = {}
    for _ in range(n):
        _, elapsed_ms = medir_tiempo(func, clock=time.perf_counter)
        registro = agregar_medicion(registro, etapa, elapsed_ms)
    return resumen_estadisticas(registro)[etapa]["p95"]


def _medir_asr_p95(whisper: Any, audio: Path, n: int) -> float | None:
    """ASR warm p95 (config del flujo outgoing_es_to_en: faster-whisper es)."""

    def transcribir() -> object:
        return list(whisper.transcribe(str(audio), language="es")[0])

    return _medir_p95(transcribir, n, "asr")


def _medir_traduccion_p95(n: int) -> float | None:
    """Traducción Argos ES→EN warm p95 (frases de entrevista reales).

    La etapa SOLO cuenta como medida si su salida es correcta. El sanity es
    LAXO a propósito (revisión #19): no se compara contra la traducción exacta
    (acoplaría el gate a la versión del modelo); se verifica que contenga la
    frase clave y no sea patológicamente larga/repetitiva — el modo de falla
    real (ARGOS_COMPUTE_TYPE sin "default") produce el bucle "mainstream".
    Además argos cachea el MISMO texto (0.0 ms): se miden n frases DISTINTAS,
    como los turnos reales del flujo.
    """
    from traductor.traduccion.argos import traducir

    salida = _normalizar(traducir(TEXTO_TRADUCCION, "es", "en"))
    palabras = salida.split()
    if (
        "distributed systems" not in salida
        or len(salida) > 80
        or max(palabras.count(p) for p in set(palabras)) > 3
    ):  # repetida patologicamente (el bucle "mainstream" repite la misma palabra)
        print(f"Traducción es→en ROTA (salida {salida[:60]!r}): etapa no medible (FALLA por regla)")
        return None
    if len(FRASES_TRADUCCION) < n:
        raise RuntimeError(f"el corpus de frases ({len(FRASES_TRADUCCION)}) no alcanza para n={n}")
    registro: dict[str, list[float]] = {}
    for frase in FRASES_TRADUCCION[:n]:
        _, elapsed_ms = medir_tiempo(
            partial(traducir, frase, "es", "en"),
            clock=time.perf_counter,
        )
        registro = agregar_medicion(registro, "traduccion", elapsed_ms)
    return resumen_estadisticas(registro)["traduccion"]["p95"]


def _detectar_cable() -> str | None:
    """Nombre del dispositivo VB-CABLE (None si no está instalado)."""
    try:
        import pyaudio
    except ImportError:
        return None
    pa = pyaudio.PyAudio()
    try:
        for i in range(pa.get_device_count()):
            info = pa.get_device_info_by_index(i)
            nombre = str(info["name"] or "")
            if "CABLE" in nombre:
                return nombre
    finally:
        pa.terminate()
    return None


def _dispositivo_cable(pa: Any, lado: str) -> int:
    """Índice del CABLE {lado} con el criterio del flujo real.

    Usa `_buscar_device` (prefiere MME + canales >=): el duplicado WASAPI a
    48000 pasa por el resampler defectuoso que pela el audio (hallazgo del PR
    #33); medir los gates por ese camino mediria un audio distinto al real.
    """
    from traductor.flujo.adaptadores import _buscar_device

    canales = "maxOutputChannels" if lado == "Input" else "maxInputChannels"
    indice = _buscar_device(pa, f"CABLE {lado}", canales, 2)
    if indice is None:
        raise RuntimeError(f"dispositivo CABLE {lado} no encontrado")
    return indice


def _mono24k_a_cable48k(pcm16_mono: bytes) -> bytes:
    """Mono 24 kHz int16 -> estéreo 48 kHz int16 para el CABLE (lo que el flujo hace).

    El 24k->48k es `np.repeat` (zero-order hold, no es un resample real: mete
    aliasing). Para MEDIR latencia da igual; si este código migra al flujo real,
    el flujo debe usar un resampler propio (nota de la revisión #19).
    """
    import numpy as np

    mono = np.frombuffer(pcm16_mono, dtype=np.int16).astype(np.float32) / 32767.0
    pcm48 = np.repeat(mono, 2)  # 24 kHz -> 48 kHz (el cable es nativo a 48 kHz)
    stereo = np.repeat(pcm48, 2)  # mono -> estéreo
    return bytes((np.clip(stereo, -1.0, 1.0) * 32767).astype(np.int16).tobytes())


def _primer_audio_index(datos: bytes) -> int | None:
    """Índice del primer sample audible (>= 5 consecutivos sobre el umbral).

    El CABLE es passthrough digital (ruido ~0): un run corto sobre el umbral
    es señal, no ruido. None si el buffer no contiene señal.
    """
    import numpy as np

    muestras = np.frombuffer(datos, dtype=np.int16)
    for i in range(len(muestras) - 4):
        if abs(muestras[i]) > UMBRAL_AUDIBLE and all(abs(muestras[i + 1 : i + 5]) > UMBRAL_AUDIBLE):
            return i
    return None


def _medir_ruteo_p95(n: int, cable: str | None) -> float | None:
    """Ruteo a VB-CABLE p95: TIME-TO-FIRST-SAMPLE-AUDIBLE.

    El CABLE es un bucle: escribimos a CABLE Input y DETECTAMOS el primer
    sample audible en CABLE Output (el dispositivo real entregándolo) — desde
    que se entrega el chunk hasta que el primer sample es reproducible. NO
    incluye la duración del chunk (corrección 2026-09-09: la versión anterior
    medía el drenado completo, 1003.5 ms para un chunk de 1 s = la duración
    del audio, no latencia).

    Auto-verificación (guard INVERTIDO, atrapa el error conocido):
    p95 >= duración del chunk -> midió el drenado -> raise; p95 <= 0 -> raise.
    Sin cable instalado devuelve None (FALLA por regla).
    """
    if cable is None:
        print("Ruteo: VB-CABLE NO instalado -> gate sin medir (FALLA por regla)")
        return None
    import numpy as np
    import pyaudio

    from traductor.tts.harness import verificar_ruteo_primer_sample

    RATE_CABLE = 48000
    FRAMES_BLOQUE = 9600  # 0.2 s a 48 kHz
    FRAMES_LECTURA = 96  # 2 ms: precisión de detección del primer sample
    pa = pyaudio.PyAudio()
    try:
        salida = pa.open(
            format=pyaudio.paInt16,
            channels=2,
            rate=RATE_CABLE,
            output=True,
            output_device_index=_dispositivo_cable(pa, "Input"),
            frames_per_buffer=FRAMES_BLOQUE,
        )
        entrada = pa.open(
            format=pyaudio.paInt16,
            channels=2,
            rate=RATE_CABLE,
            input=True,
            input_device_index=_dispositivo_cable(pa, "Output"),
            frames_per_buffer=FRAMES_LECTURA,
        )
        try:
            # el probe empieza en amplitud plena (fase pi/2): detectable al instante
            tono_mono = (
                (np.sin(np.arange(SR_FLUJO) * 0.05 + np.pi / 2) * 1000).astype(np.int16).tobytes()
            )
            tono_cable = _mono24k_a_cable48k(tono_mono)  # 1 s de audio
            duracion_ms = SR_FLUJO / SR_FLUJO * 1000.0  # 1000 ms
            bloques = [
                tono_cable[i : i + FRAMES_BLOQUE * 4]
                for i in range(0, len(tono_cable), FRAMES_BLOQUE * 4)
            ]

            def rutear_y_detectar() -> object:
                while entrada.get_read_available() > 0:  # drenar antes de medir
                    entrada.read(entrada.get_read_available(), exception_on_overflow=False)
                t0 = time.perf_counter()
                for bloque in bloques:
                    salida.write(bloque)
                    while entrada.get_read_available() > 0:
                        datos = entrada.read(
                            min(FRAMES_LECTURA, entrada.get_read_available()),
                            exception_on_overflow=False,
                        )
                        indice = _primer_audio_index(datos)
                        if indice is not None:
                            elapsed = (time.perf_counter() - t0) * 1000.0
                            return elapsed - (indice // 2) / RATE_CABLE * 1000.0
                raise RuntimeError(
                    "no se detectó el primer sample en CABLE Output: instrumento roto"
                )

            p95 = _medir_p95(rutear_y_detectar, n, "ruteo")
            verificar_ruteo_primer_sample(p95, duracion_ms)
            print(
                f"Ruteo p95 (primer sample audible): {p95:.1f} ms | duración "
                f"chunk: {duracion_ms:g} ms | límites: 0 < p95 < {duracion_ms:g} "
                "(auto-verificación PASA)"
            )
            return p95
        finally:
            salida.stop_stream()
            salida.close()
            entrada.stop_stream()
            entrada.close()
    finally:
        pa.terminate()


def _latentes_xtts(motor: Any, perfil: VoiceProfile) -> tuple[Any, Any]:
    """Latentes del perfil en el BACKEND, cacheadas (como el flujo real).

    Desde el review del PR #34 el camino declarado es el batch por frase de
    `sintetizar_stream`, que pide `_latentes` la primera vez y las cachea por
    perfil: medirlas aquí las deja fuera del p95 (ADR-014: no entran al
    presupuesto por turno). El resultado se devuelve como discriminador de
    motor con streaming en el resto del harness.
    """
    motor.sintetizar("warm-up de carga del modelo", perfil)  # carga lazy + calienta
    latentes: tuple[Any, Any] = motor._latentes(perfil)
    return latentes


def _primer_chunk_xtts(motor: Any, perfil: VoiceProfile, texto: str) -> Any:
    """Primer chunk del camino declarado: batch por frase (review PR #34).

    `sintetizar_stream` sintetiza la PRIMERA FRASE completa con `inference` y
    las latentes cacheadas (ya calentadas por `_latentes_xtts`); el chunk es
    esa frase, no el texto completo. El generador se CIERRA tras el primer
    chunk para no dejar estado de síntesis vivo en las etapas siguientes.
    """
    import numpy as np

    generador = motor.sintetizar_stream(texto, perfil)
    try:
        resultado = next(generador)
    finally:
        generador.close()
    return np.frombuffer(resultado.datos, dtype=np.float32)


def _chunk_a_pcm16(chunk: Any) -> bytes:
    """Chunk (numpy/torch float32) -> bytes int16 24 kHz para el cable."""
    import numpy as np
    import torch

    if torch.is_tensor(chunk):
        chunk = chunk.detach().cpu().numpy()
    pcm = np.asarray(chunk, dtype=np.float32)
    return bytes((np.clip(pcm, -1.0, 1.0) * 32767).astype(np.int16).tobytes())


def _medir_ttfa_primer_chunk_p95(
    motor: Any,
    n: int,
    perfil: VoiceProfile,
    texto: str,
    latentes: tuple[Any, Any] | None,
) -> float | None:
    """TTFA caliente p95 = PRIMER CHUNK reproducible (n>=20).

    Con latentes (XTTS): primera frase completa de `sintetizar_stream` (batch
    por frase del PR #34). Sin streaming (candidato B): primer chunk = síntesis
    completa, documentado en ADR-014.
    """
    if latentes is None:
        return _medir_p95(partial(motor.sintetizar, texto, perfil), n, "ttfa")
    return _medir_p95(partial(_primer_chunk_xtts, motor, perfil, texto), n, "ttfa")


def _ventanas_pipeline(whisper: Any, muestras: np.ndarray, sr: int) -> list[np.ndarray]:
    """Ventanas = los SEGMENTOS VAD completos de la grabación (inicio Y fin).

    Criterio (corrección del PR #20, revisión del usuario): el conjunto sale
    de la segmentación VAD real del corpus — cada ventana es el segmento
    [s.start, s.end] COMPLETO, sin fijar 3 s ni ignorar s.end (el flujo
    transcribe el segmento entero, no un trozo). NINGUNA ventana se descarta
    por su resultado: si un segmento resultara lento, se reporta (riesgo
    abierto del decodificador, documentado en ADR-014 — no se excluye).
    """
    segmentos, _ = whisper.transcribe(muestras, language="es")
    ventanas: list[np.ndarray] = []
    for s in segmentos:
        a = int(s.start * sr)
        b = min(int(s.end * sr), len(muestras))
        ventanas.append(np.asarray(muestras[a:b]))
    if not ventanas:
        raise RuntimeError(
            "el audio de entrada no produjo segmentos VAD: el pipeline no "
            "tiene entradas reales para medir (instrumento roto)"
        )
    return ventanas


def _medir_pipeline_p95(
    whisper: Any,
    motor: Any,
    n: int,
    audio: Path,
    perfil: VoiceProfile,
    latentes: tuple[Any, Any] | None,
    cable: str | None,
) -> tuple[float | None, dict[str, float], bool]:
    """Pipeline end-to-end p95: audio -> ASR es -> Argos -> 1er chunk -> CABLE.

    UNA corrida encadenada (no la suma de etapas): cada iteración transcribe
    un corte DISTINTO del audio de entrada (los turnos reales difieren, y
    argos cachea texto idéntico), traduce, sintetiza el primer chunk y lo
    rutea al cable con DRENADO (el chunk es reproducible cuando el dispositivo
    lo consume, no cuando el buffer lo acepta).

    Devuelve (p95, incluye_ruteo). Sin VB-CABLE mide solo la cadena hasta el
    primer chunk — informativo; el gate queda sin medir (FALLA por regla).
    """
    import soundfile as sf

    from traductor.traduccion.argos import traducir

    muestras, sr = sf.read(str(audio), dtype="float32")
    ventanas = _ventanas_pipeline(whisper, muestras, sr)
    if len(ventanas) < N_REPETICIONES:
        print(
            f"Pipeline: {len(ventanas)} ventanas VAD (corpus corto) — se ciclan "
            f"para n={N_REPETICIONES}; las traducciones repetidas son cacheadas "
            "por argos (el p95 resultante es optimista en la etapa de traducción)"
        )

    def corte(i: int) -> np.ndarray:
        return ventanas[i % len(ventanas)]

    pa: Any = None
    salida: Any = None
    entrada: Any = None
    RATE_CABLE = 48000
    FRAMES_BLOQUE = 9600
    FRAMES_LECTURA = 96  # 2 ms
    if cable is not None:
        import pyaudio

        pa = pyaudio.PyAudio()
        salida = pa.open(
            format=pyaudio.paInt16,
            channels=2,
            rate=RATE_CABLE,
            output=True,
            output_device_index=_dispositivo_cable(pa, "Input"),
            frames_per_buffer=FRAMES_BLOQUE,
        )
        entrada = pa.open(
            format=pyaudio.paInt16,
            channels=2,
            rate=RATE_CABLE,
            input=True,
            input_device_index=_dispositivo_cable(pa, "Output"),
            frames_per_buffer=FRAMES_LECTURA,
        )
    try:

        def detectar_primer_sample(t0: float) -> float:
            """Espera el primer sample audible en CABLE Output (loopback)."""
            while entrada.get_read_available() > 0:
                datos = entrada.read(
                    min(FRAMES_LECTURA, entrada.get_read_available()),
                    exception_on_overflow=False,
                )
                indice = _primer_audio_index(datos)
                if indice is not None:
                    return (time.perf_counter() - t0) * 1000.0 - (indice // 2) / RATE_CABLE * 1000.0
            return -1.0  # aún no llegó señal

        def una_iteracion(i: int) -> tuple[object, RegistroEtapas]:
            while entrada.get_read_available() > 0:  # drenar antes de medir
                entrada.read(entrada.get_read_available(), exception_on_overflow=False)
            registro = RegistroEtapas()
            registro.marcar("entrada_audio")
            segmentos = list(whisper.transcribe(corte(i), language="es")[0])
            texto_es = " ".join(s.text for s in segmentos)
            registro.marcar("asr")
            texto_en = traducir(texto_es, "es", "en")
            registro.marcar("traduccion")
            if latentes is not None:
                chunk = _primer_chunk_xtts(motor, perfil, texto_en)
            else:
                chunk = motor.sintetizar(texto_en, perfil)
            registro.marcar("tts_primer_chunk")
            if salida is not None:
                pcm = _mono24k_a_cable48k(_chunk_a_pcm16(chunk))
                bloques = [
                    pcm[i : i + FRAMES_BLOQUE * 4] for i in range(0, len(pcm), FRAMES_BLOQUE * 4)
                ]
                for _indice_bloque, bloque in enumerate(bloques):
                    salida.write(bloque)
                    # ENTREGA y PRIMER SAMPLE AUDIBLE son UN solo evento: la
                    # iteración termina cuando el detector de loopback encuentra
                    # el primer sample audible en CABLE Output. No se marca
                    # "audible" como frontera separada: el retorno del detector
                    # es sub-resolución (10 ms) y reportaría 0.1 ms (una
                    # medición que no ocurrió).
                    primer = detectar_primer_sample(registro.inicio_s())
                    if primer > 0:
                        registro.marcar("entrega_y_primer_sample")
                        registro.verificar_cierre(registro.total_ms())
                        return chunk, registro
                # el chunk terminó sin detectar (silencio inicial largo)
                while True:
                    primer = detectar_primer_sample(registro.inicio_s())
                    if primer > 0:
                        registro.marcar("entrega_y_primer_sample")
                        registro.verificar_cierre(registro.total_ms())
                        return chunk, registro
                    if time.perf_counter() - registro.inicio_s() > 5.0:
                        raise RuntimeError(
                            "no se detectó el primer sample en CABLE Output: instrumento roto"
                        )
                    time.sleep(0.01)
            registro.verificar_cierre(registro.total_ms())
            return chunk, registro

        una_iteracion(0)  # warm-up
        registro: dict[str, list[float]] = {}
        desglose_por_etapa: dict[str, list[float]] = {}
        for i in range(n):
            (_, etapas), _elapsed_ms = medir_tiempo(
                partial(una_iteracion, i), clock=time.perf_counter
            )
            registro = agregar_medicion(registro, "pipeline", _elapsed_ms)
            for etapa, ms in etapas.desglose_ms().items():
                desglose_por_etapa.setdefault(etapa, []).append(ms)
        p95 = resumen_estadisticas(registro)["pipeline"]["p95"]
        desglose_p95 = {
            etapa: p95_etapa
            for etapa, valores in desglose_por_etapa.items()
            if (p95_etapa := resumen_estadisticas({etapa: valores})[etapa]["p95"]) is not None
        }
        if salida is not None and "entrega_y_primer_sample" in desglose_p95:
            from traductor.tts.harness import verificar_resolucion_audible

            verificar_resolucion_audible(
                desglose_p95["entrega_y_primer_sample"], FRAMES_LECTURA / RATE_CABLE * 1000.0
            )
        return p95, desglose_p95, salida is not None
    finally:
        if salida is not None:
            salida.stop_stream()
            salida.close()
        if entrada is not None:
            entrada.stop_stream()
            entrada.close()
        if pa is not None:
            pa.terminate()


def main(argv: list[str] | None = None) -> None:
    if hasattr(sys.stdout, "reconfigure"):  # prints con → en consolas cp1252
        sys.stdout.reconfigure(encoding="utf-8")
    import torch

    args = parser_harness().parse_args(argv)
    perfil = perfil_benchmark(args)

    motor = _cargar_motor(args.motor)
    vram_base = vram_ocupada_mib(torch.cuda)  # motor residente, Whisper aún no
    whisper = _cargar_whisper()
    # Orden deliberado: Whisper ya residente ANTES de medir TTFA — las síntesis
    # corren bajo presión de VRAM real (co-residencia, ADR-014). Si se invierte
    # el orden, el TTFA baja "gratis" y el gate miente.
    vram_tras_whisper = _calentar_whisper_y_foto(whisper, args.warmup_audio)
    if vram_base is not None and vram_tras_whisper is not None:
        delta = vram_tras_whisper - vram_base
        print(f"delta VRAM (Whisper, foto tras warm-up): {delta:.1f} MiB")
        if delta < DELTA_WHISPER_MIN_MIB:
            raise RuntimeError(
                f"Whisper no reservó VRAM (delta {delta:.1f} MiB < "
                f"{DELTA_WHISPER_MIN_MIB:g}): el warm-up no ejercitó CT2; "
                "instrumento no fiable."
            )
    latentes: tuple[Any, Any] | None = None
    if args.motor == "xtts":
        latentes = _latentes_xtts(motor, perfil)  # carga el modelo (lazy)
    asr = _medir_asr_p95(whisper, args.warmup_audio, N_REPETICIONES)
    traduccion = _medir_traduccion_p95(N_REPETICIONES)
    cable = _detectar_cable()
    ruteo = _medir_ruteo_p95(N_REPETICIONES, cable)
    ttfa = _medir_ttfa_primer_chunk_p95(motor, N_REPETICIONES, perfil, args.texto, latentes)
    pipeline: float | None = None
    desglose_pipeline: dict[str, float] = {}
    con_ruteo = False
    if traduccion is None or cable is None:
        print(
            "Pipeline end-to-end: sin medir (VB-CABLE ausente o traducción "
            "rota — no se mide la cadena sin micrófono virtual ni con basura)"
        )
    else:
        pipeline, desglose_pipeline, con_ruteo = _medir_pipeline_p95(
            whisper, motor, N_REPETICIONES, args.warmup_audio, perfil, latentes, cable
        )
    pipeline_gate = pipeline if con_ruteo else None
    # Foto de co-residencia DESPUÉS de las síntesis: Whisper + motor con sus
    # reservas reales. Es el número del gate.
    vram = vram_ocupada_mib(torch.cuda)
    ram = medir_ram_mib()
    del whisper, motor  # liberar DESPUÉS de la foto, no antes
    medicion = componer_medicion(
        ttfa=ttfa,
        asr=asr,
        traduccion=traduccion,
        ruteo=ruteo,
        pipeline=pipeline_gate,
        vram=vram,
        ram=ram,
        args=args,
    )
    resultados = evaluar_gates(medicion)
    print(f"ASR warm p95 (faster-whisper es): {asr:.1f} ms" if asr else "ASR: sin medir")
    print(f"Traducción Argos p95: {traduccion:.1f} ms" if traduccion else "Traducción: sin medir")
    print(f"Ruteo VB-CABLE p95: {ruteo:.1f} ms" if ruteo else "Ruteo: sin medir")
    if ttfa is None:
        print(f"TTFA 1er chunk p95: sin medir (n<{N_REPETICIONES})")
    else:
        print(f"TTFA 1er chunk p95: {ttfa:.1f} ms (streaming caliente)")
    if pipeline is None:
        print("Pipeline end-to-end p95: sin medir")
    elif con_ruteo:
        print(f"Pipeline end-to-end p95 (con ruteo al cable): {pipeline:.1f} ms")
        print("  Desglose por frontera (cierre exacto por iteración, guard verificado):")
        for etapa, ms in desglose_pipeline.items():
            print(f"    {etapa:24s} p95: {ms:.1f} ms")
        suma = sum(desglose_pipeline.values())
        print(
            f"    suma de p95 por etapa: {suma:.1f} ms | total p95: {pipeline:.1f} ms "
            "(diferencia = p95 de iteraciones distintas, no residuo: por iteración "
            "la suma cierra exacto)"
        )
    else:
        print(
            f"Pipeline p95 SIN ruteo (VB-CABLE ausente, informativo): {pipeline:.1f} ms — "
            "el gate queda sin medir -> FALLA por regla"
        )
    if vram is None:
        print("VRAM co-residente: sin medir (sin CUDA)")
    else:
        print(f"VRAM co-residente: {vram:.1f} MiB (compara con nvidia-smi ±100 MiB)")
    if ram is None:
        print("RAM total: sin medir (psutil no instalado)")
    else:
        print(f"RAM total: {ram:.1f} MiB")
    print()
    print(resumen_gates(resultados))
    print()
    print("cabe_en_gates:", cabe_en_gates(medicion))


if __name__ == "__main__":
    main()
