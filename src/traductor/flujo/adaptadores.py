"""Adaptadores del flujo outgoing_es_to_en — hardware/máquina (ADR-015).

El core (`outgoing.py`) es puro y testeado; aquí viven las piezas que tocan
hardware: micrófono (RealtimeSTT), worker TTS (proceso aparte, ADR-013),
salida a VB-CABLE y teleprompter HTTP. Son delgadas y documentadas; las
líneas que tocan hardware van `# pragma: no cover` (se ejercitan en la
máquina objetivo, como el harness del ADR-014).
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path
from typing import Any, Protocol

from traductor.tts.backend_xtts import SR_XTTS


class FlujoASR(Protocol):
    """Contrato mínimo que un flujo le expone al ASR en tiempo real.

    `segmento_final` devuelve el nivel de la escalera (int en outgoing) o un
    bool (incoming): el retorno `object` cubre ambas. Parámetros positional-only
    (`/`): los flujos nombran el texto distinto (`texto_es` / `texto_en`) y el
    nombre no debe importar en el protocol (Hal r1 del PR #25: `Any` apagaba
    mypy --strict sobre estos métodos).
    """

    def parcial(self, texto: str, /) -> None: ...
    def cancelar_turno_activo(self) -> None: ...
    def segmento_final(self, texto: str, /) -> object: ...


def _pcm_f32_a_wav(datos: bytes, sample_rate: int) -> tuple[bytes, float]:
    """PCM float32 mono → WAV int16 mono (el flujo trabaja con WAV)."""
    import io
    import wave

    import numpy as np

    pcm = np.frombuffer(datos, dtype=np.float32)
    int16 = (np.clip(pcm, -1.0, 1.0) * 32767).astype(np.int16)
    buf = io.BytesIO()
    with wave.open(buf, "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(sample_rate)
        w.writeframes(int16.tobytes())
    return buf.getvalue(), float(len(int16)) / sample_rate


class AsrRetorno:
    """ASR-de-retorno para validar el audio sintetizado (ADR-019, fix 3).

    Transcribe un WAV en inglés (el idioma de salida del flujo) con el mismo
    modelo tiny del micrófono. El WAV se pasa como `BytesIO` (revisión #23):
    faster-whisper con ndarray NO resamplea (asume 16 k) y oiría el audio a
    2/3 de velocidad.
    """

    def __init__(self) -> None:  # pragma: no cover - requiere modelo
        from faster_whisper import WhisperModel

        self._whisper = WhisperModel("tiny", device="cuda", compute_type="int8_float16")

    def transcribir(self, audio: bytes) -> str | None:  # pragma: no cover
        """Texto en inglés del WAV; None si el modelo no está disponible."""
        import io

        try:
            segmentos, _ = self._whisper.transcribe(io.BytesIO(audio), language="en")
            return " ".join(s.text for s in segmentos)
        except Exception:
            return None


class AsrRealtime:  # pragma: no cover - requiere micrófono + RealtimeSTT
    """Entrada de audio → segmentos: los PARCIALES cancelan y van a pantalla;
    los FINALES se procesan en un hilo worker por turno.

    Sirve para AMBAS direcciones del ADR-015:
    - outgoing (hablar): el micrófono físico, `idioma="es"` (default).
    - incoming (escuchar): el audio REMOTO del entrevistador llega por el
      cable virtual (CABLE Input como salida de Meet/Zoom), capturado con
      `input_device_index` del CABLE Output, `idioma="en"`.

    `sample_rate` es la tasa de CAPTURA del device: el cable VB-CABLE solo
    acepta 48000 (rechaza 16000/24000/44100 con InvalidSampleRate), mientras
    que RealtimeSTT valida el device abriéndolo a SU `sample_rate` — si no se
    alinea, "Selected device validation failed" (bug cazado en la demo del
    PR #25: el micrófono físico acepta 16000, el cable no).

    RealtimeSTT ya integra VAD/endpointing y entrega sus callbacks desde SUS
    propios hilos (revisión #22: el invariante de hilo único era falso).
    - `_parcial`: el usuario/entrevistador volvió a hablar → se CANCELA el
    turno anterior en vuelo (`cancelar_turno_activo`) y el parcial va a
    pantalla. Sin esto, la cancelación del ADR-015 era código muerto.
    - `_final`: el turno corre en un hilo daemon para que la entrada siga
    fluyendo y un nuevo segmento pueda cancelarlo (cola de tamaño 1).
    """

    def __init__(
        self,
        flujo: FlujoASR,
        *,
        idioma: str = "es",
        input_device_index: int | None = None,
        sample_rate: int = 16000,
        etiqueta: str = "",
        post_speech_silence_duration: float = 1.5,
    ) -> None:
        self._flujo = flujo
        self._idioma = idioma
        self._input_device_index = input_device_index
        self._sample_rate = sample_rate
        self._etiqueta = etiqueta or f"Flujo {idioma}"
        # Silencio que CIERRA un turno. El default de RealtimeSTT (~0.6 s)
        # parte un párrafo hablado en ~5 turnos por las pausas naturales entre
        # frases; cada turno nuevo CANCELA el TTS del anterior a media frase
        # (bug reportado: la voz EN sale "de a 4 palabras, corte, 4 palabras").
        # A 1.5 s el párrafo entero es UN solo turno: el TTS lo sintetiza
        # completo sin autocancelarse. Ajustable por env sin tocar código.
        self._post_speech_silence_duration = post_speech_silence_duration

    def _parcial(self, texto: str) -> None:
        self._flujo.cancelar_turno_activo()
        self._flujo.parcial(texto)

    def _final(self, texto: str) -> None:
        import threading

        def correr() -> None:
            self._flujo.segmento_final(texto)
            # desglose del cierre por etapa (diagnóstico de latencia en vivo:
            # el core guarda `ultimo_turno_etapas` en ambos flujos)
            etapas = getattr(self._flujo, "ultimo_turno_etapas", {})
            total = getattr(self._flujo, "ultimo_turno_total_ms", 0.0)
            if etapas:
                desglose = " | ".join(f"{k} {v:.0f} ms" for k, v in etapas.items())
                print(f"[turno] cierre {total:.0f} ms: {desglose}", flush=True)

        threading.Thread(target=correr, daemon=True).start()

    def correr(self) -> None:
        from RealtimeSTT import AudioToTextRecorder

        grabador = AudioToTextRecorder(
            model="tiny",
            language=self._idioma,
            device="cuda",
            compute_type="int8",
            input_device_index=self._input_device_index,
            sample_rate=self._sample_rate,
            on_realtime_transcription_update=self._parcial,
            post_speech_silence_duration=self._post_speech_silence_duration,
        )
        print(f"{self._etiqueta} listo: ctrl+C para salir.")
        while True:
            grabador.text(self._final)


def procesar_chunk_vad(
    estado: dict[str, Any],
    arr: Any,
    rms: float,
    *,
    chunk_s: float,
    umbral_actividad: float,
    silencio_cierre_s: float,
    fragmento_max_s: float,
    cola: Any,
) -> None:
    """Máquina de estados del VAD por energía (pura y testeable).

    `estado` lleva `fragmento` (chunks acumulados), `habla` (bool),
    `silencio_desde` (segundos de silencio tras el último habla) y
    `contador_turnos`. Al CERRAR un turno, le asigna el contador FIFO (0, 1,
    2...) y encola `(numero_turno, tuple(fragmento))` — el orden de pantalla
    es el orden de la ENTRADA, no el de la transcripción (revisión del PR
    #26: antes cada cierre lanzaba su propio hilo y un turno corto podía
    adelantarse a uno largo anterior).

    `arr` es OPACO para la máquina (no toca numpy): la concatenación real la
    hace el worker al transcribir. Así se testea en CI sin numpy.
    """
    if "habla" not in estado:
        estado["fragmento"] = []
        estado["habla"] = False
        estado["silencio_desde"] = 0.0
    fragmento = estado["fragmento"]
    habla = estado["habla"]
    silencio_desde = estado["silencio_desde"]
    if rms > umbral_actividad:
        estado["habla"] = True
        estado["silencio_desde"] = 0.0
        fragmento.append(arr)
        # corte por LONGITUD también con voz continua: sin silencio, un turno
        # infinito nunca cerraría (la voz real no es continua 12 s seguidos,
        # pero el límite debe valer siempre)
        if len(fragmento) * chunk_s > fragmento_max_s:
            _cerrar_turno(estado, fragmento, cola)
        return
    if not habla:
        return
    silencio_desde += chunk_s
    estado["silencio_desde"] = silencio_desde
    fragmento.append(arr)
    cierre = silencio_desde > silencio_cierre_s
    cierre = cierre or len(fragmento) * chunk_s > fragmento_max_s
    if cierre:
        _cerrar_turno(estado, fragmento, cola)


def _cerrar_turno(estado: dict[str, Any], fragmento: list[Any], cola: Any) -> None:
    """Cierra el turno: asigna el contador FIFO y encola para el worker.

    `contador_turnos` persiste entre turnos (no se resetea aquí): cada cierre
    toma el siguiente número en orden de entrada.
    """
    numero_turno = estado.get("contador_turnos", 0)
    estado["contador_turnos"] = numero_turno + 1
    cola.put((numero_turno, tuple(fragmento)))
    estado["fragmento"] = []
    estado["habla"] = False
    estado["silencio_desde"] = 0.0


class AsrCable:  # pragma: no cover - requiere VB-CABLE + modelo
    """Captura el audio REMOTO del cable y lo transcribe (incoming EN→ES).

    Patrón del ENDURANCE (ADR-019): pyaudio lee del cable a 48000 en chunks,
    un VAD por ENERGÍA separa los turnos (el silencio del cable mide RMS < 10;
    la voz ~2000+), y faster-whisper transcribe el fragmento completo al
    cerrar el turno. El flujo incoming traduce EN→ES y lo muestra.

    UN WORKER + COLA (revisión del PR #26): cada cierre de turno encola el
    bloque en `queue.Queue` con su NÚMERO DE TURNO asignado en orden FIFO, y
    un ÚNICO worker desencola y transcribe en secuencia. Antes cada cierre
    lanzaba su propio hilo → un turno corto podía terminar de transcribir
    antes que uno largo anterior (orden invertido en pantalla y cancelación
    del ADR-015 marcando como superado al enunciado más viejo) y
    `WhisperModel.transcribe()` no es seguro bajo concurrencia.

    POR QUÉ NO RealtimeSTT para esta entrada (medido en la demo del PR #25):
    - el cable SOLO acepta 48000 y RealtimeSTT valida el device a 16000
      ("Selected device validation failed");
    - su modo manual (`use_microphone=False` + `feed_audio`) NO cierra los
      turnos con esta entrada: el VAD de Silero detecta inicio pero el FIN
      nunca se confirma (8 turnos del reproductor → 0 transcripciones en el
      proceso aparte; el cable transfiere bien el audio — RMS verificado).
    """

    def __init__(
        self,
        flujo: FlujoASR,
        *,
        indice_cable: int,
        rate_cable: int = 48000,
        chunk_s: float = 0.5,
        umbral_actividad: float = 300.0,
        silencio_cierre_s: float = 1.0,
        fragmento_max_s: float = 4.0,
    ) -> None:
        self._flujo = flujo
        self._indice_cable = indice_cable
        self._rate_cable = rate_cable
        self._chunk_s = chunk_s
        self._umbral_actividad = umbral_actividad
        self._silencio_cierre_s = silencio_cierre_s
        self._fragmento_max_s = fragmento_max_s
        self._whisper: Any | None = None

    def _cargar_whisper(self) -> Any:
        if self._whisper is None:
            from faster_whisper import WhisperModel

            # small, no tiny: tiny ALUCINA sobre el audio del cable (frases del
            # corpus de subtítulos: "Thanks for watching!") incluso con
            # fragmentos cortos. small ya fue validado en esta GPU en el
            # ADR-012 (int8_float16, proceso aparte del worker XTTS).
            self._whisper = WhisperModel("small", device="cuda", compute_type="int8_float16")
        return self._whisper

    def correr(self) -> None:
        import queue
        import threading

        import numpy as np
        import pyaudio

        whisper = self._cargar_whisper()
        pa = pyaudio.PyAudio()
        # Tasa NATIVA del device (MME = 44100): el resampler WASAPI a 48000
        # de los drivers VB-Audio inserta saltos de fase cada ~20 ms (audio
        # con clics que whisper transcribe distorsionado; ver _buscar_device).
        rate = _tasa_nativa(pa, self._indice_cable, self._rate_cable)
        # El cable es ESTÉREO: leerlo con channels=1 entrega datos que
        # faster-whisper NO transcribe (texto vacío — bug cazado en la demo:
        # pyaudio en un device estéreo con channels=1 devuelve muestras
        # inválidas). Se lee ESTÉREO y se toma el canal izquierdo.
        stream = pa.open(
            format=pyaudio.paInt16,
            channels=2,
            rate=rate,
            input=True,
            input_device_index=self._indice_cable,
            frames_per_buffer=1024,
        )
        print("Flujo incoming (EN->ES, subtitulos) listo: ctrl+C para salir.", flush=True)

        def worker(cola: Any) -> None:
            """ÚNICO transcriptor: desencola en orden y muestra en secuencia."""
            import io

            import soundfile as sf

            while True:
                numero_turno, fragmento = cola.get()
                buf = io.BytesIO()
                sf.write(
                    file=buf,
                    data=np.concatenate(fragmento).astype(np.float32),
                    samplerate=rate,
                    format="WAV",
                )
                buf.seek(0)
                try:
                    # condition_on_previous_text=False: cada fragmento se
                    # transcribe DESDE CERO (no arrastra el texto anterior como
                    # prompt) — corta el bucle de alucinación donde whisper
                    # repite una frase fantasma turno tras turno. vad_filter:
                    # Silero recorta el silencio ANTES de transcribir, así el
                    # modelo no "rellena" los tramos mudos con frases del corpus
                    # (créditos de subtítulos, "thank you"). Ambos matan la
                    # alucinación en el ORIGEN, con cualquier audio (no solo el
                    # WAV de la demo). Ver traductor.asr.alucinacion (2ª capa).
                    segmentos, _ = whisper.transcribe(
                        buf,
                        language="en",
                        condition_on_previous_text=False,
                        vad_filter=True,
                    )
                    texto_en = " ".join(s.text for s in segmentos).strip()
                except Exception as exc:  # noqa: BLE001 - el flujo no enmudece sin log
                    print(
                        f"[incoming] turno {numero_turno}: whisper falló: "
                        f"{type(exc).__name__}: {exc}",
                        flush=True,
                    )
                    continue
                if not texto_en:
                    continue
                try:
                    self._flujo.segmento_final(texto_en)
                except Exception as exc:  # noqa: BLE001 - el hilo no muere en silencio
                    print(
                        f"[incoming] turno {numero_turno}: segmento_final falló: "
                        f"{type(exc).__name__}: {exc}",
                        flush=True,
                    )

        cola: Any = queue.Queue()
        threading.Thread(target=worker, args=(cola,), daemon=True).start()
        estado: dict[str, Any] = {}
        while True:
            datos = stream.read(int(rate * self._chunk_s), exception_on_overflow=False)
            # estéreo -> MONO con downmix (L+R)/2: tomar solo el canal
            # izquierdo descartaba en silencio las fuentes que llegan por el
            # derecho (revisión del PR #33); un downmix cuesta lo mismo.
            arr = np.frombuffer(datos, dtype=np.int16).astype(np.float32)
            mono = (arr[0::2] + arr[1::2]) / 2.0
            rms = float(np.sqrt(np.mean(mono**2)))
            procesar_chunk_vad(
                estado,
                mono,
                rms,
                chunk_s=self._chunk_s,
                umbral_actividad=self._umbral_actividad,
                silencio_cierre_s=self._silencio_cierre_s,
                fragmento_max_s=self._fragmento_max_s,
                cola=cola,
            )


class TtsWorkerCliente:  # pragma: no cover - requiere venv-tts + modelo
    """Cliente del worker TTS (ADR-013): proceso aparte, jobs JSON-line.

    Lanza `worker_tts_boot.py` con el python del venv del TTS (que es donde
    vive coqui-tts, ADR-011), envía un job y lee el WAV de salida.

    JOB STREAMING (ADR-019): `sintetizar_stream` envía un job con
    `streaming: true` y rinde un WAV POR CHUNK — el PRIMERO llega en ~0.7 s y
    el flujo cierra el turno con él; los chunks siguientes llegan mientras el
    worker sigue sintetizando (el cierre ya no espera la síntesis completa).
    """

    def __init__(self, *, python: Path, directorio_salida: Path, perfil_id: str) -> None:
        self._python = str(python)
        self._directorio_salida = directorio_salida
        self._perfil_id = perfil_id
        self._proceso: Any | None = None
        self._lock_lectura: Any = None  # un solo lector del pipe a la vez

    def iniciar(self) -> None:
        import threading

        self._lock_lectura = threading.Lock()
        boot = Path(__file__).resolve().parents[3] / "scripts" / "worker_tts_boot.py"
        # el python viene del venv de la máquina, el script es del repo y el
        # directorio es un tempdir propio: ninguna entrada de red ni no
        # confiable llega a subprocess (falso positivo de S603).
        self._proceso = subprocess.Popen(  # noqa: S603 - ver comentario
            [self._python, str(boot), str(self._directorio_salida)],
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL,  # el stderr no es protocolo: sin drenar
            # llenaría el buffer y trabaría al worker (torch imprime warnings)
            text=True,
        )
        # CALENTAMIENTO AL ARRANCAR (ADR-013): la carga fría del modelo tarda
        # ~45 s — sin este job, el primer turno siempre cae por el timeout de
        # `readline`. El warm-up espera hasta 120 s y exige un resultado ok.
        self._calentar()

    def _calentar(self) -> None:
        if self._proceso is None or self._proceso.stdin is None:
            raise RuntimeError("worker TTS sin proceso para calentar")
        job = {"texto": "warm up", "perfil_id": self._perfil_id, "salida": "warmup.wav"}
        try:
            self._proceso.stdin.write(json.dumps(job) + "\n")
            self._proceso.stdin.flush()
        except OSError as exc:  # el worker cayó antes de calentar (pipe muerto)
            self.cerrar()
            raise RuntimeError(f"el worker TTS no calentó: {exc}") from exc
        linea = self._leer_resultado(timeout_s=120.0)
        if linea is None or not linea.strip():
            self.cerrar()
            raise RuntimeError("el worker TTS no respondió al calentamiento (120 s)")
        resultado = json.loads(linea)
        if not resultado["ok"]:
            self.cerrar()
            raise RuntimeError(f"el worker TTS no calentó: {resultado.get('error', 'desconocido')}")

    def sintetizar(self, texto_en: str) -> tuple[bytes, float, str] | None:
        """Job → WAV. None si el worker falló o se trabó (escalera del flujo).

        El `readline` tiene TIMEOUT (revisión #22): si el worker se traba, el
        audio de la sesión no cuelga esperando — el worker se termina y el
        siguiente turno lo reinicia (watchdog sin reiniciar la llamada).
        """
        if self._proceso is None:
            try:
                self.iniciar()  # reinicio tras un trabón (ADR-015)
            except RuntimeError:
                return None  # el reinicio falló: escalera (revisión #23)
        if self._proceso is None or self._proceso.stdin is None:
            return None
        nombre = "turno.wav"
        job = {"texto": texto_en, "perfil_id": self._perfil_id, "salida": nombre}
        try:
            self._proceso.stdin.write(json.dumps(job) + "\n")
            self._proceso.stdin.flush()
        except OSError:
            self.cerrar()  # pipe muerto (el worker cayó): escalera, se reinicia
            return None
        linea = self._leer_resultado()
        if linea is None:
            self.cerrar()  # el worker se trabó: escalera; el próximo turno reinicia
            return None
        if not linea.strip():
            self.cerrar()
            return None
        resultado = json.loads(linea)
        if not resultado["ok"]:
            return None
        return self._leer_wav(resultado)

    def sintetizar_stream(self, texto_en: str) -> Any | None:
        """Job streaming → generador de WAVs por chunk (ADR-019, fix 1).

        El PRIMER `next()` devuelve el primer chunk (~0.7 s de síntesis + el
        wrap a WAV); los siguientes llegan mientras el worker sigue
        sintetizando; `StopIteration` al recibir el `fin`. None si el worker
        falló o se trabó (escalera del flujo).
        """
        if self._proceso is None:
            try:
                self.iniciar()  # reinicio tras un trabón (ADR-015)
            except RuntimeError:
                return None  # el reinicio falló: escalera (revisión #23)
        if self._proceso is None or self._proceso.stdin is None:
            return None
        job = {
            "texto": texto_en,
            "perfil_id": self._perfil_id,
            "salida": "turno",
            "streaming": True,
        }
        try:
            self._proceso.stdin.write(json.dumps(job) + "\n")
            self._proceso.stdin.flush()
        except OSError:
            self.cerrar()  # pipe muerto (el worker cayó): escalera, se reinicia
            return None

        def _chunks() -> Any:
            while True:
                try:
                    linea = self._leer_resultado()
                except ValueError:
                    return  # pipe cerrado (worker terminado): fin del stream
                if linea is None:
                    self.cerrar()  # el worker se trabó a mitad del stream
                    return
                if not linea.strip():
                    self.cerrar()
                    return
                resultado = json.loads(linea)
                if not resultado["ok"]:
                    return
                if resultado["tipo"] == "fin":
                    return
                wav = self._leer_wav(resultado)
                if wav is None:
                    return  # formato desconocido: escalera
                yield wav

        return _chunks()

    def _leer_wav(self, resultado: dict[str, Any]) -> tuple[bytes, float, str] | None:
        """WAV del resultado del worker (pcm_f32le → WAV, ADR-011/013).

        None si el formato no se conoce (escalera del flujo).
        """
        ruta = self._directorio_salida / resultado["salida"]
        datos = ruta.read_bytes()
        formato = resultado["formato"]
        if formato == "pcm_f32le":
            # el worker escribe los bytes del AudioResult tal cual (ADR-011:
            # pcm_f32le para XTTS); el flujo trabaja con WAV: se envuelve
            audio_wav, duracion_s = _pcm_f32_a_wav(datos, SR_XTTS)
            return audio_wav, duracion_s, "xtts-kevin"
        if formato == "wav":
            import io
            import wave

            with wave.open(io.BytesIO(datos), "rb") as w:
                duracion_s = w.getnframes() / w.getframerate()
            return datos, duracion_s, "xtts-kevin"
        return None  # formato desconocido: escalera

    def _leer_resultado(self, timeout_s: float = 30.0) -> str | None:
        """readline con timeout (None si el worker no respondió).

        El pipe es UN recurso compartido: el hilo daemon del streaming y el
        siguiente turno lo leen — el lock serializa (un lector a la vez, sin
        robarse líneas del otro).
        """
        import threading

        if self._proceso is None or self._proceso.stdout is None:
            return None
        lock = self._lock_lectura
        if lock is None:
            return None  # sin worker arrancado: nada que leer
        with lock:
            linea: list[str] = []
            stdout = self._proceso.stdout

            def leer() -> None:
                linea.append(stdout.readline())

            hilo = threading.Thread(target=leer, daemon=True)
            hilo.start()
            hilo.join(timeout=timeout_s)
            if hilo.is_alive():
                return None
            return linea[0] if linea else None

    @property
    def pid(self) -> int | None:
        """PID del worker (para vigilar SU memoria, no la del harness)."""
        return self._proceso.pid if self._proceso is not None else None

    def cerrar(self) -> None:
        if self._proceso is not None:
            self._proceso.terminate()
            self._proceso = None


class SalidaCable:  # pragma: no cover - requiere VB-CABLE
    """Escribe el WAV del TTS a CABLE Input (el audio sintetizado NO vuelve
    al micrófono físico: la ruta de salida es explícitamente el cable).

    UN WRITER + COLA FIFO (fix del audio entrecortado): con streaming llegan
    varios chunks del TTS seguidos y cada `reproducir` parte su chunk en
    bloques. Antes cada llamada lanzaba SU PROPIO hilo daemon para el resto de
    sus bloques → con varios chunks había varios daemons compitiendo por el
    stream, y el lock serializaba cada escritura pero NO el ORDEN: los bloques
    de distintas partes del enunciado se intercalaban (síntoma: "una frase
    bien, salta a otra, luego una palabra suelta"). Ahora `reproducir` solo
    ENCOLA los bloques en orden y un ÚNICO hilo escritor los saca de la cola
    (FIFO) y los escribe en secuencia: el orden del audio es el orden del
    stream, sin intercalado, sin importar cuántos chunks lleguen.

    ADR-019, fix 2: `reproducir` no bloquea esperando la reproducción — encola
    y devuelve. El cierre del turno (primer sample audible) es cuando el writer
    saca el primer bloque, que ocurre casi de inmediato (está bloqueado en
    `cola.get()` esperando). La duración del audio NO es latencia.

    `abrir()`/`cerrar()` mantienen el stream y el writer vivos entre turnos
    (abrir por turno paga ~1-3 s de overhead en Windows).
    """

    def __init__(self, rate_cable: int = 48000, bloque_s: float = 0.2) -> None:
        self._rate_cable = rate_cable
        self._bloque_s = bloque_s
        self._pa: Any | None = None
        self._stream: Any | None = None
        self._rate_stream: int | None = None  # tasa REAL del stream abierto
        self._cola: Any = None  # queue.Queue de bloques; None = sentinela de cierre
        self._writer: Any = None  # único hilo escritor (preserva el orden FIFO)

    def _rate_efectiva(self) -> int:
        """Tasa del stream abierto o la de respaldo (tests con stream fake)."""
        return self._rate_stream if self._rate_stream is not None else self._rate_cable

    def abrir(self) -> None:
        import queue
        import threading

        import pyaudio

        self._pa = pyaudio.PyAudio()
        # Nombre del device de SALIDA configurable por env: por defecto el
        # VB-CABLE ("CABLE Input"); con VoiceMeeter el TTS va a su VAIO
        # (mic de Meet/OBS) y el VB-CABLE queda SOLO para el entrevistador.
        nombre = os.environ.get("TRADUCTOR_DEVICE_OUTGOING", "CABLE Input")
        indice = _buscar_device(self._pa, nombre, "maxOutputChannels", 2)
        if indice is None:
            raise RuntimeError(
                f"El device de salida '{nombre}' no está disponible: instala "
                "VB-CABLE (o configura TRADUCTOR_DEVICE_OUTGOING a un device "
                "de salida existente, p. ej. 'VoiceMeeter Input') antes de "
                "abrir la salida"
            )
        # Tasa NATIVA del device (MME = 44100): el resampler WASAPI a 48000
        # de los drivers VB-Audio inserta saltos de fase (audio con clics
        # que destruyen la transcripción; ver _buscar_device).
        self._rate_stream = _tasa_nativa(self._pa, indice, self._rate_cable)
        self._stream = self._pa.open(
            format=pyaudio.paInt16,
            channels=2,
            rate=self._rate_stream,
            output=True,
            output_device_index=indice,
        )
        self._cola = queue.Queue()
        self._writer = threading.Thread(target=self._consumir, daemon=True)
        self._writer.start()

    def _consumir(self) -> None:
        """Único escritor: saca bloques de la cola en orden y los escribe.

        Un solo consumidor garantiza que el orden de escritura al cable es el
        orden de encolado (FIFO), sin importar cuántos hilos productores haya.
        `None` es el sentinela de cierre.
        """
        cola = self._cola
        if cola is None:
            return
        while True:
            bloque = cola.get()
            if bloque is None:  # sentinela: cerrar el writer
                return
            stream = self._stream
            if stream is not None:
                stream.write(bloque)

    def cerrar(self) -> None:
        cola = self._cola
        writer = self._writer
        if cola is not None:
            cola.put(None)  # sentinela: el writer termina tras vaciar la cola
        if writer is not None:
            writer.join(timeout=2.0)
        self._writer = None
        self._cola = None
        if self._stream is not None:
            self._stream.stop_stream()
            self._stream.close()
            self._stream = None
        if self._pa is not None:
            self._pa.terminate()
            self._pa = None

    def reproducir(self, audio: bytes, duracion_s: float, nombre: str) -> None:
        """Encola los bloques del audio en orden y devuelve (no bloquea).

        El writer único los escribe al cable en secuencia FIFO. Sin stream/cola
        (no abierto): no se reproduce (escalera del cable).
        """
        stream = self._stream
        cola = self._cola
        if stream is None or cola is None:
            return  # sin stream: no reproducir (escalera del cable)
        pcm = self._audio_a_pcm_cable(audio)
        # bloque en BYTES: rate * 2 canales * 2 bytes (int16) * segundos
        bloque = max(1, int(self._rate_efectiva() * 4 * self._bloque_s))
        for i in range(0, len(pcm), bloque):
            cola.put(pcm[i : i + bloque])

    def _audio_a_pcm_cable(self, audio: bytes) -> bytes:
        """WAV → PCM int16 estéreo a la tasa del stream (resample lineal)."""
        import io

        import numpy as np
        import soundfile as sf

        datos, sr = sf.read(io.BytesIO(audio), dtype="float32")
        mono = np.asarray(datos, dtype=np.float32)
        # resample LINEAL a la tasa del stream (revisión #22: la división
        # entera distorsiona el tono con sr que no divide la tasa destino,
        # p. ej. 22050 del candidato B)
        rate_destino = self._rate_efectiva()
        ratio = rate_destino / sr
        n_salida = int(len(mono) * ratio)
        pos = np.arange(n_salida, dtype=np.float32) / ratio
        i0 = pos.astype(np.int64)
        i1 = np.minimum(i0 + 1, len(mono) - 1)
        alpha = (pos - i0).astype(np.float32)
        res = mono[i0] * (1 - alpha) + mono[i1] * alpha
        stereo = np.repeat(res, 2)
        return bytes((np.clip(stereo, -1.0, 1.0) * 32767).astype(np.int16).tobytes())


class TeleprompterHttp:  # pragma: no cover - requiere el UI corriendo
    """Manda el texto al teleprompter local (FastAPI, POST /api/transcripcion).

    `fuente` etiqueta quién habla ("yo" = mi voz→EN, outgoing; "entrevistador"
    = su voz→ES, incoming) para que el teleprompter los muestre en columnas
    distintas y no se mezclen (bug: "muestra todo lo que yo diga junto con lo
    del entrevistador").
    """

    def __init__(
        self,
        url: str = "http://localhost:8000/api/transcripcion",
        *,
        fuente: str = "yo",
    ) -> None:
        self._url = url
        self._fuente = fuente

    def mostrar(self, es: str, en: str) -> None:
        self._post({"es": es, "en": en, "fuente": self._fuente})

    def parcial(self, es: str) -> None:
        self._post({"es": es, "en": "", "fuente": self._fuente})

    def _post(self, payload: dict[str, str]) -> None:
        from contextlib import suppress

        import httpx

        with suppress(Exception):  # teleprompter caído no corta el flujo (ADR-008)
            httpx.post(self._url, json=payload, timeout=2.0)


def validar_arranque_real() -> bool:
    """Validación offline BLOQUEANTE (ADR-014/015): la traducción es→en debe
    funcionar sin red antes de aceptar una llamada (precarga del mwt)."""
    from traductor.flujo.outgoing import validar_arranque
    from traductor.traduccion.argos import traducir

    salud = validar_arranque(lambda es: traducir(es, "es", "en"))
    if not salud.disponible:
        print(f"Arranque BLOQUEADO: {salud.detalle}")
        return False
    print("Arranque offline de la traducción: OK (mwt precargado)")
    return True


def python_venv_tts() -> Path:  # pragma: no cover - ruta de máquina
    """El python del venv del TTS (coqui-tts vive ahí, ADR-011).

    parents[3]: src/traductor/flujo -> src/traductor -> src -> raíz. parents[2]
    caía en `src` y el venv nunca se encontraba (se usaba sys.executable sin
    coqui instalado -> el worker moría).
    """
    raiz = Path(__file__).resolve().parents[3]
    venv_tts = raiz / "venv-tts" / "Scripts" / "python.exe"
    if venv_tts.is_file():
        return venv_tts
    return Path(sys.executable)


def perfil_por_defecto() -> str:
    """Perfil del usuario por defecto (tienda JSON, ADR-013): el id del
    enrolamiento previo o 'kevin' si no hay ninguno."""
    id_ = os.environ.get("TRADUCTOR_PERFIL_ID", "kevin")
    return id_


def _buscar_device(pa: Any, nombre_parcial: str, canales: str, valor: int) -> int | None:
    """Índice del device de pyaudio cuyo nombre contiene `nombre_parcial` y
    cuya entrada/salida tiene AL MENOS `valor` canales; None si no existe
    (VB-CABLE ausente → el caller lanza con mensaje claro, no un
    StopIteration vacío).

    La comparación de nombre es CASE-INSENSITIVE: Windows/drivers varían la
    capitalización (el driver reporta "Voicemeeter Input" con m minúscula
    mientras el fabricante escribe "VoiceMeeter Input") y el usuario no debe
    adivinar la del driver.

    PREFIERE los devices del host API MME (hostApi 0): el motor de VB-Audio
    corre a 44100 y MME lo entrega NATIVO; el mismo device expuesto por
    WASAPI a 48000 pasa por un resampler defectuoso que inserta saltos de
    fase cada ~20 ms (audio con clics inaudibles para el oído pero que
    destruyen la transcripción — bug cazado con un tono puro de 440 Hz:
    WASAPI daba 522 Hz con saltos, MME daba 440.0 Hz exactos). El filtro
    usa `>=` porque los devices MME exponen 16 canales (no 2).
    """
    nombre = nombre_parcial.lower()
    candidatos: list[tuple[int, Any]] = []
    for i in range(pa.get_device_count()):
        info = pa.get_device_info_by_index(i)
        if nombre in str(info["name"]).lower() and int(info[canales]) >= valor:
            candidatos.append((i, info))
    if not candidatos:
        return None
    for i, info in candidatos:
        if int(info.get("hostApi", -1)) == 0:  # MME
            return i
    return candidatos[0][0]


def _tasa_nativa(pa: Any, indice: int, respaldo: int) -> int:
    """Tasa de muestreo NATIVA del device (`defaultSampleRate`), con respaldo.

    Los devices MME de VB-Audio reportan 44100 (la tasa real del motor); usar
    la tasa del device evita el resampler defectuoso de WASAPI a 48000.
    """
    try:
        return int(pa.get_device_info_by_index(indice)["defaultSampleRate"])
    except Exception:  # noqa: BLE001 - device fake en tests / driver raro
        return respaldo


def indice_cable_output() -> int:  # pragma: no cover - requiere VB-CABLE
    """Índice del device de ENTRADA del cable (CABLE Output) para el flujo
    incoming: el audio REMOTO del entrevistador se captura de aquí.

    En Meet/Zoom se configura CABLE Input como dispositivo de SALIDA de audio
    (el audio remoto entra al cable) y este flujo lee de CABLE Output. El
    nombre es configurable por env `TRADUCTOR_DEVICE_INCOMING` (default
    "CABLE Output"): con dos tubos, el incoming lee SOLO el cable del
    entrevistador, sin el TTS del outgoing mezclado.
    """
    import pyaudio

    nombre = os.environ.get("TRADUCTOR_DEVICE_INCOMING", "CABLE Output")
    pa = pyaudio.PyAudio()
    try:
        indice = _buscar_device(pa, nombre, "maxInputChannels", 2)
    finally:
        pa.terminate()
    if indice is None:
        raise RuntimeError(
            f"El device de entrada '{nombre}' no está disponible: instala "
            "VB-CABLE y configura en Meet/Zoom la SALIDA de audio en 'CABLE "
            "Input (VB-Audio Virtual Cable)' (o ajusta "
            "TRADUCTOR_DEVICE_INCOMING) antes de correr el flujo incoming"
        )
    return indice
