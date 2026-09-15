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
from typing import Any

from traductor.flujo.outgoing import FlujoOutgoing
from traductor.tts.backend_xtts import SR_XTTS


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
    """Micrófono → segmentos: los PARCIALES cancelan y van a pantalla; los
    FINALES se procesan en un hilo worker por turno.

    RealtimeSTT ya integra VAD/endpointing y entrega sus callbacks desde SUS
    propios hilos (revisión #22: el invariante de hilo único era falso).
    - `_parcial`: el usuario volvió a hablar → se CANCELA la síntesis del
    turno anterior en vuelo (`cancelar_turno_activo`) y el parcial va a
    pantalla. Sin esto, la cancelación del ADR-015 era código muerto.
    - `_final`: el turno corre en un hilo daemon para que el micrófono siga
    fluyendo y un nuevo segmento pueda cancelarlo (cola de tamaño 1).
    """

    def __init__(self, flujo: FlujoOutgoing) -> None:
        self._flujo = flujo

    def _parcial(self, texto: str) -> None:
        self._flujo.cancelar_turno_activo()
        self._flujo.parcial(texto)

    def _final(self, texto: str) -> None:
        import threading

        threading.Thread(target=self._flujo.segmento_final, args=(texto,), daemon=True).start()

    def correr(self) -> None:
        from RealtimeSTT import AudioToTextRecorder

        grabador = AudioToTextRecorder(
            model="tiny",
            language="es",
            device="cuda",
            compute_type="int8",
            on_realtime_transcription_update=self._parcial,
        )
        print("Flujo outgoing listo: habla en español. Ctrl+C para salir.")
        while True:
            grabador.text(self._final)


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

    ADR-019, fix 2: escribe POR BLOQUES — `reproducir` devuelve cuando el
    PRIMER bloque se acepta (el primer sample es audible) y el resto del
    audio se escribe en un hilo daemon. El cierre del turno ya no espera la
    reproducción completa (la duración del audio NO es latencia).

    `abrir()`/`cerrar()` permiten mantener el stream abierto entre turnos
    (abrir por turno paga ~1-3 s de overhead en Windows).
    """

    def __init__(self, rate_cable: int = 48000, bloque_s: float = 0.2) -> None:
        self._rate_cable = rate_cable
        self._bloque_s = bloque_s
        self._pa: Any | None = None
        self._stream: Any | None = None
        self._lock_escritura: Any = None  # un solo escritor del stream a la vez

    def abrir(self) -> None:
        import threading

        import pyaudio

        self._lock_escritura = threading.Lock()
        self._pa = pyaudio.PyAudio()
        indice = next(
            i
            for i in range(self._pa.get_device_count())
            if "CABLE Input" in str(self._pa.get_device_info_by_index(i)["name"])
            and self._pa.get_device_info_by_index(i)["maxOutputChannels"] == 2
        )
        self._stream = self._pa.open(
            format=pyaudio.paInt16,
            channels=2,
            rate=self._rate_cable,
            output=True,
            output_device_index=indice,
        )

    def cerrar(self) -> None:
        if self._stream is not None:
            self._stream.stop_stream()
            self._stream.close()
            self._stream = None
        if self._pa is not None:
            self._pa.terminate()
            self._pa = None

    def reproducir(self, audio: bytes, duracion_s: float, nombre: str) -> None:
        """Primer bloque al stream y devuelve; el resto en un hilo daemon.

        El primer sample audible es el momento del primer bloque aceptado:
        el cierre del turno (ADR-019) es ESE momento, no la duración del
        audio. Sin stream (no abierto): no se reproduce (escalera).
        """
        stream = self._stream
        lock = self._lock_escritura
        if stream is None or lock is None:
            return  # sin stream: no reproducir (escalera del cable)
        pcm = self._audio_a_pcm_cable(audio)
        # bloque en BYTES: rate * 2 canales * 2 bytes (int16) * segundos
        bloque = max(1, int(self._rate_cable * 4 * self._bloque_s))
        bloques = [pcm[i : i + bloque] for i in range(0, len(pcm), bloque)]
        if not bloques:
            return  # audio vacío: nada que reproducir
        if len(bloques) == 1:
            # un solo bloque: no hay nada que encolar (mutante `>= 1` lo caza)
            with lock:
                stream.write(bloques[0])
            return
        restantes = bloques[1:]
        with lock:  # un solo escritor: el daemon del turno anterior y el
            # primer bloque del turno nuevo NO pueden escribir a la vez
            stream.write(bloques[0])  # PRIMER bloque: cierre del turno
        self._lanzar_resto(stream, lock, restantes)

    def _lanzar_resto(self, stream: Any, lock: Any, restantes: list[bytes]) -> Any:
        """Hilo daemon que escribe el resto (devuelto: el test verifica que
        es daemon y que el cierre del turno no espera la reproducción)."""
        import threading

        hilo = threading.Thread(
            target=self._escribir_resto, args=(stream, lock, restantes), daemon=True
        )
        hilo.start()
        return hilo

    def _escribir_resto(self, stream: Any, lock: Any, restantes: list[bytes]) -> None:
        for bloque in restantes:
            with lock:
                stream.write(bloque)

    def _audio_a_pcm_cable(self, audio: bytes) -> bytes:
        """WAV → PCM int16 estéreo 48 kHz para el cable (resample lineal)."""
        import io

        import numpy as np
        import soundfile as sf

        datos, sr = sf.read(io.BytesIO(audio), dtype="float32")
        mono = np.asarray(datos, dtype=np.float32)
        # resample LINEAL a 48 kHz (revisión #22: la división entera distorsiona
        # el tono con sr que no divide 48000, p. ej. 22050 del candidato B)
        ratio = self._rate_cable / sr
        n_salida = int(len(mono) * ratio)
        pos = np.arange(n_salida, dtype=np.float32) / ratio
        i0 = pos.astype(np.int64)
        i1 = np.minimum(i0 + 1, len(mono) - 1)
        alpha = (pos - i0).astype(np.float32)
        res = mono[i0] * (1 - alpha) + mono[i1] * alpha
        stereo = np.repeat(res, 2)
        return bytes((np.clip(stereo, -1.0, 1.0) * 32767).astype(np.int16).tobytes())


class TeleprompterHttp:  # pragma: no cover - requiere el UI corriendo
    """Manda el texto al teleprompter local (FastAPI, POST /api/transcripcion)."""

    def __init__(self, url: str = "http://localhost:8000/api/transcripcion") -> None:
        self._url = url

    def mostrar(self, es: str, en: str) -> None:
        self._post({"es": es, "en": en})

    def parcial(self, es: str) -> None:
        self._post({"es": es, "en": ""})

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
    """El python del venv del TTS (coqui-tts vive ahí, ADR-011)."""
    raiz = Path(__file__).resolve().parents[2]
    venv_tts = raiz / "venv-tts" / "Scripts" / "python.exe"
    if venv_tts.is_file():
        return venv_tts
    return Path(sys.executable)


def perfil_por_defecto() -> str:
    """Perfil del usuario por defecto (tienda JSON, ADR-013): el id del
    enrolamiento previo o 'kevin' si no hay ninguno."""
    id_ = os.environ.get("TRADUCTOR_PERFIL_ID", "kevin")
    return id_
