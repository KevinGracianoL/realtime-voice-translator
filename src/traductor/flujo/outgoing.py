"""Flujo outgoing_es_to_en (ADR-015): mic → VAD → ASR es → Argos → teleprompter → TTS → VB-CABLE.

El core es PURO y testeable: las etapas se inyectan (traducción, TTS primario
y de fallback, teleprompter, salida de audio) y los adaptadores reales
(micrófono, worker TTS, VB-CABLE, teleprompter HTTP) viven en
`traductor.flujo.adaptadores` y se cablean en `scripts/flujo_outgoing.py`.

Reglas del ADR-015 implementadas aquí:
- los PARCIALES solo llegan a pantalla (nunca a traducción/TTS);
- cola de tamaño 1: una respuesta atrasada se descarta;
- cancelación: una solicitud nueva cancela la antigua (el audio de un turno
  superado nunca se enruta);
- timestamps por etapa con cierre exacto (`RegistroEtapas`);
- escalera de presupuesto: clonado → voz genérica → solo subtítulos (nunca
  reproducir audio sospechoso).

Cierre del turno con STREAMING (ADR-019, fixes 1-2): si la etapa TTS tiene
`sintetizar_stream`, el turno cierra con el PRIMER chunk (~0.7 s de síntesis
+ el primer bloque del cable); el resto del audio llega en un hilo daemon que
lo reproduce si el turno sigue activo o lo DRENA sin reproducir si se canceló
(el pipe del worker queda alineado: el siguiente turno lee SUS líneas).
"""

from __future__ import annotations

import threading
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any, Protocol, runtime_checkable

from traductor.tts.harness import RegistroEtapas
from traductor.tts.modelos import Salud

NIVEL_CLONADO = 2
NIVEL_VOZ_GENERICA = 3
NIVEL_SUBTITULOS = 4

_TEXTO_PROBE_ARRANQUE = "Mi experiencia mas fuerte es con sistemas distribuidos."
DETALLE_BASURA = (
    "traducción es→en produce basura al arrancar: precarga del modelo "
    "spacy mwt no efectiva (el worker de traducción NO entra al flujo)"
)


class Teleprompter(Protocol):
    def mostrar(self, es: str, en: str) -> None: ...
    def parcial(self, es: str) -> None: ...


class SalidaAudio(Protocol):
    def reproducir(self, audio: bytes, duracion_s: float, nombre: str) -> None: ...


class EtapaTts(Protocol):
    """Etapa de síntesis. None = el backend falló (escalera).

    `audio` es WAV; `nombre` identifica la voz (para logs y la escalera).
    """

    def sintetizar(self, texto_en: str) -> tuple[bytes, float, str] | None: ...


@runtime_checkable
class EtapaTtsStream(Protocol):
    """Etapa con streaming (ADR-019): `sintetizar_stream` es un generador de
    WAVs por chunk; el PRIMERO llega rápido y el flujo cierra el turno con él.
    """

    def sintetizar_stream(self, texto_en: str) -> Any | None: ...


@dataclass
class FlujoOutgoing:
    """Orquesta un turno: segmento final → traducción → TTS (escalera) → ruteo.

    Contrato de hilos (corregido en la revisión del PR #22): RealtimeSTT
    entrega parciales y finales desde SUS propios hilos, y cada `segmento_final`
    corre en un hilo worker del adaptador. Por eso el core es seguro bajo
    concurrencia:
    - el contador de turnos y el turno activo se mutan bajo `_lock`;
    - la cancelación (`cancelar_turno_activo`, invocada por el callback de
      parciales cuando el usuario vuelve a hablar) y el arranque de un turno
      nuevo invalidan el turno en vuelo: el check `turno != self._turno_activo`
      (lectura atómica) descarta su audio y su texto según el momento;
    - la ventana de carrera restante (cancelar justo entre el check y el ruteo)
      es la duración de una escritura al cable: se documenta, no se promete
      cero.

    Streaming (ADR-019): el turno cierra con el primer chunk; el hilo daemon
    que reproduce el resto verifica el turno entre chunk y chunk, y si el turno
    se canceló DRENA el generador sin reproducir (el pipe del worker queda
    alineado para el siguiente turno).
    """

    traducir: Callable[[str], str]
    tts_primario: EtapaTts
    teleprompter: Teleprompter
    salida_audio: SalidaAudio
    tts_fallback: EtapaTts | None = None
    verificar_artefactos: Callable[[bytes], str | None] | None = None
    max_sobrantes: int = 10
    reloj: Callable[[], float] = field(default_factory=lambda: __import__("time").perf_counter)
    ultimo_turno_etapas: dict[str, float] = field(default_factory=dict, init=False)
    ultimo_turno_total_ms: float = field(default=0.0, init=False)
    ultimo_turno_degradado: bool = field(default=False, init=False)
    _numero_turno: int = field(default=0, init=False)
    _turno_activo: int | None = field(default=None, init=False)
    _lock: Any = field(default_factory=threading.Lock, init=False, repr=False)

    def parcial(self, texto_es: str) -> None:
        """Parcial del ASR: SOLO a pantalla (regla del ADR-015)."""
        self.teleprompter.parcial(texto_es)

    def cancelar_turno_activo(self) -> None:
        """Cancela el turno en curso. Lo invoca el callback de PARCIALES del
        micrófono cuando el usuario vuelve a hablar: la síntesis del turno
        anterior queda invalidada y su audio no se enruta."""
        with self._lock:
            self._turno_activo = None

    def segmento_final(self, texto_es: str) -> int | None:
        """Procesa un segmento final; devuelve el nivel de la escalera usado
        (None si el turno fue cancelado antes de enrutar)."""
        with self._lock:
            self._numero_turno += 1
            turno = self._numero_turno
            self._turno_activo = turno
        self.ultimo_turno_degradado = False
        etapas = RegistroEtapas(clock=self.reloj)
        etapas.marcar("entrada")
        texto_en = self.traducir(texto_es)
        etapas.marcar("traduccion")
        if turno != self._turno_activo:
            return None  # cancelado: el texto de un turno superado no se muestra
        self.teleprompter.mostrar(texto_es, texto_en)
        escalera: list[tuple[str, EtapaTts]] = [("clonado", self.tts_primario)]
        if self.tts_fallback is not None:
            escalera.append(("generica", self.tts_fallback))
        for nombre, tts in escalera:
            if isinstance(tts, EtapaTtsStream):
                cancelado, nivel = self._enrutar_stream(turno, nombre, tts, texto_en, etapas)
                if cancelado:
                    return None  # cancelado: no probar la escalera
                if nivel is not None:
                    self.ultimo_turno_etapas = etapas.desglose_ms()
                    self.ultimo_turno_total_ms = etapas.total_ms()
                    return nivel
                continue  # el stream falló: escalera
            salida = tts.sintetizar(texto_en)
            if salida is None:
                continue  # escalera: siguiente nivel
            audio, duracion_s, _nombre = salida
            if turno != self._turno_activo:
                return None  # cancelado: el audio de un turno superado no se enruta
            if not self._audio_sin_artefactos(audio, texto_en):
                self.ultimo_turno_degradado = True
                continue  # audio sospechoso: escalera (nunca reproducirlo)
            etapas.marcar("tts")
            self.salida_audio.reproducir(audio, duracion_s, nombre)
            etapas.marcar("ruteo")
            self.ultimo_turno_etapas = etapas.desglose_ms()
            self.ultimo_turno_total_ms = etapas.total_ms()
            return NIVEL_CLONADO if nombre == "clonado" else NIVEL_VOZ_GENERICA
        self.ultimo_turno_etapas = etapas.desglose_ms()
        self.ultimo_turno_total_ms = etapas.total_ms()
        return NIVEL_SUBTITULOS  # ambos TTS fallaron: solo subtítulos, sin audio

    def _enrutar_stream(
        self,
        turno: int,
        nombre: str,
        tts: EtapaTtsStream,
        texto_en: str,
        etapas: RegistroEtapas,
    ) -> tuple[bool, int | None]:
        """Camino streaming (ADR-019): cierra con el primer chunk y el resto
        en un hilo daemon. (cancelado, nivel) — cancelado corta la escalera.

        Validación de artefactos (fix 3): se corre sobre el audio COMPLETO
        del turno (no por chunk — whisper alucina en fragmentos cortos y no
        discrimina, ADR-019) acumulando los chunks en el hilo daemon. Un
        turno degradado se marca y NO se reproduce el resto.
        """
        generador = tts.sintetizar_stream(texto_en)
        if generador is None:
            return False, None  # el worker falló: escalera
        try:
            primero = next(generador)
        except StopIteration:
            return False, None  # stream vacío: escalera
        if turno != self._turno_activo:
            return True, None  # cancelado antes del primer chunk
        etapas.marcar("tts")  # el primer chunk llegó
        self.salida_audio.reproducir(*primero)
        etapas.marcar("ruteo")  # cierre del turno: primer bloque audible
        hilo = threading.Thread(
            target=self._reproducir_resto,
            args=(turno, texto_en, generador, primero[0]),
            daemon=True,
        )
        hilo.start()
        return False, NIVEL_CLONADO if nombre == "clonado" else NIVEL_VOZ_GENERICA

    def _reproducir_resto(
        self, turno: int, texto_en: str, generador: Any, primer_audio: bytes
    ) -> None:
        """Reproduce el resto del stream si el turno sigue activo; si se
        canceló, DRENA sin reproducir (el pipe del worker queda alineado).

        Al terminar, valida el audio COMPLETO del turno contra el texto
        traducido (fix 3): si hay sobrantes por encima del umbral, el turno
        se marca degradado (el audio ya enrutado no se des-enruta; la
        validación es del audio completo, no de fragmentos — calibración en
        ADR-019: limpio máx 8, corrupto mín 11).
        """
        audios: list[bytes] = [primer_audio]
        for audio, duracion_s, _nombre in generador:
            audios.append(audio)
            if turno == self._turno_activo:
                self.salida_audio.reproducir(audio, duracion_s, _nombre)
        if self.verificar_artefactos is not None and turno == self._turno_activo:
            completo = self._concatenar_wav(audios)
            if not self._audio_sin_artefactos(completo, texto_en):
                self.ultimo_turno_degradado = True

    def _concatenar_wav(self, audios: list[bytes]) -> bytes:
        """Une los chunks WAV del turno en un solo WAV (misma tasa)."""
        import io
        import wave

        if len(audios) == 1:
            return audios[0]
        frames = bytearray()
        tasa = 0
        for a in audios:
            with wave.open(io.BytesIO(a), "rb") as w:
                if tasa == 0:
                    tasa = w.getframerate()
                frames.extend(w.readframes(w.getnframes()))
        buf = io.BytesIO()
        with wave.open(buf, "wb") as w:
            w.setnchannels(1)
            w.setsampwidth(2)
            w.setframerate(tasa)
            w.writeframes(bytes(frames))
        return buf.getvalue()

    def _audio_sin_artefactos(self, audio: bytes, texto_en: str) -> bool:
        """ASR-de-retorno del audio contra el texto TRADUCIDO (ADR-019 fix 3).

        El desvío puede venir de la traducción misma ('trauthor' — hallazgo
        de la revisión del PR #23) y no es del audio; la comparación es
        contra el texto traducido, no el intencionado. Con
        `verificar_artefactos` None la validación no corre (el turno vale).
        """
        if self.verificar_artefactos is None:
            return True
        transcripcion = self.verificar_artefactos(audio)
        if transcripcion is None:
            return True  # instrumento roto: no bloquear el turno
        sobrantes = _sobrantes(transcripcion, texto_en)
        return len(sobrantes) <= self.max_sobrantes


def _normalizar(texto: str) -> str:
    return " ".join("".join(c for c in texto.lower() if c.isalnum() or c.isspace()).split())


def _sobrantes(transcripcion: str, esperado: str) -> list[str]:
    """Palabras del ASR-de-retorno que NO están en el texto esperado.

    ADR-019, fix 3: en el streaming por chunks la validación NO puede exigir
    todas las palabras (el texto completo aún no se sintetizó); lo que SÍ se
    puede exigir es que el audio no diga palabras que el texto no contiene
    (un artefacto añade basura; 'trauthor' SÍ está en el texto traducido).
    """
    retorno = _normalizar(transcripcion).split()
    esperadas = set(_normalizar(esperado).split())
    return [palabra for palabra in retorno if palabra not in esperadas]


def validar_arranque(
    traducir: Callable[[str], str],
    *,
    probe: str = _TEXTO_PROBE_ARRANQUE,
) -> Salud:
    """Valida el arranque del flujo de forma BLOQUEANTE (ADR-014/015).

    La traducción es→en debe funcionar SIN red antes de aceptar una llamada:
    la precarga del modelo spacy `mwt` (que argos descarga on-demand y mató 1
    de 20 corridas del harness) se comprueba traduciendo una frase de prueba y
    verificando que la salida sea sana (mismo sanity laxo del harness: frase
    clave presente, longitud acotada, sin repetición patológica).
    """
    try:
        salida = _normalizar(traducir(probe))
    except Exception as exc:
        return Salud(disponible=False, detalle=f"traducción no disponible al arrancar: {exc}")
    palabras = salida.split()
    if not palabras or "distributed systems" not in salida or len(salida) > 80:
        return Salud(disponible=False, detalle=DETALLE_BASURA)
    if max(palabras.count(p) for p in set(palabras)) > 3:
        return Salud(disponible=False, detalle=DETALLE_BASURA)
    return Salud(disponible=True)
