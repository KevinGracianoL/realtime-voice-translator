"""Backend XTTS-v2 sobre el fork mantenido `coqui-tts` (idiap) — candidato primario.

Implementa `TTSBackend` (ADR-011): `sintetizar(texto, perfil)`, `verificar_salud`,
`cerrar`. El import del motor es lazy: en CI `coqui_tts` no está instalado y
`verificar_salud()` lo reporta (disponible=False con detalle) SIN cargar el
modelo (verificar_salud no tiene efectos secundarios). La síntesis real corre
en la máquina objetivo (harness del ADR-014) y las líneas que tocan el modelo
van `# pragma: no cover`.

El perfil llega pre-enrolado (ADR-011): `perfil.muestras` son las referencias
de voz; XTTS las usa para clonar el timbre en el idioma de salida.
"""

from __future__ import annotations

from array import array
from collections.abc import Iterator, Sequence
from typing import Any

from traductor.tts.modelos import AudioResult, Salud, VoiceProfile

MODELO_XTTS = "tts_models/multilingual/multi-dataset/xtts_v2"
SR_XTTS = 24000


def pcm_a_audio_result(muestras: Sequence[float], sr: int = SR_XTTS) -> AudioResult:
    """Convierte muestras float32 a `AudioResult` (formato + duración honestos).

    Pura y testeable sin el motor: fija la UNIDAD de la duración (len/sr, no
    len/sr/1000) y el formato (pcm_f32le). El motor devuelve un array float32;
    `array('f')` lo serializa sin depender de numpy en CI.
    """
    datos = array("f", muestras).tobytes()
    return AudioResult(datos=datos, formato="pcm_f32le", duracion_s=float(len(muestras)) / sr)


def _chunk_a_muestras(chunk: Any) -> list[float]:
    """Chunk del generador XTTS (tensor torch o ndarray float32) → list[float].

    Pura y testeable sin el motor ni numpy en CI: acepta listas, ndarrays o
    tensores; aplana (reshape(-1)) y convierte a float.
    """
    if hasattr(chunk, "detach") and hasattr(chunk, "cpu"):  # tensor torch
        chunk = chunk.detach().cpu()
    plano = chunk.reshape(-1) if hasattr(chunk, "reshape") else chunk
    return [float(v) for v in plano]


_CIERRES = "\"'”’»)]}"
_ABREVIATURAS = frozenset(
    {
        "sr",
        "sra",
        "srta",
        "dr",
        "dra",
        "prof",
        "ud",
        "uds",
        "etc",
        "aprox",
        "pag",
        "pág",
        "num",
        "núm",
        "mr",
        "mrs",
        "ms",
        "jr",
        "st",
        "mt",
        "vs",
        "no",
        "vol",
        "fig",
        "al",
        "ca",
    }
)
_MAX_CARACTERES_FRASE = 180


def _es_abreviatura(palabra: str) -> bool:
    """True si el token termina en abreviatura o sigla con puntos (pura).

    `'Mr.'` y `'U.S.'` no cierran frase; `'years.'` y `'stop.'` sí. Las siglas
    se detectan por forma (partes de hasta 2 letras: `U.S`, `e.g`, `a.m`) y las
    abreviaturas por lista — sin esto, el batch parte `'Mr. Smith…'` en dos
    unidades y la entonación cierra donde no toca (hallazgo del review).
    """
    nucleo = palabra.strip(_CIERRES).rstrip(".,;:!?…")
    if not nucleo:
        return False
    if "." in nucleo:
        return all(1 <= len(parte) <= 2 for parte in nucleo.split("."))
    return nucleo.lower() in _ABREVIATURAS


def _cierra_frase(palabra: str, palabras: list[str], indice: int) -> bool:
    """True si el punto de `palabra` cierra frase (pura).

    Un punto solo cierra si le sigue fin de texto o una mayúscula, y la
    palabra no es abreviatura: `'3.5 years'` y `'The U.S. team'` siguen en
    minúscula y no parten; `'Mr. Smith'` sí parece cierre por la mayúscula,
    y lo frena la lista de abreviaturas.
    """
    if indice + 1 == len(palabras):
        return True
    siguiente = palabras[indice + 1].lstrip("\"'“”‘’«([{")
    if siguiente and not siguiente[0].isupper():
        return False
    return not _es_abreviatura(palabra)


def _limitar(frases: list[str]) -> list[str]:
    """Parte las frases que superan el tope, sin perder texto (pura).

    Sin tope, un párrafo sin puntuación sería UN chunk gigante: el primer
    chunk tardaría lo que la síntesis completa (el "Thank you." corto de la
    demo es guion, no garantía). Corta por coma, luego por espacio, y en
    último caso duro — cada trozo cabe en `_MAX_CARACTERES_FRASE`.
    """
    resultado: list[str] = []
    for frase in frases:
        while len(frase) > _MAX_CARACTERES_FRASE:
            corte = frase.rfind(",", 0, _MAX_CARACTERES_FRASE + 1)
            if corte > 0:
                parte, frase = frase[: corte + 1], frase[corte + 1 :].lstrip()
            else:
                corte = frase.rfind(" ", 0, _MAX_CARACTERES_FRASE + 1)
                if corte > 0:
                    parte, frase = frase[:corte], frase[corte + 1 :]
                else:
                    parte, frase = frase[:_MAX_CARACTERES_FRASE], frase[_MAX_CARACTERES_FRASE:]
            resultado.append(parte)
        if frase:
            resultado.append(frase)
    return resultado


def _frases(texto: str) -> list[str]:
    """Parte `texto` en frases por puntuación fuerte, conservándola (pura).

    El batch por frase necesita unidades cortas: la primera debe estar lista
    rápido (cierre del turno) y cada una debe generarse más rápido de lo que
    suena (RTF < 1 sostenido). Puntúa también `:` y `;` porque el habla hace
    pausa ahí y unidades más cortas sostienen mejor el tiempo real. Los
    puntos de abreviaturas, decimales y siglas no parten (ver `_es_abreviatura`
    y `_cierra_frase`), y `_limitar` acota los tramos sin puntuar.
    """
    palabras = texto.split()
    frases: list[str] = []
    actual: list[str] = []
    for indice, palabra in enumerate(palabras):
        actual.append(palabra)
        sin_cierre = palabra.rstrip(_CIERRES)
        if not sin_cierre:
            continue
        signo = sin_cierre[-1]
        if signo not in ".!?…:;":
            continue
        if signo == "." and not _cierra_frase(palabra, palabras, indice):
            continue
        frases.append(" ".join(actual))
        actual = []
    if actual:
        frases.append(" ".join(actual))
    return _limitar(frases)


class BackendXtts:
    """TTSBackend sobre XTTS-v2. `idioma_salida`: "en" para ES→EN (ADR-015)."""

    def __init__(self, idioma_salida: str = "en") -> None:
        self._idioma_salida = idioma_salida
        self._tts: Any | None = None
        self._latentes_por_perfil: dict[str, tuple[Any, Any]] = {}

    def _cargar(self) -> Any:  # pragma: no cover - requiere coqui_tts + GPU
        """Carga el modelo una sola vez (lazy). Raises: RuntimeError."""
        if self._tts is None:
            try:
                from TTS.api import TTS
            except ImportError as exc:
                raise RuntimeError(
                    "coqui-tts no instalado: instala el fork idiap/coqui-ai-TTS "
                    "(venv propio del TTS, ver ADR-011/014)"
                ) from exc
            try:
                self._tts = TTS(MODELO_XTTS, gpu=True)
            except Exception as exc:
                raise RuntimeError(f"XTTS-v2 no cargó: {exc}") from exc
        return self._tts

    def sintetizar(self, texto: str, perfil: VoiceProfile) -> AudioResult:  # pragma: no cover
        """Sintetiza `texto` con el timbre de `perfil.muestras`.

        Raises:
            RuntimeError: si el motor no está disponible (ver `verificar_salud`).
        """
        tts = self._cargar()
        wav = tts.tts(
            texto,
            speaker_wav=list(perfil.muestras),
            language=self._idioma_salida,
            split_sentences=True,
        )
        return pcm_a_audio_result(wav, SR_XTTS)

    def sintetizar_stream(self, texto: str, perfil: VoiceProfile) -> Iterator[AudioResult]:
        """Sintetiza POR FRASE en batch con latentes cacheadas por perfil.

        Es el camino declarado tras el review del PR #34: conserva la ganancia
        del batch (RTF < 1 sostenido) Y la caché del ADR-011/014/019 — llamar
        `tts.tts(frase, speaker_wav=...)` recalcula las latentes de
        condicionamiento (756 ms) en CADA llamada, porque el fork solo las
        cachea con `speaker_id` y aquí el perfil llega pre-enrolado con
        referencias.         `inference()` es el gemelo no-streaming del
        `inference_stream` viejo: mismo vocoder, sin el bucle de streaming que
        medía RTF ~1.8 en la GTX 1650 Ti (el `SalidaCable` reproduce a 1x y el
        buffer se agotaba: "una frase bien, después palabra por palabra").
        Aplica los settings de generación del config del modelo (temperatura,
        penalizaciones, top-k/p): los defaults del método difieren del config
        y el audio cambia (medido: `repetition_penalty` 10 vs 5).

        Cada frase completa es un chunk al pipeline de cola/writer del
        ADR-019 (contrato intacto); `_frases` garantiza que la primera sea
        corta y `_limitar` acota los tramos sin puntuar.
        """
        tts = self._cargar()
        modelo = tts.synthesizer.tts_model
        config = modelo.config
        ajustes = {
            "temperature": config.temperature,
            "length_penalty": config.length_penalty,
            "repetition_penalty": config.repetition_penalty,
            "top_k": config.top_k,
            "top_p": config.top_p,
        }
        gpt, spk = self._latentes(perfil)
        for frase in _frases(texto):
            wav = modelo.inference(frase, self._idioma_salida, gpt, spk, **ajustes)["wav"]
            yield pcm_a_audio_result(_chunk_a_muestras(wav))  # sr default = SR_XTTS

    def _latentes(self, perfil: VoiceProfile) -> tuple[Any, Any]:
        """Latentes del perfil, cacheadas (una vez por id de perfil, ADR-014)."""
        if perfil.id not in self._latentes_por_perfil:
            tts = self._cargar()
            latentes = tts.synthesizer.tts_model.get_conditioning_latents(
                audio_path=list(perfil.muestras)
            )
            self._latentes_por_perfil[perfil.id] = (latentes[0], latentes[1])
        return self._latentes_por_perfil[perfil.id]

    def verificar_salud(self) -> Salud:
        """Estado SIN efectos secundarios: no carga el modelo (r1 PR #16).

        El modelo se carga explícitamente en la primera `sintetizar()` (o en
        el harness antes de medir). `verificar_salud` solo comprueba instalado
        y cargado — no contamina `vram_base` por orden de llamadas.
        """
        if self._tts is None:
            try:
                import TTS  # noqa: F401 - solo comprueba instalación
            except ImportError as exc:
                return Salud(disponible=False, detalle=f"coqui-tts no instalado: {exc}")
            return Salud(  # pragma: no cover - solo con coqui-tts instalado
                disponible=False, detalle="modelo no cargado: llama a sintetizar() primero"
            )
        return Salud(disponible=True, detalle=f"XTTS-v2 listo ({MODELO_XTTS})")

    def cerrar(self) -> None:
        """Suelta el modelo. Idempotente.

        NOTA: el allocator de torch puede retener el pool de VRAM tras esto —
        para la escalera del ADR-015 sin reiniciar, `empty_cache()` es
        best-effort y puede requerirse reiniciar el worker del TTS.
        """
        self._tts = None
