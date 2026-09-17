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


def _frases(texto: str) -> list[str]:
    """Parte `texto` en frases por puntuación fuerte, conservándola (pura).

    El batch por frase necesita unidades cortas: la primera debe estar lista
    rápido (cierre del turno) y cada una debe generarse más rápido de lo que
    suena (RTF < 1 sostenido). Puntúa también `:` y `;` porque el habla hace
    pausa ahí y unidades más cortas sostienen mejor el tiempo real.
    """
    frases: list[str] = []
    actual: list[str] = []
    for palabra in texto.split():
        actual.append(palabra)
        if palabra.endswith((".", "!", "?", "…", ":", ";")):
            frases.append(" ".join(actual))
            actual = []
    if actual:
        frases.append(" ".join(actual))
    return frases


class BackendXtts:
    """TTSBackend sobre XTTS-v2. `idioma_salida`: "en" para ES→EN (ADR-015)."""

    def __init__(self, idioma_salida: str = "en") -> None:
        self._idioma_salida = idioma_salida
        self._tts: Any | None = None

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
        """Sintetiza POR FRASE en batch: cada frase completa es un chunk.

        El camino anterior (`inference_stream`) generaba en incrementos
        pequeños y midió **RTF ~1.8 en la GTX 1650 Ti** (más lento que tiempo
        real: el stream de reproducción se quedaba sin datos y la voz salía
        "una frase bien, después palabra por palabra con pausas"). El batch
        por frase de `tts.tts()` mide **RTF 0.73-0.76** (más rápido que tiempo
        real) y una frase corta inicial ("Thank you.") devuelve el primer
        chunk en <1 s: el pipeline de cola/writer del `SalidaCable` (ADR-019)
        no cambia, solo el ritmo de producción.
        """
        tts = self._cargar()
        for frase in _frases(texto):
            wav = tts.tts(
                frase,
                speaker_wav=list(perfil.muestras),
                language=self._idioma_salida,
                split_sentences=False,
            )
            yield pcm_a_audio_result(_chunk_a_muestras(wav))  # sr default = SR_XTTS

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
