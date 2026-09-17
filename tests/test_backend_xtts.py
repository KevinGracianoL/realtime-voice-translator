"""Tests del BackendXtts — estado sin cargar (el que corre en CI).

El motor real (coqui-tts + modelo XTTS-v2) no está en CI: se prueba el
contrato en estado no cargado (healthcheck sin efectos secundarios, cierre
idempotente, error claro al sintetizar sin motor) y la conversión PCM pura
(unidad de duración fijada: len/sr, no len/sr/1000).
"""

from collections.abc import Iterator

import pytest

from traductor.tts.backend import TTSBackend
from traductor.tts.backend_xtts import (
    MODELO_XTTS,
    BackendXtts,
    _chunk_a_muestras,
    pcm_a_audio_result,
)
from traductor.tts.modelos import VoiceProfile

PERFIL = VoiceProfile(id="kevin-es", nombre="Kevin", muestras=("ref1.wav", "ref2.wav"))


def test_backend_satisface_el_contrato() -> None:
    assert isinstance(BackendXtts(), TTSBackend)


def test_idioma_salida_por_defecto_es_en() -> None:
    assert BackendXtts()._idioma_salida == "en"
    assert BackendXtts(idioma_salida="es")._idioma_salida == "es"


def test_verificar_salud_sin_motor_reporta_indisponible() -> None:
    """En CI (sin coqui-tts) el healthcheck es honesto: False con detalle."""
    salud = BackendXtts().verificar_salud()
    assert salud.disponible is False
    assert salud.detalle.startswith("coqui-tts no instalado: ")


def test_verificar_salud_no_carga_el_modelo() -> None:
    """verificar_salud no tiene efectos secundarios: no carga (r1 PR #16)."""
    backend = BackendXtts()
    backend.verificar_salud()
    assert backend._tts is None


def test_frases_parte_por_puntuacion_fuerte() -> None:
    """El batch por frase parte el texto en unidades cortas conservando la
    puntuacion (".", "!", "?", ":", ";")."""
    from traductor.tts.backend_xtts import _frases

    assert _frases("Thank you. Recently I solved it: in production. Today!") == [
        "Thank you.",
        "Recently I solved it:",
        "in production.",
        "Today!",
    ]
    # sin puntuacion: una sola frase (no se pierde nada)
    assert _frases("hola mundo") == ["hola mundo"]
    # vacio / solo espacios
    assert _frases("") == []
    assert _frases("   ") == []
    # la cola sin puntuar se conserva
    assert _frases("Uno. Dos sin punto") == ["Uno.", "Dos sin punto"]


def test_sintetizar_stream_por_frase_batch() -> None:
    """`sintetizar_stream` emite UN AudioResult por frase, con batch `tts.tts`
    (no el inference_stream lento): RTF medido 1.8 vs 0.73 en la 1650 Ti."""
    from traductor.tts.backend_xtts import _frases

    backend = BackendXtts()
    llamadas: list[tuple[str, list[str], str]] = []

    class _TtsFake:
        def tts(
            self,
            texto: str,
            speaker_wav: list[str],
            language: str,
            split_sentences: bool = True,
        ) -> list[float]:
            llamadas.append((texto, speaker_wav, language))
            assert split_sentences is False  # la frase ya es una unidad
            return [0.1] * 240  # 0.01 s de audio por frase

    backend._tts = _TtsFake()
    salidas = list(backend.sintetizar_stream("Uno. Dos!", PERFIL))
    assert len(salidas) == len(_frases("Uno. Dos!"))
    assert [c[0] for c in llamadas] == ["Uno.", "Dos!"]
    assert llamadas[0][1] == list(PERFIL.muestras)
    assert llamadas[0][2] == "en"
    for salida in salidas:
        assert salida.formato == "pcm_f32le"
        assert salida.duracion_s == pytest.approx(240 / 24000)


def test_verificar_salud_con_modelo_cargado() -> None:
    """Simulando el modelo cargado, el healthcheck dice disponible (r2 PR #16)."""
    backend = BackendXtts()
    backend._tts = object()  # sin el motor real: solo la rama cargada
    salud = backend.verificar_salud()
    assert salud.disponible is True
    assert MODELO_XTTS in salud.detalle


def test_sintetizar_sin_motor_error_claro() -> None:
    backend = BackendXtts()
    with pytest.raises(
        RuntimeError,
        match=(
            "^coqui-tts no instalado: instala el fork idiap/coqui-ai-TTS "
            r"\(venv propio del TTS, ver ADR-011/014\)$"
        ),
    ):
        backend.sintetizar("hello", PERFIL)


def test_cerrar_idempotente() -> None:
    backend = BackendXtts()
    backend.cerrar()
    backend.cerrar()
    assert backend._tts is None


def test_modelo_es_xtts_v2() -> None:
    assert MODELO_XTTS == "tts_models/multilingual/multi-dataset/xtts_v2"


def test_pcm_a_audio_result_bytes_y_duracion() -> None:
    """Unidad de duración: len/sr (si fuera len/sr/1000, este test lo caza)."""
    muestras = [0.0, 0.5, -0.5, 1.0]
    audio = pcm_a_audio_result(muestras)
    assert audio.formato == "pcm_f32le"
    assert len(audio.datos) == 4 * 4  # 4 muestras float32
    assert audio.duracion_s == pytest.approx(4 / 24000)


def test_pcm_a_audio_result_sr_personalizado() -> None:
    audio = pcm_a_audio_result([0.0, 0.25], sr=16000)
    assert audio.duracion_s == pytest.approx(2 / 16000)


def test_chunk_a_muestras_con_lista() -> None:
    assert _chunk_a_muestras([0.0, 0.5, -0.5]) == [0.0, 0.5, -0.5]


class _IterableFake:
    """Iterable de muestras (lo que devuelve reshape(-1))."""

    def __init__(self, valores: list[float]) -> None:
        self._valores = valores

    def __iter__(self) -> Iterator[float]:
        return iter(self._valores)


class _PlanoFake:
    """Objeto aplanable estilo ndarray: reshape(-1) es la ÚNICA forma de
    iterarlo (mutantes de `reshape` caen aquí: sin él no se puede iterar)."""

    def __init__(self, valores: list[float]) -> None:
        self._valores = valores

    def reshape(self, dims: int) -> _IterableFake:
        assert dims == -1, f"reshape esperado -1, recibido {dims}"
        return _IterableFake(self._valores)


class _TensorFake:
    """Tensor estilo torch: detach+cpu OBLIGATORIOS antes de aplanar
    (mutantes de `detach`/`cpu`/`and` caen aquí: sin bajar a CPU no hay
    reshape)."""

    def __init__(self, valores: list[float]) -> None:
        self._valores = valores

    def detach(self) -> "_TensorFake":
        return _TensorFake(self._valores)

    def cpu(self) -> _PlanoFake:
        return _PlanoFake(self._valores)


def test_chunk_a_muestras_sin_detach_no_baja_a_cpu() -> None:
    """Objeto con reshape pero SIN detach/cpu (ndarray plano): se usa tal cual
    (mutantes `detach`→`XXdetachXX` y `and`→`or` caen aquí)."""
    plano = _PlanoFake([1.0, 2.0])
    assert _chunk_a_muestras(plano) == [1.0, 2.0]


def test_chunk_a_muestras_detach_sin_cpu_no_baja() -> None:
    """Objeto con detach pero SIN cpu: el `and` NO entra (no es tensor);
    con `or` entraría y reventaría en `.cpu()` (mutante `and`→`or`)."""

    class _ConDetachSinCpu:
        def detach(self) -> "_ConDetachSinCpu":
            return self

        def reshape(self, dims: int) -> _IterableFake:
            assert dims == -1
            return _IterableFake([7.0])

    assert _chunk_a_muestras(_ConDetachSinCpu()) == [7.0]


def test_chunk_a_muestras_con_tensor_estilo_torch() -> None:
    """Tensor torch (detach+cpu): baja a CPU, aplana con reshape(-1) y
    convierte a float (el tensor NO es iterable directo: los mutantes de
    `detach`/`cpu` no pueden llegar a las muestras)."""
    assert _chunk_a_muestras(_TensorFake([3.0, 4.0])) == [3.0, 4.0]
