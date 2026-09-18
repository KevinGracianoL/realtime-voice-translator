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
    # cada signo fuerte parte EN MEDIO (no solo al final: la cola tambien se
    # agrega, asi que al final un signo mutado no cambiaria el resultado)
    assert _frases("Hola! Mundo.") == ["Hola!", "Mundo."]
    assert _frases("¿Que? Si.") == ["¿Que?", "Si."]
    assert _frases("Bueno… seguimos") == ["Bueno…", "seguimos"]
    assert _frases("Uno; dos.") == ["Uno;", "dos."]
    # tokens de solo cierres o solo puntuacion: no rompen el recorrido
    assert _frases("Uno » Dos.") == ["Uno » Dos."]
    assert _frases("Hola ... Seguimos.") == ["Hola ...", "Seguimos."]


def test_frases_no_parte_abreviaturas_ni_decimales() -> None:
    """El punto tras abreviatura, sigla o decimal NO cierra frase (review PR
    #34: partir `'Mr. Smith…'` o `'The U.S. team…'` sonaba a fin de oracion
    en medio, y `'3.5'` ya quedaba bien porque no termina en punto)."""
    from traductor.tts.backend_xtts import _frases

    assert _frases("I worked there for 3.5 years. The U.S. team was great.") == [
        "I worked there for 3.5 years.",
        "The U.S. team was great.",
    ]
    assert _frases("Mr. Smith led the project. Version 2.0 shipped.") == [
        "Mr. Smith led the project.",
        "Version 2.0 shipped.",
    ]
    # sigla seguida de MAYUSCULA: la frena la lista de siglas, no el guard de
    # minuscula (sin esto, `U.S.` partiria en `'The U.S.'` + `'Team…'`)
    assert _frases("The U.S. Team arrived. We left.") == [
        "The U.S. Team arrived.",
        "We left.",
    ]
    assert _frases("The a.m. flight left. We arrived.") == [
        "The a.m. flight left.",
        "We arrived.",
    ]
    # abreviatura en la PENULTIMA posicion (indice + 1 tocando el borde del
    # guard): debe evaluarse igual y no partir
    assert _frases("I like Mr. Smith.") == ["I like Mr. Smith."]


def test_frases_cierra_tras_comilla_de_cierre() -> None:
    """`'He said \"stop.\" Then he left.'` parte: el punto esta antes de la
    comilla de cierre y el token siguiente sigue en mayuscula (review PR #34:
    el camino viejo no partia aqui)."""
    from traductor.tts.backend_xtts import _frases

    assert _frases('He said "stop." Then he left.') == ['He said "stop."', "Then he left."]


def test_es_abreviatura_casos_del_contrato() -> None:
    """Casos quirurgicos de la deteccion (cazan mutantes del review): la
    comilla de cierre no estorba, la `X` final no se recorta y los limites
    exactos de las siglas son partes de 1-2 letras."""
    from traductor.tts.backend_xtts import _es_abreviatura

    assert _es_abreviatura('"Mr."') is True
    assert _es_abreviatura("Ph.D") is True  # parte de 2 letras: sigla
    assert _es_abreviatura("Dr.Who") is False  # parte de 3: no es sigla
    assert _es_abreviatura("MrX") is False  # la X final no se recorta


def test_frases_tokens_raros_no_parten_de_mas() -> None:
    """Variantes que el review destapo: una palabra con `X` final no es
    puntuacion fuerte (`XXX`), la comilla de APERTURA del token siguiente no
    confunde el chequeo de mayuscula (`Xbox`, `"Then"`) y un punto seguido de
    minuscula no cierra (caza el mutante que ignora el token siguiente)."""
    from traductor.tts.backend_xtts import _frases

    assert _frases("Marcas XXX y YYY. Fin.") == ["Marcas XXX y YYY.", "Fin."]
    assert _frases('He said "stop." "Then" he left.') == ['He said "stop."', '"Then" he left.']
    assert _frases('He said "stop." Xbox is here.') == ['He said "stop."', "Xbox is here."]
    assert _frases("Hola mundo. seguimos igual.") == ["Hola mundo. seguimos igual."]


def test_limitar_parte_por_espacios_guiones_y_palabras_largas() -> None:
    """Sin tope, un parrafo sin puntuar seria UN chunk gigante (review): los
    tramos que superan el tope se parten con `textwrap` (stdlib), sin perder
    texto. Los literales 180/181 fijan el tope aunque la constante mute."""
    from traductor.tts.backend_xtts import _MAX_CARACTERES_FRASE, _frases, _limitar

    assert _limitar([]) == []
    assert _limitar(["corta"]) == ["corta"]

    # el tope exacto no parte; un caracter mas, si (literales fijos)
    assert _limitar(["w" * 180]) == ["w" * 180]
    assert _limitar(["a" * 181]) == ["a" * 180, "a"]
    assert _limitar(["y" * _MAX_CARACTERES_FRASE]) == ["y" * _MAX_CARACTERES_FRASE]

    # palabra con guion: textwrap rompe en el guion (default) antes que a lo bruto
    assert _limitar(["x" * 100 + "-" + "y" * 100]) == ["x" * 100 + "-", "y" * 100]

    # corte por espacios: todos los trozos cabe en el tope y nada se pierde
    palabras = ("palabra " * 40).strip()
    partes = _limitar([palabras])
    assert all(len(parte) <= _MAX_CARACTERES_FRASE for parte in partes)
    assert " ".join(partes) == palabras

    # via _frases: frase > tope sin puntuar sale partida en trozos acotados
    partes_frase = _frases("z" * 400)
    assert all(len(parte) <= _MAX_CARACTERES_FRASE for parte in partes_frase)
    assert "".join(partes_frase) == "z" * 400


def test_sintetizar_stream_por_frase_con_latentes_cacheadas() -> None:
    """`sintetizar_stream` emite UN AudioResult por frase via `inference` con
    las latentes del perfil CACHEADAS (una vez por perfil, ADR-011/014): el
    review demostro que `tts.tts(speaker_wav=...)` las recalcula en cada
    llamada (756 ms por frase)."""
    backend = BackendXtts()
    inferencias: list[tuple[str, str, object, object]] = []
    ajustes_recibidos: list[dict[str, object]] = []

    class _ConfigFake:
        temperature = 0.75
        length_penalty = 1.0
        repetition_penalty = 5.0
        top_k = 50
        top_p = 0.85

    class _ModeloFake:
        def __init__(self) -> None:
            self.config = _ConfigFake()
            self.latentes_pedidas: list[list[str]] = []

        def get_conditioning_latents(self, audio_path: list[str]) -> tuple[str, str]:
            self.latentes_pedidas.append(list(audio_path))
            return ("gpt-cond", "spk-emb")

        def inference(
            self, texto: str, language: str, gpt: object, spk: object, **ajustes: object
        ) -> dict[str, list[float]]:
            inferencias.append((texto, language, gpt, spk))
            ajustes_recibidos.append(ajustes)
            return {"wav": [0.1] * 240}  # 0.01 s de audio por frase

    class _SynthesizerFake:
        def __init__(self) -> None:
            self.tts_model = _ModeloFake()

    class _TtsFake:
        def __init__(self) -> None:
            self.synthesizer = _SynthesizerFake()

    backend._tts = _TtsFake()
    modelo = backend._tts.synthesizer.tts_model
    salidas = list(backend.sintetizar_stream("Uno. Dos!", PERFIL))
    assert len(salidas) == 2
    assert [c[0] for c in inferencias] == ["Uno.", "Dos!"]
    assert inferencias[0][1] == "en"
    assert inferencias[0][2] == "gpt-cond"  # latentes[0] -> gpt
    assert inferencias[0][3] == "spk-emb"  # latentes[1] -> speaker
    # los settings del config van a inference: con los defaults del metodo el
    # audio cambia (repetition_penalty 10 vs 5 medido, review PR #34)
    assert ajustes_recibidos[0] == {
        "temperature": 0.75,
        "length_penalty": 1.0,
        "repetition_penalty": 5.0,
        "top_k": 50,
        "top_p": 0.85,
    }
    assert all(ajustes == ajustes_recibidos[0] for ajustes in ajustes_recibidos)
    assert modelo.latentes_pedidas == [list(PERFIL.muestras)]  # UNA vez
    for salida in salidas:
        assert salida.formato == "pcm_f32le"
        assert salida.duracion_s == pytest.approx(240 / 24000)

    # segundo texto del MISMO perfil: la cache no recalcula
    list(backend.sintetizar_stream("Tres.", PERFIL))
    assert len(modelo.latentes_pedidas) == 1

    # perfil distinto: recalcula (la clave de cache es el id)
    otro = VoiceProfile(id="otro", nombre="Otro", muestras=("x.wav",))
    list(backend.sintetizar_stream("Cuatro.", otro))
    assert len(modelo.latentes_pedidas) == 2
    assert modelo.latentes_pedidas[1] == ["x.wav"]


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
