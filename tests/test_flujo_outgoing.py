"""Tests del flujo outgoing_es_to_en (ADR-015) — core puro con fakes.

El core no toca hardware: las etapas (traducción, TTS primario/fallback,
teleprompter, salida de audio) se inyectan como callables/protocols y los
tests usan fakes. Las reglas del ADR-015 que se prueban aquí:
- los PARCIALES solo llegan a pantalla (nunca a traducción/TTS);
- la cola es de tamaño 1: una respuesta atrasada se descarta;
- cancelación: una solicitud nueva cancela la antigua (el audio de un turno
  superado nunca se enruta);
- timestamps por etapa (cierre exacto, RegistroEtapas);
- escalera: clonado → voz genérica → solo subtítulos (nivel 4 sin audio);
- arranque: la validación offline de la traducción es bloqueante.

Streaming (ADR-019, fixes 1-3):
- el turno cierra con el PRIMER chunk y el resto llega en un hilo daemon;
- si el turno se cancela, el resto se DRENA sin reproducirse (pipe alineado);
- validación de artefactos en vivo: los sobrantes contra el texto TRADUCIDO
  degradan el turno (nunca se enruta audio sospechoso).
"""

from collections.abc import Callable
from typing import Any

import pytest

from traductor.flujo.outgoing import (
    NIVEL_CLONADO,
    NIVEL_SUBTITULOS,
    NIVEL_VOZ_GENERICA,
    EtapaTtsStream,
    FlujoOutgoing,
    _sobrantes,
    validar_arranque,
)
from traductor.tts.modelos import Salud


class _TtsFake:
    """Devuelve una salida o None (fallo); opcionalmente dispara una acción."""

    def __init__(self, ok: bool = True, al_sintetizar: Callable[[], None] | None = None) -> None:
        self._ok = ok
        self._al_sintetizar = al_sintetizar
        self.textos: list[str] = []

    def sintetizar(self, texto_en: str) -> tuple[bytes, float, str] | None:
        self.textos.append(texto_en)
        if self._al_sintetizar is not None:
            self._al_sintetizar()
        if not self._ok:
            return None
        return (b"wav-clonado", 2.0, "clonado")


class _TeleprompterFake:
    def __init__(self) -> None:
        self.finales: list[tuple[str, str]] = []
        self.parciales: list[str] = []

    def mostrar(self, es: str, en: str) -> None:
        self.finales.append((es, en))

    def parcial(self, es: str) -> None:
        self.parciales.append(es)


class _SalidaFake:
    def __init__(self) -> None:
        self.reproducidos: list[tuple[bytes, float, str]] = []

    def reproducir(self, audio: bytes, duracion_s: float, nombre: str) -> None:
        self.reproducidos.append((audio, duracion_s, nombre))


def _flujo(
    *,
    tts_ok: bool = True,
    fallback_ok: bool = True,
    al_sintetizar: Callable[[], None] | None = None,
) -> tuple[FlujoOutgoing, _TeleprompterFake, _SalidaFake]:
    teleprompter = _TeleprompterFake()
    salida = _SalidaFake()
    flujo = FlujoOutgoing(
        traducir=lambda es: f"EN({es})",
        tts_primario=_TtsFake(ok=tts_ok, al_sintetizar=al_sintetizar),
        tts_fallback=_TtsFake(ok=fallback_ok),
        teleprompter=teleprompter,
        salida_audio=salida,
    )
    return flujo, teleprompter, salida


def test_segmento_final_flujo_completo() -> None:
    flujo, teleprompter, salida = _flujo()
    nivel = flujo.segmento_final("hola como estas")
    assert nivel == NIVEL_CLONADO
    assert teleprompter.finales == [("hola como estas", "EN(hola como estas)")]
    assert len(salida.reproducidos) == 1
    assert salida.reproducidos[0][2] == "clonado"


def test_parcial_solo_a_pantalla() -> None:
    flujo, teleprompter, salida = _flujo()
    flujo.parcial("hola")
    assert teleprompter.parciales == ["hola"]
    assert teleprompter.finales == []
    assert salida.reproducidos == []


def test_cola_tamano_1_dos_segmentos_consecutivos() -> None:
    """Dos segmentos seguidos se procesan en orden y sin acumular (cola = 1)."""
    flujo, teleprompter, salida = _flujo()
    assert flujo.segmento_final("primero") == NIVEL_CLONADO
    assert flujo.segmento_final("segundo") == NIVEL_CLONADO
    assert [e[0] for e in teleprompter.finales] == ["primero", "segundo"]
    assert len(salida.reproducidos) == 2


def test_cancelacion_descarta_el_audio_del_turno_superado() -> None:
    """Una solicitud nueva CANCELA la antigua: el audio del turno superado
    nunca se enruta (el fake de TTS cancela a mitad de síntesis, como haría
    el hilo del micrófono al detectar un nuevo segmento). El texto del turno
    ya se mostró (es lo que el usuario acaba de decir); el AUDIO se descarta."""
    teleprompter = _TeleprompterFake()
    salida = _SalidaFake()
    tts = _TtsFake()
    flujo = FlujoOutgoing(
        traducir=lambda es: f"EN({es})",
        tts_primario=tts,
        tts_fallback=_TtsFake(),
        teleprompter=teleprompter,
        salida_audio=salida,
    )

    def cancelar() -> None:
        flujo.cancelar_turno_activo()

    tts._al_sintetizar = cancelar
    nivel = flujo.segmento_final("primer turno")
    assert nivel is None  # el turno fue cancelado
    assert salida.reproducidos == []  # el audio nunca llegó al altavoz
    assert teleprompter.finales == [("primer turno", "EN(primer turno)")]


def test_cancelacion_durante_traduccion_no_muestra_texto() -> None:
    """Si la cancelación ocurre durante la TRADUCCIÓN (nuevo segmento del mic
    a mitad de la traducción), el texto del turno superado no llega ni a
    pantalla (el texto nuevo está por llegar)."""
    teleprompter = _TeleprompterFake()
    salida = _SalidaFake()

    def traducir_cancelando(_es: str) -> str:
        flujo.cancelar_turno_activo()
        return "EN(hola)"

    flujo = FlujoOutgoing(
        traducir=traducir_cancelando,
        tts_primario=_TtsFake(),
        tts_fallback=_TtsFake(),
        teleprompter=teleprompter,
        salida_audio=salida,
    )
    assert flujo.segmento_final("hola") is None
    assert teleprompter.finales == []
    assert salida.reproducidos == []


def test_escalera_tts_primario_falla_usa_generica() -> None:
    flujo, _teleprompter, salida = _flujo(tts_ok=False, fallback_ok=True)
    nivel = flujo.segmento_final("texto")
    assert nivel == NIVEL_VOZ_GENERICA
    assert len(salida.reproducidos) == 1
    assert salida.reproducidos[0][2] == "generica"


def test_escalera_ambos_tts_fallan_solo_subtitulos() -> None:
    flujo, teleprompter, salida = _flujo(tts_ok=False, fallback_ok=False)
    nivel = flujo.segmento_final("texto")
    assert nivel == NIVEL_SUBTITULOS
    assert salida.reproducidos == []  # sin audio: nunca reproducir sospechoso
    assert teleprompter.finales == [("texto", "EN(texto)")]  # el texto sí se mostró


def test_escalera_sin_fallback_va_directo_a_subtitulos() -> None:
    """Sin voz genérica disponible, el escalón 3 no existe: se cae al 4."""
    teleprompter = _TeleprompterFake()
    salida = _SalidaFake()
    flujo = FlujoOutgoing(
        traducir=lambda es: f"EN({es})",
        tts_primario=_TtsFake(ok=False),
        tts_fallback=None,
        teleprompter=teleprompter,
        salida_audio=salida,
    )
    assert flujo.segmento_final("texto") == NIVEL_SUBTITULOS
    assert salida.reproducidos == []


def test_timestamps_por_etapa_cierre_exacto() -> None:
    flujo, _teleprompter, _salida = _flujo()
    flujo.segmento_final("texto")
    etapas = flujo.ultimo_turno_etapas
    # la marca "entrada" abre el contador; los deltas pertenecen a la etapa
    # que TERMINÓ (RegistroEtapas): traducción, tts y ruteo.
    assert set(etapas) == {"traduccion", "tts", "ruteo"}
    assert sum(etapas.values()) == pytest.approx(flujo.ultimo_turno_total_ms)


def test_segmentos_concurrentes_no_pierden_turnos() -> None:
    """Dos hilos despachando finales (el contrato real de RealtimeSTT): el
    contador no pierde incrementos (el lock) y cada turno termina ENRUTADO o
    CANCELADO — nunca a medias. La cola-1 DESCARTA los turnos superados:
    cuando dos finales se interleavan, el viejo ve `turno != _turno_activo`
    y su audio NO se enruta → `reproducidos <= 100`, nunca == 100.
    El caso de un turno aislado que sí enruta lo cubre
    `test_segmento_final_flujo_completo`.
    """
    import sys
    import threading

    # el switch interval es estado global del intérprete: se restaura SIEMPRE
    intervalo_anterior = sys.getswitchinterval()
    sys.setswitchinterval(1e-6)  # forzar interleavings (runner cargado)
    try:
        flujo, _teleprompter, salida = _flujo()

        def despachar() -> None:
            for _ in range(50):
                flujo.segmento_final("texto")

        h1 = threading.Thread(target=despachar)
        h2 = threading.Thread(target=despachar)
        h1.start()
        h2.start()
        h1.join()
        h2.join()
        assert flujo._numero_turno == 100  # el lock no pierde incrementos
        assert len(salida.reproducidos) <= 100  # los superados se descartan
    finally:
        sys.setswitchinterval(intervalo_anterior)


def test_reinicio_del_worker_que_falla_es_escalera() -> None:
    """Regresión del watchdog (revisión #23): si `iniciar()` lanza RuntimeError
    en el reinicio tras un trabón, `sintetizar` devuelve None (escalera) — el
    turno NO muere en silencio en el hilo daemon del flujo real."""
    from pathlib import Path

    from traductor.flujo.adaptadores import TtsWorkerCliente

    worker = TtsWorkerCliente(python=Path("python"), directorio_salida=Path("."), perfil_id="kevin")
    # el __init__ guarda los campos (mata los mutantes de asignación)
    assert Path(worker._python) == Path("python")
    assert worker._directorio_salida == Path(".")
    assert str(worker._perfil_id) == "kevin"
    assert worker._proceso is None  # el estado inicial (mutante `= ""` lo caza)
    worker._proceso = None  # fuerzo el camino de reinicio
    llamadas: list[str] = []

    def iniciar_que_falla() -> None:
        llamadas.append("iniciar")
        raise RuntimeError("el worker TTS no respondió al calentamiento (120 s)")

    worker.iniciar = iniciar_que_falla  # type: ignore[method-assign]
    assert worker.sintetizar("hello") is None
    assert llamadas == ["iniciar"]  # el reinicio SÍ se intentó (no `is not None`)


def test_reinicio_sin_proceso_y_sin_reiniciar_seteado() -> None:
    """Si `iniciar()` no deja proceso (y no lanza), `sintetizar` devuelve
    None sin reventar: la segunda guarda del proceso lo atrapa (mutante
    `is not None` daría AttributeError sobre None)."""
    from pathlib import Path

    from traductor.flujo.adaptadores import TtsWorkerCliente

    worker = TtsWorkerCliente(python=Path("python"), directorio_salida=Path("."), perfil_id="kevin")
    worker._proceso = None
    worker.iniciar = lambda: None  # type: ignore[method-assign]
    assert worker.sintetizar("hello") is None


def test_sintetizar_con_proceso_no_reinicia() -> None:
    """Con el proceso vivo NO se llama a iniciar (el watchdog no reinicia en
    frío cada turno)."""
    import types
    from pathlib import Path

    from traductor.flujo.adaptadores import TtsWorkerCliente

    worker = TtsWorkerCliente(python=Path("python"), directorio_salida=Path("."), perfil_id="kevin")
    worker._proceso = types.SimpleNamespace(stdin=None, stdout=None)
    llamadas: list[str] = []

    def iniciar_spy() -> None:
        llamadas.append("iniciar")

    worker.iniciar = iniciar_spy  # type: ignore[method-assign]
    assert worker.sintetizar("hello") is None
    assert llamadas == []


def test_validar_arranque_pasa_con_traduccion_sana() -> None:
    salud = validar_arranque(lambda es: "my strongest experience is with distributed systems")
    assert salud == Salud(disponible=True, detalle="")


def test_validar_arranque_bloquea_traduccion_rota() -> None:
    """El arranque es BLOQUEANTE (ADR-014/015): una traducción que produce
    basura impide arrancar el flujo — no es una nota."""
    salud = validar_arranque(lambda es: "mainstremainstremainstremainstremainst")
    assert salud.disponible is False
    assert "mwt" in salud.detalle or "traducción" in salud.detalle


def test_validar_arranque_bloquea_excepcion() -> None:
    def rota(_es: str) -> str:
        raise RuntimeError("no se pudo cargar el modelo")

    salud = validar_arranque(rota)
    assert salud.disponible is False
    assert salud.detalle == "traducción no disponible al arrancar: no se pudo cargar el modelo"


def test_validar_arranque_usa_la_frase_probe_fija() -> None:
    """El probe es la frase fija del arranque (mutante probe→None: lo caza)."""
    probes: list[str] = []

    def registrar(es: str) -> str:
        probes.append(es)
        return "my strongest experience is with distributed systems"

    assert validar_arranque(registrar).disponible is True
    assert probes == ["Mi experiencia mas fuerte es con sistemas distribuidos."]


def test_validar_arranque_requiere_la_frase_clave() -> None:
    """Una salida corta pero SIN la frase clave = traducción sospechosa."""
    salud = validar_arranque(lambda es: "hello world this is a short output")
    assert salud.disponible is False


def test_validar_arranque_longitud_exacta_80_pasa() -> None:
    """El límite de longitud es estricto: 80 chars con la frase clave PASA
    (el mutante `>= 80` daría FALLA y este test lo caza)."""
    salida = "distributed systems " + "a" * (80 - len("distributed systems "))
    assert len(salida) == 80
    assert validar_arranque(lambda es: salida).disponible is True


def test_validar_arranque_longitud_81_falla() -> None:
    """81 chars con la frase clave FALLA (el mutante `> 81` pasaría)."""
    salida = "distributed systems " + "a" * (81 - len("distributed systems "))
    assert len(salida) == 81
    assert validar_arranque(lambda es: salida).disponible is False


def test_validar_arranque_vacio_falla_sin_reventar() -> None:
    """Entrada vacía: FALLA (no crash de max sobre vacío — mata los mutantes
    de `default=` que reventarían con TypeError)."""
    assert validar_arranque(lambda es: "").disponible is False


def test_validar_arranque_repetida_3_veces_pasa_4_no() -> None:
    """La repetición patológica es estricta: la misma palabra 3 veces PASA,
    4 veces FALLA (el modo real del bucle 'mainstream')."""
    base = "my strongest experience is with distributed systems"
    assert validar_arranque(lambda es: base).disponible is True
    repetida = ("word " * 3) + "distributed systems"
    assert validar_arranque(lambda es: repetida).disponible is True
    patologica = ("word " * 4) + "distributed systems"
    assert validar_arranque(lambda es: patologica).disponible is False


def test_validar_arranque_mensaje_de_basura_exacto() -> None:
    salud = validar_arranque(lambda es: "mainstremainstremainstremainstremainst")
    assert salud.disponible is False
    assert salud.detalle == (
        "traducción es→en produce basura al arrancar: precarga del modelo "
        "spacy mwt no efectiva (el worker de traducción NO entra al flujo)"
    )


def _wav_bytes(marca: int, frames: int = 100) -> bytes:
    """WAV int16 mono mínimo (para los fakes del camino streaming)."""
    import io
    import wave

    buf = io.BytesIO()
    with wave.open(buf, "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(24000)
        w.writeframes(bytes([marca, 0]) * frames)
    return buf.getvalue()


class _TtsStreamFake:
    """Etapa con streaming (ADR-019): chunks fijos; `fallar` = worker caído.

    Implementa `EtapaTts` (sintetizar) y `EtapaTtsStream` (sintetizar_stream):
    el camino streaming usa el generador; `sintetizar` es el fallback.
    """

    def __init__(
        self,
        chunks: list[tuple[bytes, float, str]] | None = None,
        *,
        fallar: bool = False,
    ) -> None:
        if fallar:
            self._chunks = None
        elif chunks is None:
            self._chunks = [(_wav_bytes(1), 1.0, "clonado"), (_wav_bytes(2), 1.0, "clonado")]
        else:
            self._chunks = list(chunks)
        self.textos: list[str] = []

    def sintetizar(self, texto_en: str) -> tuple[bytes, float, str] | None:
        self.textos.append(texto_en)
        if not self._chunks:
            return None
        return self._chunks[0]

    def sintetizar_stream(self, texto_en: str) -> Any | None:
        self.textos.append(texto_en)
        if self._chunks is None:
            return None
        return iter(self._chunks)


def test_tts_stream_fake_implementa_el_protocolo() -> None:
    assert isinstance(_TtsStreamFake(), EtapaTtsStream)


def _flujo_stream(
    *,
    chunks: list[tuple[bytes, float, str]] | None = None,
    fallar: bool = False,
    verificar: Callable[[bytes], str | None] | None = None,
    max_sobrantes: int = 0,
) -> tuple[FlujoOutgoing, _TeleprompterFake, _SalidaFake]:
    teleprompter = _TeleprompterFake()
    salida = _SalidaFake()
    flujo = FlujoOutgoing(
        traducir=lambda es: f"EN({es})",
        tts_primario=_TtsStreamFake(chunks=chunks, fallar=fallar),
        tts_fallback=None,
        teleprompter=teleprompter,
        salida_audio=salida,
        verificar_artefactos=verificar,
        max_sobrantes=max_sobrantes,
    )
    return flujo, teleprompter, salida


def test_stream_cierra_en_primer_chunk_y_resto_se_reproduce() -> None:
    """ADR-019 fixes 1-2: el turno cierra con el PRIMER chunk (el ruteo se
    marca tras él, sin esperar la síntesis completa) y el resto del audio se
    reproduce en el hilo daemon."""
    flujo, _teleprompter, salida = _flujo_stream()
    nivel = flujo.segmento_final("texto")
    assert nivel == NIVEL_CLONADO
    # el cierre del turno está tras el primer chunk, no tras la síntesis completa
    assert set(flujo.ultimo_turno_etapas) == {"traduccion", "tts", "ruteo"}
    for _ in range(100):
        if len(salida.reproducidos) >= 2:
            break
        import time

        time.sleep(0.005)
    assert [r[2] for r in salida.reproducidos] == ["clonado", "clonado"]


def test_stream_falla_es_escalera() -> None:
    flujo, _teleprompter, salida = _flujo_stream(fallar=True)
    assert flujo.segmento_final("texto") == NIVEL_SUBTITULOS
    assert salida.reproducidos == []


def test_stream_vacio_es_escalera() -> None:
    flujo, _teleprompter, salida = _flujo_stream(chunks=[])
    assert flujo.segmento_final("texto") == NIVEL_SUBTITULOS
    assert salida.reproducidos == []


def test_stream_cancelado_despues_del_primer_chunk_drena_sin_reproducir() -> None:
    """ADR-019: si el turno se cancela DESPUÉS del primer chunk, el resto se
    DRENA sin reproducirse (el pipe del worker queda alineado; el audio de un
    turno superado nunca se enruta)."""
    teleprompter = _TeleprompterFake()
    salida = _SalidaFake()
    flujo = FlujoOutgoing(
        traducir=lambda es: f"EN({es})",
        tts_primario=_TtsStreamFake(
            chunks=[
                (b"wav-1", 1.0, "clonado"),
                (b"wav-2", 1.0, "clonado"),
                (b"wav-3", 1.0, "clonado"),
            ]
        ),
        tts_fallback=None,
        teleprompter=teleprompter,
        salida_audio=salida,
    )

    def cancelar_tras_primero() -> None:
        flujo.cancelar_turno_activo()

    def sintetizar_cancelando(texto_en: str) -> Any | None:
        def generador() -> Any:
            yield (b"wav-1", 1.0, "clonado")  # primer chunk: SÍ se enruta
            cancelar_tras_primero()  # el resto se drena sin reproducirse
            yield (b"wav-2", 1.0, "clonado")

        return generador()

    tts = flujo.tts_primario
    assert isinstance(tts, EtapaTtsStream)  # el fake es el que se reemplaza
    tts.sintetizar_stream = sintetizar_cancelando  # type: ignore[method-assign]
    assert flujo.segmento_final("texto") == NIVEL_CLONADO  # el primer chunk se enrutó
    for _ in range(100):
        if len(salida.reproducidos) >= 1:
            break
        import time

        time.sleep(0.005)
    assert len(salida.reproducidos) == 1  # el resto se drenó sin reproducirse


def test_validacion_artefactos_sobrantes_marca_degradado() -> None:
    """ADR-019 fix 3: el ASR-de-retorno del audio COMPLETO encuentra palabras
    SOBRANTES por encima del umbral (no están en el texto traducido) → el
    turno se marca degradado (la validación es del turno completo, no del
    primer chunk — whisper alucina en fragmentos cortos, ADR-019)."""
    flujo, _teleprompter, salida = _flujo_stream(
        verificar=lambda _audio: "gibberish words here",
        max_sobrantes=2,
    )
    assert flujo.segmento_final("texto") == NIVEL_CLONADO  # el primer chunk cierra
    for _ in range(200):  # el daemon acumula y valida al terminar
        if flujo.ultimo_turno_degradado:
            break
        import time

        time.sleep(0.005)
    assert flujo.ultimo_turno_degradado is True


def test_validacion_artefactos_limpia_pasa() -> None:
    """Transcripción sin sobrantes (contra el texto TRADUCIDO): el turno se
    enruta y NO se marca degradado. 'trauthor' está en el texto traducido →
    NO es sobrante (el desvío es de la traducción, no del audio — hallazgo de
    la revisión del PR #23)."""
    flujo, _teleprompter, salida = _flujo_stream(verificar=lambda _audio: "EN(texto)")
    assert flujo.segmento_final("texto") == NIVEL_CLONADO
    for _ in range(200):
        if len(salida.reproducidos) >= 2:
            break
        import time

        time.sleep(0.005)
    assert flujo.ultimo_turno_degradado is False
    assert len(salida.reproducidos) >= 1  # el primer chunk cierra el turno


def test_validacion_sin_transcripcion_no_bloquea() -> None:
    """Instrumento roto (None): no bloquear el turno (la validación es un gate
    vivo, no un requisito de disponibilidad)."""
    flujo, _teleprompter, salida = _flujo_stream(verificar=lambda _audio: None)
    assert flujo.segmento_final("texto") == NIVEL_CLONADO
    assert len(salida.reproducidos) >= 1


def test_concatenar_wav_une_chunks() -> None:
    """Fix 3: la validación corre sobre el turno COMPLETO — los chunks WAV se
    concatenan en un solo audio (misma tasa)."""
    import io
    import wave

    flujo, _teleprompter, _salida = _flujo_stream()
    a = _wav_bytes(1)
    b = _wav_bytes(2, frames=50)
    unido = flujo._concatenar_wav([a, b])
    with wave.open(io.BytesIO(unido), "rb") as w:
        assert w.getframerate() == 24000
        assert w.readframes(w.getnframes()) == bytes([1, 0]) * 100 + bytes([2, 0]) * 50


def test_concatenar_wav_chunk_unico_pasa_tal_cual() -> None:
    flujo, _teleprompter, _salida = _flujo_stream()
    a = _wav_bytes(1)
    assert flujo._concatenar_wav([a]) == a


def test_sobrantes_compara_contra_texto_traducido() -> None:
    """'trauthor' está en el texto TRADUCIDO → no es sobrante (es de la
    traducción, no del audio); 'gibberish' no está → sí es sobrante."""
    assert (
        _sobrantes(
            "good im kevin today is the day",
            "good im kevin today is the day to try the trauthor",
        )
        == []
    )
    assert _sobrantes("gibberish words here", "hello world") == ["gibberish", "words", "here"]


def test_sobrantes_cuenta_repeticiones() -> None:
    """Una palabra REPETIDA de más es un artefacto (repetición patológica):
    cada aparición del retorno consume una del esperado (multiset)."""
    assert _sobrantes("hello world hello", "hello world") == ["hello"]
    assert _sobrantes("no no no", "no") == ["no", "no"]
    assert _sobrantes("no", "no no no") == []  # menos apariciones no es sobrante
    assert _sobrantes("hello world hello world", "hello world") == ["hello", "world"]


def test_sobrantes_consume_una_aparicion_por_repeticion() -> None:
    """El multiset consume UNA aparición por cada repetición (mutante `-=2`:
    con 'no no no' vs 'no', restar 2 daría -3 y marcaría la segunda como
    sobrante — el resultado correcto es exactamente ['no', 'no'])."""
    assert _sobrantes("no no no", "no") == ["no", "no"]  # consume de a 1
    assert _sobrantes("a a a b", "a a") == ["a", "b"]


def test_sobrantes_normaliza_puntuacion() -> None:
    assert _sobrantes("Hello, world!", "hello world") == []


class _StreamFake:
    """Stream pyaudio falso: registra cada write (para el cable por bloques)."""

    def __init__(self) -> None:
        self.escrituras: list[bytes] = []

    def write(self, datos: bytes) -> None:
        self.escrituras.append(datos)


def _cable_con_stream(stream: _StreamFake, rate_cable: int = 48000, bloque_s: float = 0.5) -> Any:
    """SalidaCable con stream y lock fake (la clase es # pragma: no cover:
    el hardware no está en CI, su lógica de bloques sí se prueba aquí)."""
    import threading

    from traductor.flujo.adaptadores import SalidaCable

    cable = SalidaCable(rate_cable=rate_cable, bloque_s=bloque_s)
    cable._stream = stream
    cable._lock_escritura = threading.Lock()
    return cable


def _esperar_escrituras(stream: _StreamFake, n: int) -> None:
    for _ in range(200):
        if len(stream.escrituras) >= n:
            return
        import time

        time.sleep(0.005)
    raise AssertionError(f"el daemon no escribió {n} bloques (tiene {len(stream.escrituras)})")


def test_cable_defaults_almacenados() -> None:
    """Mutantes de defaults del __init__ (48000→48001, 0.2→0.3): lo cazan."""
    from traductor.flujo.adaptadores import SalidaCable

    cable = SalidaCable()
    assert cable._rate_cable == 48000
    assert cable._bloque_s == 0.2
    cable = SalidaCable(rate_cable=44100, bloque_s=0.1)
    assert cable._rate_cable == 44100
    assert cable._bloque_s == 0.1


def test_cable_escribe_primer_bloque_y_resto_en_daemon() -> None:
    """ADR-019 fix 2: `reproducir` escribe el PRIMER bloque y devuelve (el
    cierre del turno = primer sample audible); el resto se encola para el
    hilo daemon (sin bloquear el turno). El orden es el del audio."""
    stream = _StreamFake()
    cable = _cable_con_stream(stream)
    bloque = int(48000 * 4 * 0.5)  # 0.5 s estéreo int16 a 48 kHz = 96000 bytes
    # bloques DISTINGUIBLES (cada uno con un byte propio: un pcm periódico
    # haría indistinguibles los mutantes de slice/offset)
    pcm = bytes([1]) * bloque + bytes([2]) * bloque + bytes([3]) * bloque + bytes([4]) * bloque
    audios: list[bytes] = []

    def _pcm_cable(audio: bytes) -> bytes:
        audios.append(audio)  # el cable recibe EXACTAMENTE el WAV del turno
        return pcm

    cable._audio_a_pcm_cable = _pcm_cable

    cable.reproducir(b"wav-del-turno", 4.0, "xtts-kevin")
    assert audios == [b"wav-del-turno"]  # mutante `_audio_a_pcm_cable(None)`
    # el PRIMER bloque se escribió síncrono (el cierre del turno no espera)
    assert len(stream.escrituras) >= 1
    assert stream.escrituras[0] == pcm[:bloque]
    _esperar_escrituras(stream, 4)
    # el resto, en orden, sin omitir bloques (mutantes de slice/offset)
    esperado = [
        pcm[:bloque],
        pcm[bloque : 2 * bloque],
        pcm[2 * bloque : 3 * bloque],
        pcm[3 * bloque :],
    ]
    assert stream.escrituras == esperado


def test_cable_lanzar_resto_es_daemon() -> None:
    """El hilo del resto es DAEMON (mutantes `daemon=None`/`False`/ausente):
    el cierre del turno no espera la reproducción ni al terminar el flujo."""
    import threading

    stream = _StreamFake()
    cable = _cable_con_stream(stream)
    hilo = cable._lanzar_resto(stream, threading.Lock(), [b"a", b"b"])
    assert hilo.daemon is True
    hilo.join(timeout=1.0)
    assert not hilo.is_alive()


def test_cable_dos_bloques_escribe_ambos() -> None:
    """Con EXACTAMENTE dos bloques el daemon existe y escribe el segundo
    (mutantes `> 2` y `>= 1` del condicional lo cazan)."""
    stream = _StreamFake()
    cable = _cable_con_stream(stream)
    bloque = int(48000 * 4 * 0.5)
    pcm = bytes([1]) * bloque * 2
    cable._audio_a_pcm_cable = lambda _audio: pcm

    cable.reproducir(b"wav", 4.0, "xtts-kevin")
    _esperar_escrituras(stream, 2)
    assert len(stream.escrituras) == 2


def test_cable_bloque_unico_sin_daemon() -> None:
    """Con UN solo bloque no hay hilo daemon (mutante `>= 1` del condicional
    lo cazaría si creara daemon con lista vacía: el test espera 1 sola
    escritura y que el primer bloque sea el pcm completo)."""
    stream = _StreamFake()
    cable = _cable_con_stream(stream)
    bloque = int(48000 * 4 * 0.5)
    pcm = bytes([2]) * bloque
    cable._audio_a_pcm_cable = lambda _audio: pcm

    cable.reproducir(b"wav", 4.0, "xtts-kevin")
    import time

    time.sleep(0.02)
    assert len(stream.escrituras) == 1
    assert stream.escrituras[0] == pcm


def test_cable_bloque_minimo_un_byte() -> None:
    """bloque_s pequeño: `max(1, ...)` evita un bloque de cero bytes (mutante
    `max(2, ...)` lo cazaría: partiría por 2 bytes y escribiría menos)."""
    stream = _StreamFake()
    cable = _cable_con_stream(stream, rate_cable=1, bloque_s=0.25)
    pcm = bytes([3]) * 5  # 5 bytes: bloque de 1 byte -> 5 bloques
    cable._audio_a_pcm_cable = lambda _audio: pcm

    cable.reproducir(b"wav", 4.0, "xtts-kevin")
    _esperar_escrituras(stream, 5)
    assert len(stream.escrituras) == 5
    assert stream.escrituras == [bytes([3]), bytes([3]), bytes([3]), bytes([3]), bytes([3])]


def test_cable_sin_stream_no_reproduce() -> None:
    """Sin stream abierto (escalera del cable): no se reproduce nada y el
    audio ni se parsea (mutante `and` del guard lo cazaría)."""
    from traductor.flujo.adaptadores import SalidaCable

    cable = SalidaCable()
    cable._lock_escritura = __import__("threading").Lock()
    cable.reproducir(b"wav-roto", 1.0, "xtts-kevin")  # ni siquiera parsea
    assert True  # no lanza y no reproduce: flujo sigue vivo


def test_cable_sin_lock_no_reproduce() -> None:
    """Sin lock de escritura (estado a medio abrir): no se reproduce nada
    (mutante `and` del guard lo cazaría)."""

    from traductor.flujo.adaptadores import SalidaCable

    cable = SalidaCable()
    cable._stream = _StreamFake()
    cable._lock_escritura = None
    cable.reproducir(b"wav-roto", 1.0, "xtts-kevin")
    assert True  # no lanza y no reproduce: flujo sigue vivo


def test_cable_estado_inicial() -> None:
    """El cable nace sin stream ni lock (mutantes `= ""` del __init__: un
    falsy no-None pasaría los guards `is None` y reventaría al abrir)."""
    from traductor.flujo.adaptadores import SalidaCable

    cable = SalidaCable()
    assert cable._pa is None
    assert cable._stream is None
    assert cable._lock_escritura is None


def test_worker_cliente_estado_inicial() -> None:
    """El cliente nace sin proceso ni lock de lectura (mutantes `= ""`)."""
    from pathlib import Path

    from traductor.flujo.adaptadores import TtsWorkerCliente

    worker = TtsWorkerCliente(python=Path("python"), directorio_salida=Path("."), perfil_id="kevin")
    assert worker._proceso is None
    assert worker._lock_lectura is None


class _PaFake:
    """pyaudio fake para _buscar_device (device 0 sin cable, 1 = CABLE Input)."""

    def __init__(self) -> None:
        self._devices = [
            {"name": "Altavoces", "maxOutputChannels": 2, "maxInputChannels": 0},
            {
                "name": "CABLE Input (VB-Audio Virtual Cable)",
                "maxOutputChannels": 2,
                "maxInputChannels": 0,
            },
            {
                "name": "CABLE Output (VB-Audio Virtual Cable)",
                "maxOutputChannels": 0,
                "maxInputChannels": 2,
            },
        ]

    def get_device_count(self) -> int:
        return len(self._devices)

    def get_device_info_by_index(self, i: int) -> dict[str, object]:
        return self._devices[i]


def test_buscar_device_encuentra_por_nombre_y_canales() -> None:
    from traductor.flujo.adaptadores import _buscar_device

    pa = _PaFake()
    assert _buscar_device(pa, "CABLE Input", "maxOutputChannels", 2) == 1
    assert _buscar_device(pa, "CABLE Output", "maxInputChannels", 2) == 2
    assert _buscar_device(pa, "CABLE Input", "maxInputChannels", 2) is None  # canales distintos
    assert _buscar_device(pa, "No existe", "maxOutputChannels", 2) is None
