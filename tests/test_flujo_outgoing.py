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
"""

from collections.abc import Callable

import pytest

from traductor.flujo.outgoing import (
    NIVEL_CLONADO,
    NIVEL_SUBTITULOS,
    NIVEL_VOZ_GENERICA,
    FlujoOutgoing,
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
    worker._proceso = None  # fuerzo el camino de reinicio

    def iniciar_que_falla() -> None:
        raise RuntimeError("el worker TTS no respondió al calentamiento (120 s)")

    worker.iniciar = iniciar_que_falla  # type: ignore[method-assign]
    assert worker.sintetizar("hello") is None


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
