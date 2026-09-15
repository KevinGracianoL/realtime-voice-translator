"""Tests del flujo incoming_en_to_es (ADR-015) — core puro con fakes.

El core no toca hardware: las etapas (traducción EN→ES, teleprompter) se
inyectan como callables/protocols y los tests usan fakes. Las reglas del
ADR-015 que se prueban aquí:
- los PARCIALES solo llegan a pantalla (nunca a traducción);
- la cola es de tamaño 1: una traducción atrasada se descarta;
- cancelación: una solicitud nueva cancela la antigua (el texto de un turno
  superado no se muestra);
- timestamps por etapa (cierre exacto, RegistroEtapas);
- arranque: la validación offline de la traducción en→es es bloqueante.
"""

from collections.abc import Callable

import pytest

from traductor.flujo.incoming import (
    FlujoIncoming,
    validar_arranque_en_es,
)
from traductor.tts.modelos import Salud


class _TeleprompterFake:
    def __init__(self) -> None:
        self.finales: list[tuple[str, str]] = []
        self.parciales: list[str] = []

    def mostrar(self, es: str, en: str) -> None:
        self.finales.append((es, en))

    def parcial(self, texto: str) -> None:
        self.parciales.append(texto)


def _flujo(
    *,
    al_traducir: Callable[[], None] | None = None,
) -> tuple[FlujoIncoming, _TeleprompterFake]:
    teleprompter = _TeleprompterFake()

    def traducir(texto_en: str) -> str:
        if al_traducir is not None:
            al_traducir()
        return f"ES({texto_en})"

    flujo = FlujoIncoming(traducir=traducir, teleprompter=teleprompter)
    return flujo, teleprompter


def test_segmento_final_muestra_es_y_en() -> None:
    flujo, teleprompter = _flujo()
    assert flujo.segmento_final("hello world") is True
    assert teleprompter.finales == [("ES(hello world)", "hello world")]


def test_segmentos_consecutivos_se_muestran_en_orden() -> None:
    """Dos segmentos seguidos se procesan en orden y sin acumular (cola = 1)."""
    flujo, teleprompter = _flujo()
    assert flujo.segmento_final("first") is True
    assert flujo.segmento_final("second") is True
    assert [f[1] for f in teleprompter.finales] == ["first", "second"]


def test_parcial_solo_a_pantalla() -> None:
    """Los parciales del ASR solo van a pantalla, NUNCA a traducción."""
    teleprompter = _TeleprompterFake()
    flujo = FlujoIncoming(
        traducir=lambda es: (_ for _ in ()).throw(AssertionError("no se traduce un parcial")),
        teleprompter=teleprompter,
    )
    flujo.parcial("hello")
    assert teleprompter.parciales == ["hello"]
    assert teleprompter.finales == []


def test_cancelacion_durante_traduccion_no_muestra() -> None:
    """Si la cancelación ocurre durante la TRADUCCIÓN (nuevo segmento del
    entrevistador a mitad de la traducción), el texto del turno superado no
    llega a pantalla."""
    teleprompter = _TeleprompterFake()
    flujo = FlujoIncoming(
        traducir=lambda en: (_ for _ in ()).throw(AssertionError("no se debe traducir")),
        teleprompter=teleprompter,
    )

    # el fake cancela durante la traducción
    def traducir_cancelando(en: str) -> str:
        flujo.cancelar_turno_activo()
        return f"ES({en})"

    flujo.traducir = traducir_cancelando
    assert flujo.segmento_final("hello") is False
    assert teleprompter.finales == []


def test_cancelacion_antes_de_mostrar_descarta_el_turno() -> None:
    """Una solicitud nueva CANCELA la antigua: si la cancelación llega tras la
    traducción pero antes de mostrar, el texto superado no se muestra."""
    teleprompter = _TeleprompterFake()
    flujo = _flujo()[0]

    def traducir_cancelando(en: str) -> str:
        flujo.cancelar_turno_activo()
        return f"ES({en})"

    flujo.traducir = traducir_cancelando
    flujo.teleprompter = teleprompter
    assert flujo.segmento_final("hello") is False
    assert teleprompter.finales == []


def test_timestamps_por_etapa_cierre_exacto() -> None:
    flujo, _teleprompter = _flujo()
    flujo.segmento_final("hello")
    etapas = flujo.ultimo_turno_etapas
    assert set(etapas) == {"traduccion", "pantalla"}
    assert sum(etapas.values()) == pytest.approx(flujo.ultimo_turno_total_ms)


def test_validar_arranque_pasa_con_traduccion_sana() -> None:
    salud = validar_arranque_en_es(
        lambda en: "mi experiencia más fuerte es con sistemas distribuidos"
    )
    assert salud == Salud(disponible=True, detalle="")


def test_validar_arranque_bloquea_traduccion_rota() -> None:
    """El arranque es BLOQUEANTE: una traducción que produce basura impide
    arrancar el flujo — no es una nota."""
    salud = validar_arranque_en_es(lambda en: "sistemasistemasistemasistematizado")
    assert salud.disponible is False
    assert "en→es" in salud.detalle or "traducción" in salud.detalle


def test_validar_arranque_bloquea_excepcion() -> None:
    def rota(_en: str) -> str:
        raise RuntimeError("no se pudo cargar el modelo")

    salud = validar_arranque_en_es(rota)
    assert salud.disponible is False
    assert salud.detalle == "traducción no disponible al arrancar: no se pudo cargar el modelo"


def test_validar_arranque_usa_el_probe_fijo() -> None:
    """El probe es la frase EN fija del arranque (mutante probe→None: lo caza)."""
    probes: list[str] = []

    def registrar(en: str) -> str:
        probes.append(en)
        return "mi experiencia más fuerte es con sistemas distribuidos"

    assert validar_arranque_en_es(registrar).disponible is True
    assert probes == ["My strongest experience is with distributed systems."]


def test_validar_arranque_requiere_la_frase_clave() -> None:
    """Una salida corta pero SIN la frase clave = traducción sospechosa."""
    salud = validar_arranque_en_es(lambda en: "hola mundo este es un texto corto")
    assert salud.disponible is False


def test_validar_arranque_longitud_exacta_80_pasa() -> None:
    """El límite de longitud es estricto: 80 chars con la frase clave PASA
    (el mutante `>= 80` daría FALLA y este test lo caza)."""
    salida = "sistemas distribuidos " + "a" * (80 - len("sistemas distribuidos "))
    assert len(salida) == 80
    assert validar_arranque_en_es(lambda en: salida).disponible is True


def test_validar_arranque_longitud_81_falla() -> None:
    salida = "sistemas distribuidos " + "a" * (81 - len("sistemas distribuidos "))
    assert len(salida) == 81
    assert validar_arranque_en_es(lambda en: salida).disponible is False


def test_validar_arranque_vacio_falla_sin_reventar() -> None:
    assert validar_arranque_en_es(lambda en: "").disponible is False


def test_validar_arranque_repetida_3_veces_pasa_4_no() -> None:
    """La repetición patológica es estricta: la misma palabra 3 veces PASA,
    4 veces FALLA (el mutante `>= 3` lo caza)."""
    base = "mi experiencia más fuerte es con sistemas distribuidos"
    assert validar_arranque_en_es(lambda en: base).disponible is True
    repetida = ("palabra " * 3) + "sistemas distribuidos"
    assert validar_arranque_en_es(lambda en: repetida).disponible is True
    patologica = ("palabra " * 4) + "sistemas distribuidos"
    assert validar_arranque_en_es(lambda en: patologica).disponible is False
