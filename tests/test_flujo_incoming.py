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
from typing import Any

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


def _vad(
    *, chunk_s: float = 0.5, silencio: float = 1.0, max_s: float = 12.0
) -> Callable[[list[float]], list[tuple[int, tuple[Any, ...]]]]:
    """Helper: AsrCable mínimo con la máquina de estados pura."""
    import queue

    from traductor.flujo.adaptadores import procesar_chunk_vad

    cola: queue.Queue[tuple[int, tuple[Any, ...]]] = queue.Queue()
    estado: dict[str, Any] = {}
    numero_chunk = 0

    def alimentar(rms_por_chunk: list[float]) -> list[tuple[int, tuple[Any, ...]]]:
        nonlocal numero_chunk
        for rms in rms_por_chunk:
            procesar_chunk_vad(
                estado,
                f"chunk-{numero_chunk}",
                rms,
                chunk_s=chunk_s,
                umbral_actividad=300.0,
                silencio_cierre_s=silencio,
                fragmento_max_s=max_s,
                cola=cola,
            )
            numero_chunk += 1
        turnos: list[tuple[int, tuple[Any, ...]]] = []
        while not cola.empty():
            turnos.append(cola.get())
        return turnos

    return alimentar


def test_vad_encola_turno_al_cerrar_el_silencio() -> None:
    """El turno se encola con SU número al cerrar el silencio: voz (RMS alto)
    + silencio > 1 s → un turno con los chunks correctos (el cierre ocurre
    cuando el silencio acumulado SUPERA 1.0 s: tercer chunk de silencio)."""
    alimentar = _vad()
    turnos = alimentar([500.0, 400.0, 10.0, 5.0, 3.0, 2.0])
    assert len(turnos) == 1
    numero, fragmento = turnos[0]
    assert numero == 0  # FIFO: el contador arranca en 0
    assert fragmento == ("chunk-0", "chunk-1", "chunk-2", "chunk-3", "chunk-4")


def test_vad_ignora_silencio_inicial() -> None:
    """Silencio antes de la primera voz: NO se encola nada."""
    alimentar = _vad()
    assert alimentar([5.0, 3.0, 500.0, 4.0]) == []


def test_vad_dos_turnos_en_orden_fifo() -> None:
    """Dos turnos consecutivos se encolan con números ASCENDENTES (el orden
    de pantalla es el de la entrada — revisión del PR #26). El silencio debe
    SUPERAR 1.0 s: tres chunks de 0.5 s (0.5, 1.0, 1.5)."""
    alimentar = _vad()
    turnos = alimentar([500.0, 3.0, 2.0, 2.0, 400.0, 1.0, 1.0, 1.0])
    assert [(n, len(f)) for n, f in turnos] == [(0, 4), (1, 4)]


def test_vad_corte_por_fragmento_maximo() -> None:
    """Un fragmento que excede `fragmento_max_s` se cierra aunque NO haya
    silencio (voz continua): 0.5*3 = 1.5 s > 1.0 s → corta en el chunk 2."""
    alimentar = _vad(max_s=1.0)
    turnos = alimentar([500.0, 500.0, 500.0, 500.0])
    assert len(turnos) == 1
    assert len(turnos[0][1]) == 3


def test_vad_frontera_umbral_exacto_no_dispara() -> None:
    """RMS EXACTAMENTE en el umbral NO activa voz (mutante `>=` del umbral:
    con 300.0 exacto el mutante activaría habla y el estado cambiaría)."""
    import queue

    from traductor.flujo.adaptadores import procesar_chunk_vad

    cola_u: queue.Queue[tuple[int, tuple[Any, ...]]] = queue.Queue()
    estado_u: dict[str, Any] = {}
    procesar_chunk_vad(
        estado_u,
        "c0",
        300.0,  # umbral exacto
        chunk_s=0.5,
        umbral_actividad=300.0,
        silencio_cierre_s=1.0,
        fragmento_max_s=12.0,
        cola=cola_u,
    )
    assert estado_u["habla"] is False
    assert cola_u.empty()


def test_vad_rama_voz_marca_habla_y_resetea_silencio() -> None:
    """Un chunk con VOZ deja el estado en habla=True, silencio_desde=0.0 y el
    chunk en el fragmento (mutantes de las asignaciones de la rama de voz:
    `habla=False` o `silencio_desde != 0` romperían estos asserts)."""
    import queue

    from traductor.flujo.adaptadores import procesar_chunk_vad

    cola_v: queue.Queue[tuple[int, tuple[Any, ...]]] = queue.Queue()
    estado_v: dict[str, Any] = {}
    procesar_chunk_vad(
        estado_v,
        "c-voz",
        500.0,
        chunk_s=0.5,
        umbral_actividad=300.0,
        silencio_cierre_s=1.0,
        fragmento_max_s=12.0,
        cola=cola_v,
    )
    assert estado_v["habla"] is True
    assert estado_v["silencio_desde"] == 0.0
    assert estado_v["fragmento"] == ["c-voz"]
    assert cola_v.empty()  # una sola voz no cierra


def test_vad_frontera_longitud_rama_silencio() -> None:
    """El cálculo de longitud del cierre en la RAMA DE SILENCIO (`cierre or
    len(fragmento) * chunk_s > fragmento_max_s`): con fragmento de 3 chunks
    (1.5s) y max=1.0 cierra aunque el silencio no alcance el umbral.

    Mutantes: `* chunk_s`→`/` daría 3/0.5=6.0 > 1.0 (cierra igual, invisible),
    y `>`→`>=` es la frontera — se cazan con 2 chunks (1.0s exacto, NO cierra
    con `>`, SÍ con `>=`)."""
    import queue

    from traductor.flujo.adaptadores import procesar_chunk_vad

    cola_l: queue.Queue[tuple[int, tuple[Any, ...]]] = queue.Queue()
    estado_l: dict[str, Any] = {}
    # voz + silencio con fragmento de 2 chunks (1.0s exacto): con `>=` el
    # mutante cerraría; el código correcto (`>`) no (silencio aún < 1.0)
    for rms in (500.0, 2.0):
        procesar_chunk_vad(
            estado_l,
            f"c{rms}",
            rms,
            chunk_s=0.5,
            umbral_actividad=300.0,
            silencio_cierre_s=100.0,  # solo la longitud decide
            fragmento_max_s=1.0,
            cola=cola_l,
        )
    assert cola_l.empty()  # 2 chunks = 1.0s exacto NO corta (`>` estricto)
    procesar_chunk_vad(
        estado_l,
        "c2.0",
        2.0,
        chunk_s=0.5,
        umbral_actividad=300.0,
        silencio_cierre_s=100.0,
        fragmento_max_s=1.0,
        cola=cola_l,
    )
    assert not cola_l.empty()  # 3 chunks = 1.5s > 1.0 SÍ corta


def test_vad_frontera_silencio_exacto_no_cierra() -> None:
    """Silencio acumulado EXACTAMENTE en `silencio_cierre_s` NO cierra aún
    (mutante `>=` del cierre: 1.0 exacto cerraría y el estado cambiaría)."""
    import queue

    from traductor.flujo.adaptadores import procesar_chunk_vad

    cola_s: queue.Queue[tuple[int, tuple[Any, ...]]] = queue.Queue()
    estado_s: dict[str, Any] = {}
    # voz (0.5s) + 2 silencios (1.0s acumulado = exacto): NO debe cerrar
    for rms in (500.0, 3.0, 2.0):
        procesar_chunk_vad(
            estado_s,
            f"c{rms}",
            rms,
            chunk_s=0.5,
            umbral_actividad=300.0,
            silencio_cierre_s=1.0,
            fragmento_max_s=12.0,
            cola=cola_s,
        )
    assert estado_s["habla"] is True
    assert cola_s.empty()
    # el tercer silencio (1.5s) SÍ cierra
    procesar_chunk_vad(
        estado_s,
        "c2.0",
        2.0,
        chunk_s=0.5,
        umbral_actividad=300.0,
        silencio_cierre_s=1.0,
        fragmento_max_s=12.0,
        cola=cola_s,
    )
    assert not cola_s.empty()


def test_vad_frontera_fragmento_max_exacto_no_corta() -> None:
    """Longitud EXACTAMENTE en `fragmento_max_s` NO corta aún (mutante `>=`
    del corte: 2 chunks = 1.0s exacto cortaría)."""
    import queue

    from traductor.flujo.adaptadores import procesar_chunk_vad

    cola_m: queue.Queue[tuple[int, tuple[Any, ...]]] = queue.Queue()
    estado_m: dict[str, Any] = {}
    for rms in (500.0, 500.0):  # 2 chunks = 1.0s exacto, max=1.0
        procesar_chunk_vad(
            estado_m,
            f"c{rms}",
            rms,
            chunk_s=0.5,
            umbral_actividad=300.0,
            silencio_cierre_s=100.0,  # el silencio no interfiere
            fragmento_max_s=1.0,
            cola=cola_m,
        )
    assert cola_m.empty()  # NO corta en el límite exacto
    procesar_chunk_vad(
        estado_m,
        "c500",
        500.0,
        chunk_s=0.5,
        umbral_actividad=300.0,
        silencio_cierre_s=100.0,
        fragmento_max_s=1.0,
        cola=cola_m,
    )
    assert not cola_m.empty()  # 3 chunks = 1.5s > 1.0 SÍ corta


def test_vad_estado_inicial_explicito() -> None:
    """El estado vacío se inicializa completo (mutantes de `estado.get` con
    defaults: el inicializador es el ÚNICO punto de defaults, sin ramas que
    mutar). Un estado con `habla` faltante arranca en silencio."""
    import queue

    from traductor.flujo.adaptadores import procesar_chunk_vad

    cola: queue.Queue[tuple[int, tuple[Any, ...]]] = queue.Queue()
    estado: dict[str, Any] = {}
    # silencio antes de la primera voz: NO encola y el estado queda inicializado
    procesar_chunk_vad(
        estado,
        "c0",
        5.0,
        chunk_s=0.5,
        umbral_actividad=300.0,
        silencio_cierre_s=1.0,
        fragmento_max_s=12.0,
        cola=cola,
    )
    assert estado == {"fragmento": [], "habla": False, "silencio_desde": 0.0}
    assert cola.empty()


def test_vad_cierre_resetea_estado_y_contador_persiste() -> None:
    """Tras un cierre, el estado vuelve a silencio (mutantes de los resets de
    `_cerrar_turno`: fragmento/habla/silencio_desde) y el contador persiste
    (el siguiente turno toma el número siguiente — FIFO)."""
    import queue

    from traductor.flujo.adaptadores import procesar_chunk_vad

    cola: queue.Queue[tuple[int, tuple[Any, ...]]] = queue.Queue()
    estado: dict[str, Any] = {}
    # voz + silencio suficiente → cierra turno 0
    for rms in (500.0, 3.0, 2.0, 2.0):
        procesar_chunk_vad(
            estado,
            f"c{rms}",
            rms,
            chunk_s=0.5,
            umbral_actividad=300.0,
            silencio_cierre_s=1.0,
            fragmento_max_s=12.0,
            cola=cola,
        )
    numero, fragmento = cola.get()
    assert (numero, len(fragmento)) == (0, 4)
    # ESTADO POST-CIERRE: todo reseteado (los mutantes de los resets fallan)
    assert estado == {
        "fragmento": [],
        "habla": False,
        "silencio_desde": 0.0,
        "contador_turnos": 1,
    }
    # segundo turno: el contador persiste (FIFO)
    for rms in (400.0, 1.0, 1.0, 1.0):
        procesar_chunk_vad(
            estado,
            f"c{rms}",
            rms,
            chunk_s=0.5,
            umbral_actividad=300.0,
            silencio_cierre_s=1.0,
            fragmento_max_s=12.0,
            cola=cola,
        )
    numero2, fragmento2 = cola.get()
    assert (numero2, len(fragmento2)) == (1, 4)


def test_alucinacion_no_llega_al_teleprompter() -> None:
    """Bug 'whisper habla por mí': una frase fantasma NO abre turno ni se muestra."""
    flujo, teleprompter = _flujo()
    # frase fantasma típica del modelo tiny sobre silencio/eco del cable
    assert flujo.segmento_final("Thank you for watching this video") is False
    assert teleprompter.finales == []
    # el contador de turnos no avanzó: la siguiente frase REAL es el turno 1
    assert flujo.segmento_final("what is your experience with distributed systems") is True
    assert teleprompter.finales == [
        (
            "ES(what is your experience with distributed systems)",
            "what is your experience with distributed systems",
        )
    ]


def test_habla_real_con_thank_you_dentro_si_se_muestra() -> None:
    """Una intervención real que CONTIENE 'thank you' no se filtra."""
    flujo, teleprompter = _flujo()
    assert flujo.segmento_final("thank you for taking the time to meet me today") is True
    assert len(teleprompter.finales) == 1


def test_asrcable_fragmento_max_default_corto() -> None:
    """El default de fragmento_max_s es 4.0 (no 12.0): whisper tiny no alucina
    con fragmentos cortos. Un default largo revive el bug de la demo (12.5 s →
    'I don't know what you're talking about' x3). Fija el default para que no
    se revierta por accidente."""
    from unittest.mock import MagicMock

    from traductor.flujo.adaptadores import AsrCable

    cable = AsrCable(MagicMock(), indice_cable=0)
    assert cable._fragmento_max_s == 4.0


def test_asrcable_init_almacena_todos_los_atributos() -> None:
    """El constructor guarda TODO lo que recibe (caza los mutantes de
    asignación/borrado del __init__, no solo el atributo nuevo)."""
    from unittest.mock import MagicMock

    from traductor.flujo.adaptadores import AsrCable

    flujo = MagicMock()
    cable = AsrCable(
        flujo,
        indice_cable=5,
        rate_cable=48000,
        chunk_s=0.5,
        umbral_actividad=300.0,
        silencio_cierre_s=1.0,
        fragmento_max_s=4.0,
    )
    assert cable._flujo is flujo
    assert cable._indice_cable == 5
    assert cable._rate_cable == 48000
    assert cable._chunk_s == 0.5
    assert cable._umbral_actividad == 300.0
    assert cable._silencio_cierre_s == 1.0
    assert cable._fragmento_max_s == 4.0
    assert cable._whisper is None


def test_asrcable_init_defaults() -> None:
    """Los DEFAULTS del constructor quedan fijos (caza los mutantes de
    default: rate_cable->48001, chunk_s, umbral, silencio, fragmento)."""
    from unittest.mock import MagicMock

    from traductor.flujo.adaptadores import AsrCable

    cable = AsrCable(MagicMock(), indice_cable=0)
    assert cable._rate_cable == 48000
    assert cable._chunk_s == 0.5
    assert cable._umbral_actividad == 300.0
    assert cable._silencio_cierre_s == 1.0
    assert cable._fragmento_max_s == 4.0
    assert cable._whisper is None
