"""Flujo incoming_en_to_es (ADR-015): audio remoto → VAD → ASR en → Argos → subtítulos.

El core es PURO y testeable: las etapas se inyectan (traducción EN→ES,
teleprompter) y los adaptadores reales (captura del audio remoto, teleprompter
HTTP) viven en `traductor.flujo.adaptadores` y se cablean en
`scripts/flujo_incoming.py`.

Reglas del ADR-015 implementadas aquí:
- los PARCIALES solo llegan a pantalla (nunca a traducción);
- cola de tamaño 1: una traducción atrasada se descarta;
- cancelación: una solicitud nueva cancela la antigua (el texto de un turno
  superado no se muestra);
- timestamps por etapa con cierre exacto (`RegistroEtapas`);
- validación de arranque BLOQUEANTE: la traducción EN→ES funciona sin red
  antes de aceptar la llamada (precarga del mwt, mismo requisito que el
  outgoing).

El clon dinámico de la voz del entrevistador es OPCIONAL (ADR-015: subtítulos
primero; el clon solo si las muestras pasan los controles). Este flujo
implementa los SUBTÍTULOS — la parte del clon queda como extensión futura.
"""

from __future__ import annotations

import threading
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any, Protocol

from traductor.asr.alucinacion import es_alucinacion
from traductor.tts.harness import RegistroEtapas
from traductor.tts.modelos import Salud

NIVEL_SUBTITULOS = 4  # escalera ADR-015: nivel mínimo, subtítulos en ambos sentidos

_TEXTO_PROBE_EN = "My strongest experience is with distributed systems."
FRASE_CLAVE_EN_A_ES = "sistemas distribuidos"
DETALLE_BASURA_EN_A_ES = (
    "traducción en→es produce basura al arrancar: precarga del modelo "
    "spacy mwt no efectiva (el worker de traducción NO entra al flujo)"
)


class Teleprompter(Protocol):
    def mostrar(self, es: str, en: str) -> None: ...
    def parcial(self, texto: str) -> None: ...


@dataclass
class FlujoIncoming:
    """Orquesta el subtitulado: segmento final EN → traducción EN→ES → pantalla.

    Contrato de hilos (mismo que el outgoing): RealtimeSTT entrega parciales y
    finales desde SUS propios hilos. El core es seguro bajo concurrencia:
    - el contador de turnos y el turno activo se mutan bajo `_lock`;
    - la cancelación y el arranque de un turno nuevo invalidan el turno en
      vuelo: el check `turno != self._turno_activo` descarta su texto;
    - los parciales solo llegan a pantalla (regla del ADR-015).
    """

    traducir: Callable[[str], str]
    teleprompter: Teleprompter
    reloj: Callable[[], float] = field(default_factory=lambda: __import__("time").perf_counter)
    ultimo_turno_etapas: dict[str, float] = field(default_factory=dict, init=False)
    ultimo_turno_total_ms: float = field(default=0.0, init=False)
    _numero_turno: int = field(default=0, init=False)
    _turno_activo: int | None = field(default=None, init=False)
    _lock: Any = field(default_factory=threading.Lock, init=False, repr=False)

    def parcial(self, texto_en: str) -> None:
        """Parcial del ASR: SOLO a pantalla (regla del ADR-015)."""
        self.teleprompter.parcial(texto_en)

    def cancelar_turno_activo(self) -> None:
        """Cancela el turno en curso (nuevo segmento del entrevistador)."""
        with self._lock:
            self._turno_activo = None

    def segmento_final(self, texto_en: str) -> bool:
        """Traduce un segmento final EN→ES y lo muestra en pantalla.

        Devuelve True si el texto se mostró; False si el turno fue cancelado
        (o superado) antes de mostrarlo, o si el texto es una alucinación de
        whisper (silencio/ruido que el modelo "rellena" con frases fantasma
        — bug: "whisper habla por mí y dice frases raras"). Las alucinaciones
        se descartan ANTES de asignar número de turno: no cuentan como turno
        ni tocan el teleprompter.
        """
        if es_alucinacion(texto_en):
            return False
        with self._lock:
            self._numero_turno += 1
            turno = self._numero_turno
            self._turno_activo = turno
        etapas = RegistroEtapas(clock=self.reloj)
        etapas.marcar("entrada")
        texto_es = self.traducir(texto_en)
        etapas.marcar("traduccion")
        if turno != self._turno_activo:
            return False  # cancelado: el texto de un turno superado no se muestra
        self.teleprompter.mostrar(texto_es, texto_en)
        etapas.marcar("pantalla")
        self.ultimo_turno_etapas = etapas.desglose_ms()
        self.ultimo_turno_total_ms = etapas.total_ms()
        return True


def _normalizar(texto: str) -> str:
    return " ".join("".join(c for c in texto.lower() if c.isalnum() or c.isspace()).split())


def validar_arranque_en_es(
    traducir: Callable[[str], str],
    *,
    probe: str = _TEXTO_PROBE_EN,
    frase_clave: str = FRASE_CLAVE_EN_A_ES,
) -> Salud:
    """Valida el arranque del flujo incoming de forma BLOQUEANTE (ADR-014/015).

    La traducción EN→ES debe funcionar SIN red antes de aceptar una llamada:
    la precarga del modelo spacy `mwt` se comprueba traduciendo una frase de
    prueba y verificando que la salida sea sana (mismo sanity laxo del
    harness del outgoing: frase clave presente, longitud acotada, sin
    repetición patológica).
    """
    try:
        salida = _normalizar(traducir(probe))
    except Exception as exc:
        return Salud(disponible=False, detalle=f"traducción no disponible al arrancar: {exc}")
    palabras = salida.split()
    if not palabras or frase_clave not in salida or len(salida) > 80:
        return Salud(disponible=False, detalle=DETALLE_BASURA_EN_A_ES)
    if max(palabras.count(p) for p in set(palabras)) > 3:
        return Salud(disponible=False, detalle=DETALLE_BASURA_EN_A_ES)
    return Salud(disponible=True)
