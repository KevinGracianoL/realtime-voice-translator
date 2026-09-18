"""Tipos de dominio del TTS — neutrales, sin dependencia de backend.

Son el lenguaje común entre el pipeline (traducción → TTS → audio virtual) y
cualquier proveedor futuro (XTTS, etc.). El PR #12 NO instancia backends:
solo define estos tipos y los protocolos de `backend`/`tienda`.
"""

from __future__ import annotations

from dataclasses import dataclass


def _no_vacio(valor: str, campo: str) -> str:
    """Limpia y valida un campo obligatorio. Excepciones: ValueError."""
    limpio = valor.strip()
    if not limpio:
        raise ValueError(f"{campo} vacío: se necesita un valor")
    return limpio


@dataclass(frozen=True)
class VoiceProfile:
    """Perfil de voz para clonación: identidad + muestras de referencia.

    `muestras` son rutas a audios grabados del usuario (enrolamiento). Al ser
    inmutable, un perfil no puede mutarse a medias entre distintos usos.
    """

    id: str
    nombre: str
    muestras: tuple[str, ...]

    def __post_init__(self) -> None:
        # dataclasses no permite setattr en frozen sin object.__setattr__
        object.__setattr__(self, "id", _no_vacio(self.id, "id"))
        object.__setattr__(self, "nombre", _no_vacio(self.nombre, "nombre"))
        if not self.muestras:
            raise ValueError("muestras vacías: se necesita al menos una muestra")


@dataclass(frozen=True)
class AudioResult:
    """Audio sintetizado: bytes + formato + duración opcional."""

    datos: bytes
    formato: str
    duracion_s: float | None = None

    def __post_init__(self) -> None:
        object.__setattr__(self, "formato", _no_vacio(self.formato, "formato"))


@dataclass(frozen=True)
class Salud:
    """Resultado de healthcheck de un backend TTS.

    `disponible=False` exige `detalle`: un backend caído sin explicación no
    sirve de puerta para los gates de latencia.
    """

    disponible: bool
    detalle: str = ""

    def __post_init__(self) -> None:
        object.__setattr__(self, "detalle", self.detalle.strip())
        if not self.disponible and not self.detalle:
            raise ValueError("detalle vacío: backend caído sin explicación")
