"""Enrolamiento: muestras grabadas del usuario → VoiceProfile validado.

El backend TTS solo recibe perfiles ya validados (ADR-011: `sintetizar`
recibe el perfil por llamada). Aquí se construye el perfil desde el disco,
verificando que cada muestra exista y sea un archivo real antes de aceptarla.
"""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path

from traductor.tts.modelos import VoiceProfile


def enrolar(perfil_id: str, nombre: str, muestras: Sequence[str | Path]) -> VoiceProfile:
    """Construye un `VoiceProfile` validando las muestras de referencia.

    Excepciones:
        ValueError: si no hay muestras, alguna no existe, o alguna no es un
        archivo. La validación de `VoiceProfile` (id/nombre) sigue aplicando.
    """
    if not muestras:
        raise ValueError("muestras vacías: se necesita al menos una muestra")
    rutas = [str(m) for m in muestras]
    faltantes = [r for r in rutas if not Path(r).exists()]
    if faltantes:
        raise ValueError(f"muestras no encontradas: {', '.join(faltantes)}")
    no_archivo = [r for r in rutas if not Path(r).is_file()]
    if no_archivo:
        raise ValueError(f"muestras no son archivos: {', '.join(no_archivo)}")
    return VoiceProfile(id=perfil_id, nombre=nombre, muestras=tuple(rutas))
