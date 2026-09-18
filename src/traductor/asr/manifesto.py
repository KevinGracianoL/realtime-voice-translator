"""Manifest de audios de referencia — carga y validación del benchmark ASR.

Un manifest es una lista JSON de muestras: cada una con `ruta` (relativa al
propio manifest) e `idioma` (es/en), más `referencia` opcional (texto exacto
esperado, para WER). Ejemplo en `scripts/audio/manifesto.json`.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class Muestra:
    """Un audio de referencia para el benchmark: ruta absoluta resuelta."""

    ruta: str
    idioma: str
    referencia: str = ""


def cargar_manifesto(ruta_manifesto: Path) -> list[Muestra]:
    """Carga y valida el manifest. Las rutas se resuelven contra su directorio.

    Excepciones:
        FileNotFoundError: si el archivo no existe.
        ValueError: si la forma es inválida (no lista, o una entrada sin
        ruta/idioma).
    """
    if not ruta_manifesto.exists():
        raise FileNotFoundError(f"manifest no encontrado: {ruta_manifesto}")
    # read_bytes().decode() es UTF-8 en cualquier plataforma (sin literal mutable)
    datos = json.loads(ruta_manifesto.read_bytes().decode())
    if not isinstance(datos, list):
        raise ValueError("manifest debe ser una lista de muestras")
    raiz = ruta_manifesto.parent
    muestras: list[Muestra] = []
    for entrada in datos:
        ruta_relativa = entrada.get("ruta")
        idioma = entrada.get("idioma")
        if not ruta_relativa or not idioma:
            raise ValueError(f"entrada sin ruta o idioma: {entrada}")
        muestras.append(
            Muestra(
                ruta=str(raiz / ruta_relativa),
                idioma=str(idioma).strip(),
                referencia=str(entrada.get("referencia", "")),
            )
        )
    return muestras


def muestras_existentes(muestras: list[Muestra]) -> tuple[list[Muestra], list[str]]:
    """Separa las muestras cuyo audio existe de las que faltan (nombres)."""
    existentes = [m for m in muestras if Path(m.ruta).exists()]
    faltantes = [Path(m.ruta).name for m in muestras if not Path(m.ruta).exists()]
    return existentes, faltantes
