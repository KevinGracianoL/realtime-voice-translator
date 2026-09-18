"""Persistencia real de perfiles — un JSON por perfil en un directorio.

Implementa `VoiceProfileStore` (contrato de `tienda.py`). Esta clase es una
FRONTERA contra el sistema de archivos (ADR-013): el `perfil_id` se valida con
lista blanca (default-deny, no blacklist) antes de tocar el FS — un id como
"../secreto" debe fallar, no escribir/borrar fuera. Lectura/escritura UTF-8
por bytes. Un archivo corrupto no se silencia: `listar` lo omite pero
`listar_errores` lo reporta con su ruta.
"""

from __future__ import annotations

import json
import re
from pathlib import Path

from traductor.tts.modelos import VoiceProfile

_ID_VALIDO = re.compile(r"^[A-Za-z0-9_-]+$")


def _validar_id(perfil_id: str) -> None:
    """Valida el id con lista blanca antes de cualquier operación de FS.

    Excepciones:
        ValueError: si el id no es `[A-Za-z0-9_-]+`.
    """
    if not _ID_VALIDO.fullmatch(perfil_id):
        raise ValueError(f"perfil_id inválido (solo [A-Za-z0-9_-]): {perfil_id!r}")


class TiendaPerfilesJson:
    """VoiceProfileStore sobre `directorio`, un `{id}.json` por perfil."""

    def __init__(self, directorio: Path) -> None:
        self._directorio = directorio

    def _ruta(self, perfil_id: str) -> Path:
        _validar_id(perfil_id)
        return self._directorio / f"{perfil_id}.json"

    @staticmethod
    def _serializar(perfil: VoiceProfile) -> dict[str, object]:
        return {"id": perfil.id, "nombre": perfil.nombre, "muestras": list(perfil.muestras)}

    @staticmethod
    def _deserializar(datos: dict[str, object]) -> VoiceProfile:
        id_perfil = datos.get("id")
        nombre = datos.get("nombre")
        muestras = datos.get("muestras")
        if not isinstance(id_perfil, str) or not isinstance(nombre, str):
            raise ValueError("id y nombre deben ser texto")
        if not isinstance(muestras, list):
            raise ValueError("muestras debe ser una lista")
        return VoiceProfile(
            id=id_perfil,
            nombre=nombre,
            muestras=tuple(str(m) for m in muestras),
        )

    def _leer(self, ruta: Path) -> VoiceProfile:
        try:
            datos: dict[str, object] = json.loads(ruta.read_bytes().decode())
        except (json.JSONDecodeError, UnicodeDecodeError) as exc:
            raise ValueError(f"perfil corrupto ({ruta}): {exc}") from exc
        try:
            return self._deserializar(datos)
        except ValueError as exc:
            raise ValueError(f"perfil inválido ({ruta}): {exc}") from exc

    def listar(self) -> list[VoiceProfile]:
        if not self._directorio.exists():
            return []
        perfiles: list[VoiceProfile] = []
        for archivo in sorted(self._directorio.glob("*.json")):
            try:
                perfiles.append(self._leer(archivo))
            except ValueError:
                continue
        return perfiles

    def listar_errores(self) -> list[str]:
        """Rutas de los perfiles corruptos (listar los omite, no los calla)."""
        if not self._directorio.exists():
            return []
        errores: list[str] = []
        for archivo in sorted(self._directorio.glob("*.json")):
            try:
                self._leer(archivo)
            except ValueError:
                errores.append(str(archivo))
        return errores

    def obtener(self, perfil_id: str) -> VoiceProfile | None:
        ruta = self._ruta(perfil_id)
        if not ruta.exists():
            return None
        return self._leer(ruta)

    def guardar(self, perfil: VoiceProfile) -> None:
        self._directorio.mkdir(parents=True, exist_ok=True)
        # json.dumps con default (ASCII-escaped): bytes válidos UTF-8 en cualquier
        # plataforma. Sin literal ensure_ascii que mute a un equivalente (None==False).
        datos = json.dumps(self._serializar(perfil))
        self._ruta(perfil.id).write_bytes(datos.encode())
        return None

    def eliminar(self, perfil_id: str) -> bool:
        ruta = self._ruta(perfil_id)
        if not ruta.exists():
            return False
        ruta.unlink()
        return True
