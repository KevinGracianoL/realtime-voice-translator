"""Tests de TiendaPerfilesJson — persistencia real de VoiceProfile (PR #14)."""

import re
from pathlib import Path

import pytest

from traductor.tts.modelos import VoiceProfile
from traductor.tts.tienda import VoiceProfileStore
from traductor.tts.tienda_json import TiendaPerfilesJson


def test_tienda_implementa_el_contrato(tmp_path: Path) -> None:
    assert isinstance(TiendaPerfilesJson(tmp_path), VoiceProfileStore)


def test_tienda_guardar_y_obtener(tmp_path: Path) -> None:
    tienda = TiendaPerfilesJson(tmp_path)
    perfil = VoiceProfile(id="kevin-es", nombre="Kevin", muestras=("a.wav", "b.wav"))
    tienda.guardar(perfil)
    assert tienda.obtener("kevin-es") == perfil


def test_tienda_obtener_ausente_none(tmp_path: Path) -> None:
    tienda = TiendaPerfilesJson(tmp_path)
    assert tienda.obtener("no-existe") is None


def test_tienda_listar_vacio(tmp_path: Path) -> None:
    tienda = TiendaPerfilesJson(tmp_path)
    assert tienda.listar() == []


def test_tienda_listar_sin_directorio(tmp_path: Path) -> None:
    tienda = TiendaPerfilesJson(tmp_path / "no-existe")
    assert tienda.listar() == []
    assert tienda.listar_errores() == []


def test_tienda_listar_varios(tmp_path: Path) -> None:
    tienda = TiendaPerfilesJson(tmp_path)
    tienda.guardar(VoiceProfile(id="b", nombre="B", muestras=("1.wav",)))
    tienda.guardar(VoiceProfile(id="a", nombre="A", muestras=("2.wav",)))
    assert [p.id for p in tienda.listar()] == ["a", "b"]


def test_tienda_listar_ignora_no_json(tmp_path: Path) -> None:
    tienda = TiendaPerfilesJson(tmp_path)
    (tmp_path / "nota.txt").write_text("hola", encoding="utf-8")
    tienda.guardar(VoiceProfile(id="a", nombre="A", muestras=("1.wav",)))
    assert [p.id for p in tienda.listar()] == ["a"]


def test_tienda_guardar_reemplaza(tmp_path: Path) -> None:
    tienda = TiendaPerfilesJson(tmp_path)
    tienda.guardar(VoiceProfile(id="a", nombre="Antes", muestras=("1.wav",)))
    tienda.guardar(VoiceProfile(id="a", nombre="Después", muestras=("2.wav",)))
    perfil = tienda.obtener("a")
    assert perfil is not None
    assert perfil.nombre == "Después"


def test_tienda_eliminar(tmp_path: Path) -> None:
    tienda = TiendaPerfilesJson(tmp_path)
    tienda.guardar(VoiceProfile(id="a", nombre="A", muestras=("1.wav",)))
    assert tienda.eliminar("a") is True
    assert tienda.eliminar("a") is False
    assert tienda.obtener("a") is None


def test_tienda_archivo_utf8(tmp_path: Path) -> None:
    """Tildes sobreviven el round-trip: guardar → leer es idéntico."""
    tienda = TiendaPerfilesJson(tmp_path)
    perfil = VoiceProfile(id="a", nombre="Álvaro", muestras=("café.wav",))
    tienda.guardar(perfil)
    assert tienda.obtener("a") == perfil


def test_tienda_id_no_texto_raise(tmp_path: Path) -> None:
    """Un perfil con id no-texto (solo uno de los campos mal) falla claro CON ruta."""
    tienda = TiendaPerfilesJson(tmp_path)
    archivo = tmp_path / "a.json"
    archivo.write_bytes(b'{"id": 1, "nombre": "A", "muestras": ["1.wav"]}')
    mensaje = f"perfil inválido ({archivo}): id y nombre deben ser texto"
    with pytest.raises(ValueError, match=re.escape(mensaje)):
        tienda.obtener("a")


def test_tienda_muestras_no_lista_raise(tmp_path: Path) -> None:
    tienda = TiendaPerfilesJson(tmp_path)
    archivo = tmp_path / "a.json"
    archivo.write_bytes(b'{"id": "a", "nombre": "A", "muestras": "1.wav"}')
    mensaje = f"perfil inválido ({archivo}): muestras debe ser una lista"
    with pytest.raises(ValueError, match=re.escape(mensaje)):
        tienda.obtener("a")


def test_tienda_archivo_corrupto_raise(tmp_path: Path) -> None:
    """Un perfil corrupto NO se silencia: obtener falla claro CON la ruta."""
    tienda = TiendaPerfilesJson(tmp_path)
    archivo = tmp_path / "a.json"
    archivo.write_bytes(b"{no-json")
    with pytest.raises(ValueError, match=re.escape(f"perfil corrupto ({archivo})")):
        tienda.obtener("a")


def test_tienda_listar_con_corrupto_omite_y_reporta(tmp_path: Path) -> None:
    """listar no se rompe entero: omite el corrupto, listar_errores lo reporta."""
    tienda = TiendaPerfilesJson(tmp_path)
    tienda.guardar(VoiceProfile(id="sano", nombre="Sano", muestras=("1.wav",)))
    corrupto = tmp_path / "roto.json"
    corrupto.write_bytes(b"{no-json")
    assert [p.id for p in tienda.listar()] == ["sano"]
    assert tienda.listar_errores() == [str(corrupto)]


def test_tienda_listar_errores_sin_corruptos(tmp_path: Path) -> None:
    tienda = TiendaPerfilesJson(tmp_path)
    tienda.guardar(VoiceProfile(id="a", nombre="A", muestras=("1.wav",)))
    assert tienda.listar_errores() == []


def test_tienda_guardar_crea_directorio(tmp_path: Path) -> None:
    tienda = TiendaPerfilesJson(tmp_path / "sub" / "dir")
    tienda.guardar(VoiceProfile(id="a", nombre="A", muestras=("1.wav",)))
    assert (tmp_path / "sub" / "dir" / "a.json").exists()


def test_tienda_id_fuera_del_directorio_raise(tmp_path: Path) -> None:
    """id con '..' o '/': lista blanca, default-deny — nunca toca el FS (revisión r1)."""
    tienda = TiendaPerfilesJson(tmp_path)
    victima = tmp_path.parent / "victima.json"
    victima.write_bytes(b"CONTENIDO ORIGINAL")
    with pytest.raises(ValueError, match="perfil_id inválido"):
        tienda.obtener("../victima")
    with pytest.raises(ValueError, match="perfil_id inválido"):
        tienda.guardar(VoiceProfile(id="../fuera", nombre="A", muestras=("1.wav",)))
    with pytest.raises(ValueError, match="perfil_id inválido"):
        tienda.eliminar("../victima")
    assert victima.read_bytes() == b"CONTENIDO ORIGINAL"
    assert not (tmp_path / ".." / "fuera.json").exists()


def test_tienda_ids_validos_con_guion_guion_bajo() -> None:
    tienda = TiendaPerfilesJson(Path("no-se-usa"))
    assert tienda._ruta("kevin-es") == Path("no-se-usa") / "kevin-es.json"
    assert tienda._ruta("a_1") == Path("no-se-usa") / "a_1.json"
