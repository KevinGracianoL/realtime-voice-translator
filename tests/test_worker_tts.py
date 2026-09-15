"""Tests del worker TTS aislado — protocolo job/resultado con backend inyectado.

Cubre el camino feliz y los modos de fallo del protocolo (Hal r1): escritura
que falla, salida fuera del directorio, perfil inválido, y que el worker NUNCA
muere por un job malo.
"""

import io
import json
from pathlib import Path

from traductor.tts.backend import TTSBackend
from traductor.tts.modelos import AudioResult, Salud, VoiceProfile
from traductor.tts.tienda import VoiceProfileStore
from traductor.tts.worker import Job, main, procesar_job


class BackendFake:
    """TTSBackend falso: devuelve audio fijo o lanza si se pide."""

    def __init__(self, fallar: bool = False) -> None:
        self._fallar = fallar
        self.sintesis: list[tuple[str, VoiceProfile]] = []

    def sintetizar(self, texto: str, perfil: VoiceProfile) -> AudioResult:
        if self._fallar:
            raise RuntimeError("GPU no disponible")
        self.sintesis.append((texto, perfil))
        return AudioResult(datos=f"audio-{texto}".encode(), formato="wav")

    def verificar_salud(self) -> Salud:
        return Salud(disponible=True)

    def cerrar(self) -> None:
        return None


class TiendaFake:
    """VoiceProfileStore en memoria."""

    def __init__(self, perfiles: dict[str, VoiceProfile] | None = None) -> None:
        self._perfiles = dict(perfiles or {})

    def listar(self) -> list[VoiceProfile]:
        return list(self._perfiles.values())

    def obtener(self, perfil_id: str) -> VoiceProfile | None:
        if perfil_id == "revienta":
            raise RuntimeError("tienda rota")
        return self._perfiles.get(perfil_id)

    def guardar(self, perfil: VoiceProfile) -> None:
        self._perfiles[perfil.id] = perfil

    def eliminar(self, perfil_id: str) -> bool:
        return self._perfiles.pop(perfil_id, None) is not None


PERFIL = VoiceProfile(id="kevin-es", nombre="Kevin", muestras=("a.wav",))


def test_worker_implementa_el_contrato() -> None:
    assert isinstance(BackendFake(), TTSBackend)
    assert isinstance(TiendaFake(), VoiceProfileStore)


def test_procesar_job_ok(tmp_path: Path) -> None:
    backend = BackendFake()
    salida = tmp_path / "salida.wav"
    job = Job(texto="hola", perfil_id="kevin-es", salida="salida.wav")
    resultado = procesar_job(
        job, backend, TiendaFake({"kevin-es": PERFIL}), directorio_salida=tmp_path
    )
    assert resultado["ok"] is True
    assert resultado["salida"] == "salida.wav"
    assert resultado["formato"] == "wav"
    assert salida.read_bytes() == b"audio-hola"
    assert resultado["elapsed_ms"] >= 0
    assert backend.sintesis == [("hola", PERFIL)]


def test_procesar_job_usa_reloj_inyectable(tmp_path: Path) -> None:
    job = Job(texto="hola", perfil_id="kevin-es", salida="salida.wav")
    reloj = iter([0.0, 0.5])

    def clock() -> float:
        return next(reloj)

    resultado = procesar_job(
        job,
        BackendFake(),
        TiendaFake({"kevin-es": PERFIL}),
        directorio_salida=tmp_path,
        clock=clock,
    )
    assert resultado["ok"] is True
    assert resultado["elapsed_ms"] == 500.0


def test_procesar_job_crea_directorio(tmp_path: Path) -> None:
    job = Job(texto="hola", perfil_id="kevin-es", salida="sub/dir/salida.wav")
    resultado = procesar_job(
        job, BackendFake(), TiendaFake({"kevin-es": PERFIL}), directorio_salida=tmp_path
    )
    assert resultado["ok"] is True
    assert (tmp_path / "sub" / "dir" / "salida.wav").exists()


def test_procesar_job_perfil_ausente(tmp_path: Path) -> None:
    job = Job(texto="hola", perfil_id="otro", salida="salida.wav")
    resultado = procesar_job(job, BackendFake(), TiendaFake(), directorio_salida=tmp_path)
    assert resultado["ok"] is False
    assert "perfil no encontrado" in resultado["error"]
    assert not (tmp_path / "salida.wav").exists()


def test_procesar_job_perfil_que_revienta(tmp_path: Path) -> None:
    """La tienda lanza (impl rota): error como valor, el worker no muere."""
    job = Job(texto="hola", perfil_id="revienta", salida="salida.wav")
    resultado = procesar_job(job, BackendFake(), TiendaFake(), directorio_salida=tmp_path)
    assert resultado["ok"] is False
    assert "no se pudo obtener el perfil" in resultado["error"]


def test_procesar_job_backend_falla_no_mata(tmp_path: Path) -> None:
    job = Job(texto="hola", perfil_id="kevin-es", salida="salida.wav")
    resultado = procesar_job(
        job, BackendFake(fallar=True), TiendaFake({"kevin-es": PERFIL}), directorio_salida=tmp_path
    )
    assert resultado["ok"] is False
    assert "GPU no disponible" in resultado["error"]
    assert not (tmp_path / "salida.wav").exists()


def test_procesar_job_escritura_falla_no_mata(tmp_path: Path) -> None:
    """mkdir/write_bytes fallan (causa externa esperable): error, no crash (Hal r1)."""
    bloque = tmp_path / "archivo.txt"
    bloque.write_bytes(b"x")
    job = Job(texto="hola", perfil_id="kevin-es", salida="archivo.txt/debajo.wav")
    resultado = procesar_job(
        job, BackendFake(), TiendaFake({"kevin-es": PERFIL}), directorio_salida=tmp_path
    )
    assert resultado["ok"] is False
    assert "no se pudo escribir" in resultado["error"]


def test_procesar_job_salida_fuera_directorio(tmp_path: Path) -> None:
    """salida con '..' se rechaza: nunca se escribe fuera (Hal r1)."""
    fuera = tmp_path.parent / "fuera.wav"
    job = Job(texto="hola", perfil_id="kevin-es", salida="../fuera.wav")
    resultado = procesar_job(
        job, BackendFake(), TiendaFake({"kevin-es": PERFIL}), directorio_salida=tmp_path
    )
    assert resultado["ok"] is False
    assert "fuera del directorio" in resultado["error"]
    assert not fuera.exists()


def test_procesar_job_salida_absoluta_fuera_rechazada(tmp_path: Path) -> None:
    """Ruta absoluta fuera del directorio de trabajo: rechazada."""
    fuera = tmp_path.parent / "abs.wav"
    job = Job(texto="hola", perfil_id="kevin-es", salida=str(fuera))
    resultado = procesar_job(
        job, BackendFake(), TiendaFake({"kevin-es": PERFIL}), directorio_salida=tmp_path
    )
    assert resultado["ok"] is False
    assert "fuera del directorio" in resultado["error"]
    assert not fuera.exists()


def test_main_procesa_jobs(tmp_path: Path) -> None:
    entrada = io.StringIO(
        json.dumps({"texto": "hola", "perfil_id": "kevin-es", "salida": "salida.wav"}) + "\n"
    )
    salida_texto = io.StringIO()
    main(
        BackendFake(),
        TiendaFake({"kevin-es": PERFIL}),
        tmp_path,
        entrada=entrada,
        salida=salida_texto,
    )
    lineas = [json.loads(linea) for linea in salida_texto.getvalue().strip().splitlines()]
    assert lineas[0]["ok"] is True
    assert (tmp_path / "salida.wav").read_bytes() == b"audio-hola"


def test_main_no_muere_con_job_malo_y_sigue(tmp_path: Path) -> None:
    """Un job malo (escritura que falla) NO mata al worker: el siguiente se procesa."""
    bloque = tmp_path / "archivo.txt"
    bloque.write_bytes(b"x")
    entrada = io.StringIO(
        json.dumps({"texto": "hola", "perfil_id": "kevin-es", "salida": "archivo.txt/debajo.wav"})
        + "\n"
        + json.dumps({"texto": "adiós", "perfil_id": "kevin-es", "salida": "ok.wav"})
        + "\n"
    )
    salida_texto = io.StringIO()
    main(
        BackendFake(),
        TiendaFake({"kevin-es": PERFIL}),
        tmp_path,
        entrada=entrada,
        salida=salida_texto,
    )
    lineas = [json.loads(linea) for linea in salida_texto.getvalue().strip().splitlines()]
    assert lineas[0]["ok"] is False
    assert lineas[1]["ok"] is True
    assert (tmp_path / "ok.wav").read_bytes() == "audio-adiós".encode()


def test_main_job_invalido_continua(tmp_path: Path) -> None:
    entrada = io.StringIO(
        "{no-json}\n"
        + json.dumps({"texto": "hola", "perfil_id": "kevin-es", "salida": "salida.wav"})
        + "\n"
    )
    salida_texto = io.StringIO()
    main(
        BackendFake(),
        TiendaFake({"kevin-es": PERFIL}),
        tmp_path,
        entrada=entrada,
        salida=salida_texto,
    )
    lineas = [json.loads(linea) for linea in salida_texto.getvalue().strip().splitlines()]
    assert lineas[0]["ok"] is False
    assert "job inválido" in lineas[0]["error"]
    assert lineas[1]["ok"] is True


def test_main_continua_despues_de_linea_vacia(tmp_path: Path) -> None:
    entrada = io.StringIO(
        "\n" + json.dumps({"texto": "hola", "perfil_id": "kevin-es", "salida": "salida.wav"}) + "\n"
    )
    salida_texto = io.StringIO()
    main(
        BackendFake(),
        TiendaFake({"kevin-es": PERFIL}),
        tmp_path,
        entrada=entrada,
        salida=salida_texto,
    )
    lineas = [json.loads(linea) for linea in salida_texto.getvalue().strip().splitlines()]
    assert lineas[0]["ok"] is True
    assert (tmp_path / "salida.wav").exists()


def test_main_salta_lineas_vacias(tmp_path: Path) -> None:
    entrada = io.StringIO("\n\n")
    salida_texto = io.StringIO()
    main(BackendFake(), TiendaFake(), tmp_path, entrada=entrada, salida=salida_texto)
    assert salida_texto.getvalue() == ""
