"""Tests del worker TTS aislado — protocolo job/resultado con backend inyectado.

Cubre el camino feliz y los modos de fallo del protocolo (revisión r1): escritura
que falla, salida fuera del directorio, perfil inválido, y que el worker NUNCA
muere por un job malo.
"""

import io
import json
from collections.abc import Iterator
from pathlib import Path
from typing import Any

import pytest

from traductor.tts.backend import TTSBackend, TTSBackendStream
from traductor.tts.modelos import AudioResult, Salud, VoiceProfile
from traductor.tts.tienda import VoiceProfileStore
from traductor.tts.worker import Job, main, procesar_job, procesar_job_stream


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
    assert "perfil no encontrado: otro" in resultado["error"]
    assert not (tmp_path / "salida.wav").exists()


def test_procesar_job_perfil_que_revienta(tmp_path: Path) -> None:
    """La tienda lanza (impl rota): error como valor, el worker no muere."""
    job = Job(texto="hola", perfil_id="revienta", salida="salida.wav")
    resultado = procesar_job(job, BackendFake(), TiendaFake(), directorio_salida=tmp_path)
    assert resultado["ok"] is False
    assert "no se pudo obtener el perfil: tienda rota" in resultado["error"]


def test_procesar_job_backend_falla_no_mata(tmp_path: Path) -> None:
    job = Job(texto="hola", perfil_id="kevin-es", salida="salida.wav")
    resultado = procesar_job(
        job, BackendFake(fallar=True), TiendaFake({"kevin-es": PERFIL}), directorio_salida=tmp_path
    )
    assert resultado["ok"] is False
    assert "síntesis falló: GPU no disponible" in resultado["error"]
    assert not (tmp_path / "salida.wav").exists()


def test_procesar_job_escritura_falla_no_mata(tmp_path: Path) -> None:
    """mkdir/write_bytes fallan (causa externa esperable): error, no crash (revisión r1)."""
    bloque = tmp_path / "archivo.txt"
    bloque.write_bytes(b"x")
    job = Job(texto="hola", perfil_id="kevin-es", salida="archivo.txt/debajo.wav")
    resultado = procesar_job(
        job, BackendFake(), TiendaFake({"kevin-es": PERFIL}), directorio_salida=tmp_path
    )
    assert resultado["ok"] is False
    assert "no se pudo escribir archivo.txt/debajo.wav" in resultado["error"]


def test_procesar_job_salida_fuera_directorio(tmp_path: Path) -> None:
    """salida con '..' se rechaza: nunca se escribe fuera (revisión r1)."""
    fuera = tmp_path.parent / "fuera.wav"
    job = Job(texto="hola", perfil_id="kevin-es", salida="../fuera.wav")
    resultado = procesar_job(
        job, BackendFake(), TiendaFake({"kevin-es": PERFIL}), directorio_salida=tmp_path
    )
    assert resultado["ok"] is False
    assert "salida fuera del directorio de trabajo" in resultado["error"]
    assert not fuera.exists()


def test_procesar_job_salida_absoluta_fuera_rechazada(tmp_path: Path) -> None:
    """Ruta absoluta fuera del directorio de trabajo: rechazada."""
    fuera = tmp_path.parent / "abs.wav"
    job = Job(texto="hola", perfil_id="kevin-es", salida=str(fuera))
    resultado = procesar_job(
        job, BackendFake(), TiendaFake({"kevin-es": PERFIL}), directorio_salida=tmp_path
    )
    assert resultado["ok"] is False
    assert "salida fuera del directorio de trabajo" in resultado["error"]
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


class BackendStreamFake(BackendFake):
    """Backend con streaming: dos chunks fijos."""

    def sintetizar_stream(self, texto: str, perfil: VoiceProfile) -> Iterator[AudioResult]:
        self.sintesis.append((texto, perfil))
        yield AudioResult(datos=f"chunk0-{texto}".encode(), formato="pcm_f32le")
        yield AudioResult(datos=f"chunk1-{texto}".encode(), formato="pcm_f32le")


class BackendSinStream:
    """Backend sin streaming: el worker debe fallar con error claro."""

    def sintetizar(self, texto: str, perfil: VoiceProfile) -> AudioResult:
        return AudioResult(datos=b"x", formato="wav")

    def verificar_salud(self) -> Salud:
        return Salud(disponible=True)

    def cerrar(self) -> None:
        return None


def test_worker_stream_fake_implementa_el_contrato() -> None:
    assert isinstance(BackendFake(), TTSBackend)
    assert isinstance(BackendStreamFake(), TTSBackend)
    assert isinstance(BackendStreamFake(), TTSBackendStream)
    assert not isinstance(BackendSinStream(), TTSBackendStream)


def test_procesar_job_stream_emite_chunks_y_fin(tmp_path: Path) -> None:
    job = Job(texto="hola", perfil_id="kevin-es", salida="turno", streaming=True)
    resultados: list[dict[str, Any]] = [
        dict(r)
        for r in procesar_job_stream(
            job, BackendStreamFake(), TiendaFake({"kevin-es": PERFIL}), directorio_salida=tmp_path
        )
    ]
    assert [r.get("tipo", "error") for r in resultados] == ["chunk", "chunk", "fin"]
    assert (tmp_path / "turno.0.pcm").read_bytes() == b"chunk0-hola"
    assert (tmp_path / "turno.1.pcm").read_bytes() == b"chunk1-hola"
    assert float(resultados[0].get("elapsed_ms", -1)) >= 0
    assert float(resultados[2].get("elapsed_ms", -1)) >= float(resultados[0].get("elapsed_ms", -1))


def test_procesar_job_stream_campos_exactos(tmp_path: Path) -> None:
    """Cada línea del stream lleva ok/tipo/salida/formato/elapsed_ms EXACTOS
    (mutantes de claves, valores y unidades del elapsed los cazan)."""
    job = Job(texto="hola", perfil_id="kevin-es", salida="turno", streaming=True)
    # t0 NO es cero: distingue `clock() - t0` de `clock() + t0` (mutante)
    reloj = iter([10.0, 10.5, 10.7, 10.9])

    def clock() -> float:
        return next(reloj)

    resultados: list[dict[str, Any]] = [
        dict(r)
        for r in procesar_job_stream(
            job,
            BackendStreamFake(),
            TiendaFake({"kevin-es": PERFIL}),
            directorio_salida=tmp_path,
            clock=clock,
        )
    ]
    chunk0, chunk1, fin = resultados
    assert chunk0["ok"] is True
    assert chunk0["tipo"] == "chunk"
    assert chunk0["salida"] == "turno.0.pcm"
    assert chunk0["formato"] == "pcm_f32le"
    assert chunk0["elapsed_ms"] == pytest.approx(500.0)
    assert chunk1["salida"] == "turno.1.pcm"
    assert chunk1["elapsed_ms"] == pytest.approx(700.0)
    assert fin["ok"] is True
    assert fin["tipo"] == "fin"
    assert fin["elapsed_ms"] == pytest.approx(900.0)


def test_procesar_job_stream_pasa_el_perfil(tmp_path: Path) -> None:
    """El backend recibe el PERFIL de la tienda (mutante `perfil=None`)."""
    backend = BackendStreamFake()
    job = Job(texto="hola", perfil_id="kevin-es", salida="turno", streaming=True)
    list(
        procesar_job_stream(
            job, backend, TiendaFake({"kevin-es": PERFIL}), directorio_salida=tmp_path
        )
    )
    assert backend.sintesis == [("hola", PERFIL)]


def test_procesar_job_stream_perfil_ausente(tmp_path: Path) -> None:
    job = Job(texto="hola", perfil_id="otro", salida="turno", streaming=True)
    resultados = list(
        procesar_job_stream(job, BackendStreamFake(), TiendaFake(), directorio_salida=tmp_path)
    )
    assert resultados[0]["ok"] is False
    assert "perfil no encontrado: otro" in resultados[0]["error"]


def test_procesar_job_stream_perfil_que_revienta(tmp_path: Path) -> None:
    """La tienda lanza en el camino streaming: error como valor, sin fin."""
    job = Job(texto="hola", perfil_id="revienta", salida="turno", streaming=True)
    resultados = list(
        procesar_job_stream(job, BackendStreamFake(), TiendaFake(), directorio_salida=tmp_path)
    )
    assert resultados[0]["ok"] is False
    assert "no se pudo obtener el perfil: tienda rota" in resultados[0]["error"]


def test_procesar_job_stream_sintesis_que_revienta(tmp_path: Path) -> None:
    """El stream que lanza al crearse: error como valor, sin fin, worker vivo."""

    class BackendRevientaStream(BackendFake):
        def sintetizar_stream(self, texto: str, perfil: VoiceProfile) -> Iterator[AudioResult]:
            raise RuntimeError("GPU no disponible")

    job = Job(texto="hola", perfil_id="kevin-es", salida="turno", streaming=True)
    resultados = list(
        procesar_job_stream(
            job,
            BackendRevientaStream(),
            TiendaFake({"kevin-es": PERFIL}),
            directorio_salida=tmp_path,
        )
    )
    assert resultados[0]["ok"] is False
    assert "síntesis falló: GPU no disponible" in resultados[0]["error"]
    assert len(resultados) == 1


def test_procesar_job_stream_backend_sin_streaming_falla_limpio(tmp_path: Path) -> None:
    """Un backend sin `sintetizar_stream` NO rompe el worker: error claro."""
    job = Job(texto="hola", perfil_id="kevin-es", salida="turno", streaming=True)
    resultados = list(
        procesar_job_stream(
            job, BackendSinStream(), TiendaFake({"kevin-es": PERFIL}), directorio_salida=tmp_path
        )
    )
    assert resultados[0]["ok"] is False
    assert "BackendSinStream no soporta streaming" in resultados[0]["error"]


def test_procesar_job_stream_escritura_falla_no_mata(tmp_path: Path) -> None:
    """Un chunk que no se puede escribir vuelve error y corta (sin fin)."""
    bloque = tmp_path / "bloque"
    bloque.write_text("no soy un directorio")
    job = Job(texto="hola", perfil_id="kevin-es", salida="bloque/turno", streaming=True)
    resultados = list(
        procesar_job_stream(
            job, BackendStreamFake(), TiendaFake({"kevin-es": PERFIL}), directorio_salida=tmp_path
        )
    )
    assert resultados[0]["ok"] is False
    assert "no se pudo escribir bloque/turno.0.pcm" in resultados[0]["error"]
    assert len(resultados) == 1  # sin fin: el cliente ve el error y escala


def test_procesar_job_stream_salida_fuera_directorio(tmp_path: Path) -> None:
    job = Job(texto="hola", perfil_id="kevin-es", salida="../fuera", streaming=True)
    resultados = list(
        procesar_job_stream(
            job, BackendStreamFake(), TiendaFake({"kevin-es": PERFIL}), directorio_salida=tmp_path
        )
    )
    assert resultados[0]["ok"] is False
    assert "salida fuera del directorio de trabajo: ../fuera.0.pcm" in resultados[0]["error"]


def test_main_procesa_job_streaming(tmp_path: Path) -> None:
    """El main rutea un job `streaming: true` al camino por chunks."""
    entrada = io.StringIO(
        json.dumps({"texto": "hola", "perfil_id": "kevin-es", "salida": "turno", "streaming": True})
        + "\n"
    )
    salida_texto = io.StringIO()
    main(
        BackendStreamFake(),
        TiendaFake({"kevin-es": PERFIL}),
        tmp_path,
        entrada=entrada,
        salida=salida_texto,
    )
    lineas = [json.loads(linea) for linea in salida_texto.getvalue().strip().splitlines()]
    assert [linea.get("tipo", "error") for linea in lineas] == ["chunk", "chunk", "fin"]
    assert (tmp_path / "turno.0.pcm").read_bytes() == b"chunk0-hola"


def test_main_streaming_sigue_con_el_siguiente_job(tmp_path: Path) -> None:
    """Tras un job streaming el worker sigue con el SIGUIENTE job (mutante
    `continue`→`break` del main: con break, el job normal no se procesaría)."""
    entrada = io.StringIO(
        json.dumps({"texto": "uno", "perfil_id": "kevin-es", "salida": "s1", "streaming": True})
        + "\n"
        + json.dumps({"texto": "dos", "perfil_id": "kevin-es", "salida": "s2.wav"})
        + "\n"
    )
    salida_texto = io.StringIO()
    main(
        BackendStreamFake(),
        TiendaFake({"kevin-es": PERFIL}),
        tmp_path,
        entrada=entrada,
        salida=salida_texto,
    )
    lineas = [json.loads(linea) for linea in salida_texto.getvalue().strip().splitlines()]
    assert [linea.get("tipo", "normal") for linea in lineas] == ["chunk", "chunk", "fin", "normal"]
    assert lineas[-1]["ok"] is True
    assert (tmp_path / "s2.wav").read_bytes() == b"audio-dos"


def test_main_job_streaming_sin_streaming_en_backend(tmp_path: Path) -> None:
    """Job streaming contra backend sin streaming: error claro, worker vivo."""
    entrada = io.StringIO(
        json.dumps({"texto": "hola", "perfil_id": "kevin-es", "salida": "turno", "streaming": True})
        + "\n"
    )
    salida_texto = io.StringIO()
    main(
        BackendSinStream(),
        TiendaFake({"kevin-es": PERFIL}),
        tmp_path,
        entrada=entrada,
        salida=salida_texto,
    )
    lineas = [json.loads(linea) for linea in salida_texto.getvalue().strip().splitlines()]
    assert lineas[0]["ok"] is False
    assert "no soporta streaming" in lineas[0]["error"]
