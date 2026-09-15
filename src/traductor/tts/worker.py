"""Worker TTS aislado: corre un `TTSBackend` en su propio proceso.

Protocolo: jobs JSON-line por la entrada (`texto`, `perfil_id`, `salida`) y
resultados JSON-line por la salida (`ok`, más `salida`/`elapsed_ms` o `error`).
Un worker NO muere por un job malo: los fallos esperados (parseo, perfil
ausente o inválido, síntesis que falla, escritura que falla, salida fuera del
directorio de trabajo) vuelven como `ResultadoError`, no como excepción.

JOB STREAMING (ADR-019, fix 1): un job con `streaming: true` escribe un
archivo POR CHUNK (`turno.0.pcm`, `turno.1.pcm`, ...) y emite una línea de
resultado por chunk (`tipo: "chunk"`) más una línea final (`tipo: "fin"`).
El cliente cierra el turno con el PRIMER chunk (~0.7 s) mientras el worker
sigue sintetizando; el resto del audio llega por las líneas siguientes.

Frontera contra procesos (ADR-013): `perfil_id` lo valida la tienda (lista
blanca) y `job.salida` se confina al `directorio_salida` del worker
(`resolve()` + `is_relative_to`) — nunca se escribe fuera por un job mentiroso.
El motor real se inyecta (ADR-011: sin motor elegido aún).
"""

from __future__ import annotations

import json
import sys
import time
from collections.abc import Callable, Iterator
from dataclasses import dataclass
from functools import partial
from pathlib import Path
from typing import Literal, TextIO, TypedDict

from traductor.latencia.medidor import medir_tiempo
from traductor.tts.backend import TTSBackend, TTSBackendStream
from traductor.tts.modelos import AudioResult
from traductor.tts.tienda import VoiceProfileStore


@dataclass(frozen=True)
class Job:
    """Una petición de síntesis.

    `salida` es RELATIVA al `directorio_salida` del worker; rutas absolutas o
    con `..` fuera del directorio se rechazan con `ResultadoError`.
    `streaming` pide el modo por chunks (ADR-019).
    """

    texto: str
    perfil_id: str
    salida: str
    streaming: bool = False


class ResultadoOk(TypedDict):
    ok: Literal[True]
    salida: str
    formato: str
    elapsed_ms: float


class ResultadoChunk(TypedDict):
    ok: Literal[True]
    tipo: Literal["chunk"]
    salida: str
    formato: str
    elapsed_ms: float


class ResultadoFin(TypedDict):
    ok: Literal[True]
    tipo: Literal["fin"]
    elapsed_ms: float


class ResultadoError(TypedDict):
    ok: Literal[False]
    error: str


Resultado = ResultadoOk | ResultadoError


def procesar_job(
    job: Job,
    backend: TTSBackend,
    tienda: VoiceProfileStore,
    *,
    directorio_salida: Path,
    clock: Callable[[], float] = time.perf_counter,
) -> Resultado:
    """Sintetiza `job.texto` con el perfil de `job.perfil_id` y escribe audio.

    Nunca lanza por un fallo esperado: la tienda, la síntesis y la escritura
    (incluida la validación de `salida`) devuelven `ResultadoError`.
    """
    try:
        perfil = tienda.obtener(job.perfil_id)
    except Exception as exc:
        return {"ok": False, "error": f"no se pudo obtener el perfil: {exc}"}
    if perfil is None:
        return {"ok": False, "error": f"perfil no encontrado: {job.perfil_id}"}
    try:
        audio, elapsed_ms = medir_tiempo(
            partial(backend.sintetizar, job.texto, perfil),
            clock=clock,
        )
    except Exception as exc:
        return {"ok": False, "error": f"síntesis falló: {exc}"}
    validado = _validar_salida(directorio_salida, job.salida)
    if not isinstance(validado, Path):  # ResultadoError
        return validado
    ruta = validado
    error = _escribir(ruta, job.salida, audio)
    if error is not None:
        return error
    return {"ok": True, "salida": job.salida, "formato": audio.formato, "elapsed_ms": elapsed_ms}


def _parsear_job(linea: str) -> Job:
    datos = json.loads(linea)
    return Job(
        texto=str(datos["texto"]),
        perfil_id=str(datos["perfil_id"]),
        salida=str(datos["salida"]),
        streaming=bool(datos.get("streaming")),
    )


def _validar_salida(directorio_salida: Path, salida: str) -> Path | ResultadoError:
    """Confina `salida` al directorio de trabajo (ADR-013)."""
    base = directorio_salida.resolve()
    ruta = (directorio_salida / salida).resolve()
    if not ruta.is_relative_to(base):
        return {"ok": False, "error": f"salida fuera del directorio de trabajo: {salida}"}
    return ruta


def _escribir(ruta: Path, salida: str, audio: AudioResult) -> ResultadoError | None:
    try:
        ruta.parent.mkdir(parents=True, exist_ok=True)
        ruta.write_bytes(audio.datos)
    except OSError as exc:
        return {"ok": False, "error": f"no se pudo escribir {salida}: {exc}"}
    return None


def procesar_job_stream(
    job: Job,
    backend: TTSBackend,
    tienda: VoiceProfileStore,
    *,
    directorio_salida: Path,
    clock: Callable[[], float] = time.perf_counter,
) -> Iterator[ResultadoChunk | ResultadoFin | ResultadoError]:
    """Sintetiza por chunks (ADR-019): un archivo + una línea por chunk.

    Es un GENERADOR: cada resultado se RINDE en cuanto está listo — el
    PRIMERO llega en ~0.7 s y el cliente cierra el turno con él mientras el
    worker sigue sintetizando el resto. Si devolviera una lista, el primer
    chunk no cruzaría el pipe hasta el final (bug del streaming, ver ADR-019
    fix 1). Los fallos esperados vuelven como `ResultadoError`.
    """
    try:
        perfil = tienda.obtener(job.perfil_id)
    except Exception as exc:
        yield {"ok": False, "error": f"no se pudo obtener el perfil: {exc}"}
        return
    if perfil is None:
        yield {"ok": False, "error": f"perfil no encontrado: {job.perfil_id}"}
        return
    if not isinstance(backend, TTSBackendStream):
        yield {
            "ok": False,
            "error": f"el backend {type(backend).__name__} no soporta streaming",
        }
        return
    try:
        chunks = backend.sintetizar_stream(job.texto, perfil)
    except Exception as exc:
        yield {"ok": False, "error": f"síntesis falló: {exc}"}
        return

    t0 = clock()
    for i, audio in enumerate(chunks):
        salida = f"{job.salida}.{i}.pcm"
        validado = _validar_salida(directorio_salida, salida)
        if not isinstance(validado, Path):  # ResultadoError
            yield validado
            return
        ruta = validado
        error = _escribir(ruta, salida, audio)
        if error is not None:
            yield error
            return
        yield {
            "ok": True,
            "tipo": "chunk",
            "salida": salida,
            "formato": audio.formato,
            "elapsed_ms": (clock() - t0) * 1000.0,
        }
    yield {"ok": True, "tipo": "fin", "elapsed_ms": (clock() - t0) * 1000.0}


def main(
    backend: TTSBackend,
    tienda: VoiceProfileStore,
    directorio_salida: Path,
    *,
    entrada: TextIO = sys.stdin,
    salida: TextIO = sys.stdout,
) -> None:
    """Lee jobs JSON-line de `entrada` y escribe resultados JSON-line.

    Un job inválido devuelve `ResultadoError` y el bucle sigue. El único modo
    de que el worker muera es un bug de programación (no un job malo).
    Un job `streaming: true` emite una línea por chunk y un `fin`.
    """
    for linea in entrada:
        if not linea.strip():
            continue
        try:
            job = _parsear_job(linea)
        except Exception as exc:
            salida.write(json.dumps({"ok": False, "error": f"job inválido: {exc}"}) + "\n")
            salida.flush()
            continue
        if job.streaming:
            for r in procesar_job_stream(job, backend, tienda, directorio_salida=directorio_salida):
                salida.write(json.dumps(r) + "\n")
                salida.flush()  # el primer chunk cruza el pipe en cuanto llega
            continue
        salida.write(
            json.dumps(procesar_job(job, backend, tienda, directorio_salida=directorio_salida))
            + "\n"
        )
        salida.flush()
