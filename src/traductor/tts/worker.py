"""Worker TTS aislado: corre un `TTSBackend` en su propio proceso.

Protocolo: jobs JSON-line por la entrada (`texto`, `perfil_id`, `salida`) y
resultados JSON-line por la salida (`ok`, más `salida`/`elapsed_ms` o `error`).
Un worker NO muere por un job malo: los fallos esperados (parseo, perfil
ausente o inválido, síntesis que falla, escritura que falla, salida fuera del
directorio de trabajo) vuelven como `ResultadoError`, no como excepción.

Frontera contra procesos (ADR-013): `perfil_id` lo valida la tienda (lista
blanca) y `job.salida` se confina al `directorio_salida` del worker
(`resolve()` + `is_relative_to`) — nunca se escribe fuera por un job mentiroso.
El motor real se inyecta (ADR-011: sin motor elegido aún).
"""

from __future__ import annotations

import json
import sys
import time
from collections.abc import Callable
from dataclasses import dataclass
from functools import partial
from pathlib import Path
from typing import Literal, TextIO, TypedDict

from traductor.latencia.medidor import medir_tiempo
from traductor.tts.backend import TTSBackend
from traductor.tts.tienda import VoiceProfileStore


@dataclass(frozen=True)
class Job:
    """Una petición de síntesis.

    `salida` es RELATIVA al `directorio_salida` del worker; rutas absolutas o
    con `..` fuera del directorio se rechazan con `ResultadoError`.
    """

    texto: str
    perfil_id: str
    salida: str


class ResultadoOk(TypedDict):
    ok: Literal[True]
    salida: str
    formato: str
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
    base = directorio_salida.resolve()
    ruta = (directorio_salida / job.salida).resolve()
    if not ruta.is_relative_to(base):
        return {"ok": False, "error": f"salida fuera del directorio de trabajo: {job.salida}"}
    try:
        ruta.parent.mkdir(parents=True, exist_ok=True)
        ruta.write_bytes(audio.datos)
    except OSError as exc:
        return {"ok": False, "error": f"no se pudo escribir {job.salida}: {exc}"}
    return {"ok": True, "salida": job.salida, "formato": audio.formato, "elapsed_ms": elapsed_ms}


def _parsear_job(linea: str) -> Job:
    datos = json.loads(linea)
    return Job(
        texto=str(datos["texto"]),
        perfil_id=str(datos["perfil_id"]),
        salida=str(datos["salida"]),
    )


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
        salida.write(
            json.dumps(procesar_job(job, backend, tienda, directorio_salida=directorio_salida))
            + "\n"
        )
        salida.flush()
