"""Tests del benchmark ASR — WER, agregación y tabla comparativa.

PR #13 NO sustituye RealtimeSTT: mide faster-whisper vs moonshine (ES y EN)
para decidir con evidencia. Aquí se prueba lo puro (métricas y formato); los
motores reales corren en `scripts/benchmark_asr.py` en la máquina objetivo.
"""

import json
from pathlib import Path

import pytest

from traductor.asr.benchmark import resumen_asr, tabla_comparativa
from traductor.asr.manifesto import Muestra, cargar_manifesto, muestras_existentes
from traductor.asr.medicion import medir_motores
from traductor.asr.resultado import ResultadoAsr
from traductor.asr.wer import distancia_edicion, normalizar_texto, wer


def test_wer_identicas_cero() -> None:
    assert wer("hola mundo", "hola mundo") == 0.0


def test_wer_borrado() -> None:
    assert wer("hola mundo", "hola") == 0.5


def test_wer_insercion() -> None:
    assert wer("hola mundo", "hola mundo feliz") == 0.5


def test_wer_sustitucion() -> None:
    assert wer("hola mundo", "hola tierra") == 0.5


def test_wer_normaliza_espacios() -> None:
    assert wer("hola   mundo", "hola mundo") == 0.0


def test_wer_vacio_vacio_cero() -> None:
    assert wer("", "") == 0.0


def test_wer_referencia_vacia_con_texto() -> None:
    assert wer("", "hola") == 1.0


def test_normalizar_texto_minusculas_sin_puntuacion() -> None:
    assert normalizar_texto("Si, claro!") == "si claro"
    assert normalizar_texto("Hola, como estas?") == "hola como estas"
    assert normalizar_texto("El cafe esta frio.") == "el cafe esta frio"


def test_normalizar_texto_colapsa_espacios() -> None:
    assert normalizar_texto("  hola   mundo  ") == "hola mundo"


def test_normalizar_texto_conserva_tildes_y_digitos() -> None:
    assert normalizar_texto("café") == "café"
    assert normalizar_texto("3") == "3"


def test_wer_con_puntuacion_es_cero() -> None:
    """Transcripción perfecta con puntuación/mayúsculas: WER 0 (revisión r1)."""
    assert wer("si claro", "Si, claro!") == 0.0
    assert wer("hola como estas", "Hola, como estas?") == 0.0
    assert wer("el cafe esta frio", "El cafe esta frio.") == 0.0


def test_wer_numeros_siguen_siendo_error() -> None:
    assert wer("tres", "3") == 1.0


def test_wer_tildes_siguen_siendo_error() -> None:
    assert wer("café", "cafe") == 1.0


def test_resultado_valido() -> None:
    r = ResultadoAsr(engine="faster-whisper", idioma="es", frase="hola", elapsed_ms=123.0)
    assert r.engine == "faster-whisper"
    assert r.idioma == "es"
    assert r.elapsed_ms == 123.0


def test_resultado_latencia_negativa_raise() -> None:
    with pytest.raises(ValueError, match="elapsed_ms"):
        ResultadoAsr(engine="faster-whisper", idioma="es", frase="hola", elapsed_ms=-1.0)


def test_resultado_idioma_vacio_raise() -> None:
    with pytest.raises(ValueError, match="idioma"):
        ResultadoAsr(engine="moonshine", idioma=" ", frase="hola", elapsed_ms=1.0)


def test_resumen_asr_agrupa_por_engine_idioma() -> None:
    resultados = [
        ResultadoAsr("faster-whisper", "es", "hola", 100.0, "hola"),
        ResultadoAsr("faster-whisper", "es", "hola", 300.0, "hola"),
        ResultadoAsr("moonshine", "es", "hola", 200.0, "hola"),
        ResultadoAsr("moonshine", "es", "hala", 200.0, "hola"),
    ]
    resumen = resumen_asr(resultados)
    assert "faster-whisper|es" in resumen
    assert "moonshine|es" in resumen
    assert resumen["faster-whisper|es"]["n"] == 2
    assert resumen["faster-whisper|es"]["p50_ms"] == 200.0
    assert resumen["faster-whisper|es"]["wer_mean"] == 0.0
    assert resumen["moonshine|es"]["wer_mean"] == 0.5


def test_resumen_asr_p95_none_con_pocas() -> None:
    r = ResultadoAsr("faster-whisper", "en", "hi", 100.0, "hi")
    resumen = resumen_asr([r])
    assert resumen["faster-whisper|en"]["p95_ms"] is None


def test_resumen_asr_sin_referencia_wer_none() -> None:
    r = ResultadoAsr("faster-whisper", "es", "hola", 100.0)
    resumen = resumen_asr([r])
    assert resumen["faster-whisper|es"]["wer_n"] == 0.0
    assert resumen["faster-whisper|es"]["wer_mean"] is None


def test_resumen_asr_vacio() -> None:
    assert resumen_asr([]) == {}


def test_tabla_comparativa_formato() -> None:
    r = ResultadoAsr("faster-whisper", "es", "hola", 100.0, "hola")
    tabla = tabla_comparativa(resumen_asr([r]))
    assert "engine" in tabla
    assert "faster-whisper" in tabla
    assert "es" in tabla


def test_tabla_comparativa_output_exacto() -> None:
    resumen: dict[str, dict[str, float | None]] = {
        "faster-whisper|es": {
            "n": 2.0,
            "wer_n": 2.0,
            "p50_ms": 200.0,
            "p95_ms": None,
            "wer_mean": 0.0,
        },
        "moonshine|es": {"n": 1.0, "wer_n": 1.0, "p50_ms": 200.0, "p95_ms": None, "wer_mean": 0.5},
    }
    tabla = tabla_comparativa(resumen)
    assert tabla == (
        "engine          idioma   n  wer_n   p50_ms   p95_ms     wer\n"
        "faster-whisper   es      2     2   200.0      -    0.00\n"
        "moonshine        es      1     1   200.0      -    0.50"
    )


def test_tabla_comparativa_vacia() -> None:
    tabla = tabla_comparativa({})
    assert tabla == "engine          idioma   n  wer_n   p50_ms   p95_ms     wer\n\n(sin datos)"


def test_tabla_comparativa_wer_n_visible() -> None:
    """n=2 con WER calculado sobre una muestra: la tabla lo muestra (revisión r1)."""
    resumen: dict[str, dict[str, float | None]] = {
        "faster-whisper|es": {
            "n": 2.0,
            "wer_n": 1.0,
            "p50_ms": 200.0,
            "p95_ms": None,
            "wer_mean": 0.0,
        }
    }
    tabla = tabla_comparativa(resumen)
    assert "es      2     1" in tabla


def test_tabla_comparativa_n_cero() -> None:
    resumen: dict[str, dict[str, float | None]] = {
        "x|es": {"n": 0.0, "wer_n": 0.0, "p50_ms": None, "p95_ms": None, "wer_mean": None}
    }
    tabla = tabla_comparativa(resumen)
    assert "es      0     0" in tabla


def test_tabla_comparativa_clave_con_varios_pipes() -> None:
    resumen: dict[str, dict[str, float | None]] = {
        "a|b|c": {"n": 1.0, "wer_n": 1.0, "p50_ms": 10.0, "p95_ms": None, "wer_mean": None}
    }
    tabla = tabla_comparativa(resumen)
    assert "a                b|c     1     1" in tabla


def test_tabla_comparativa_p95_finito() -> None:
    """p95 real (n≥20) se pinta con número, no con el guion de None."""
    resumen: dict[str, dict[str, float | None]] = {
        "faster-whisper|es": {
            "n": 20.0,
            "wer_n": 20.0,
            "p50_ms": 200.0,
            "p95_ms": 300.0,
            "wer_mean": 0.1,
        }
    }
    tabla = tabla_comparativa(resumen)
    assert tabla == (
        "engine          idioma   n  wer_n   p50_ms   p95_ms     wer\n"
        "faster-whisper   es     20    20   200.0   300.0    0.10"
    )


def test_tabla_comparativa_inf() -> None:
    """Valores no finitos se pintan en minúscula (inf, no INF): f vs F."""
    resumen: dict[str, dict[str, float | None]] = {
        "x|es": {
            "n": 1.0,
            "wer_n": 1.0,
            "p50_ms": float("inf"),
            "p95_ms": float("inf"),
            "wer_mean": float("inf"),
        }
    }
    tabla = tabla_comparativa(resumen)
    assert tabla == (
        "engine          idioma   n  wer_n   p50_ms   p95_ms     wer\n"
        "x                es      1     1     inf     inf     inf"
    )


def test_distancia_edicion_referencia_vacia() -> None:
    assert distancia_edicion([], ["a"]) == 1


def test_distancia_edicion_hipotesis_vacia() -> None:
    assert distancia_edicion(["a", "b"], []) == 2


def _escribir_manifest(tmp_path: Path, entradas: object) -> Path:
    ruta = tmp_path / "manifesto.json"
    ruta.write_text(json.dumps(entradas), encoding="utf-8")
    return ruta


def test_cargar_manifesto_resuelve_rutas(tmp_path: Path) -> None:
    ruta = _escribir_manifest(tmp_path, [{"ruta": "a.wav", "idioma": "es", "referencia": "hola"}])
    muestras = cargar_manifesto(ruta)
    assert muestras == [Muestra(ruta=str(tmp_path / "a.wav"), idioma="es", referencia="hola")]


def test_cargar_manifesto_utf8() -> None:
    """Referencias con tildes se leen UTF-8 en cualquier plataforma."""
    import tempfile

    with tempfile.TemporaryDirectory() as tmp:
        ruta = Path(tmp) / "manifesto.json"
        ruta.write_bytes('[{"ruta": "a.wav", "idioma": "es", "referencia": "el café"}]'.encode())
        muestras = cargar_manifesto(ruta)
        assert muestras[0].referencia == "el café"


def test_cargar_manifesto_referencia_opcional(tmp_path: Path) -> None:
    ruta = _escribir_manifest(tmp_path, [{"ruta": "a.wav", "idioma": "en"}])
    muestras = cargar_manifesto(ruta)
    assert muestras[0].referencia == ""


def test_cargar_manifesto_faltante_raise(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError, match="manifest"):
        cargar_manifesto(tmp_path / "no-existe.json")


def test_cargar_manifesto_no_lista_raise(tmp_path: Path) -> None:
    ruta = _escribir_manifest(tmp_path, {"ruta": "a.wav"})
    with pytest.raises(ValueError, match="^manifest debe ser una lista de muestras$"):
        cargar_manifesto(ruta)


def test_cargar_manifesto_entrada_incompleta_raise(tmp_path: Path) -> None:
    ruta = _escribir_manifest(tmp_path, [{"ruta": "a.wav"}])
    with pytest.raises(ValueError, match="ruta o idioma"):
        cargar_manifesto(ruta)


def test_muestras_existentes_separa(tmp_path: Path) -> None:
    (tmp_path / "existe.wav").write_bytes(b"RIFF")
    muestras = [
        Muestra(ruta=str(tmp_path / "existe.wav"), idioma="es"),
        Muestra(ruta=str(tmp_path / "falta.wav"), idioma="en"),
    ]
    existentes, faltantes = muestras_existentes(muestras)
    assert len(existentes) == 1
    assert existentes[0].ruta == str(tmp_path / "existe.wav")
    assert faltantes == ["falta.wav"]


def test_medir_motores_mide_y_referencia(tmp_path: Path) -> None:
    audio = tmp_path / "a.wav"
    audio.write_bytes(b"RIFF")
    muestras = [Muestra(ruta=str(audio), idioma="es", referencia="hola")]

    def transcribir(ruta: Path, idioma: str) -> str:
        assert ruta == audio
        assert idioma == "es"
        return "Hola!"

    resultados = medir_motores({"faster-whisper": transcribir}, muestras)
    assert len(resultados) == 1
    r = resultados[0]
    assert r.engine == "faster-whisper"
    assert r.frase == "Hola!"
    assert r.elapsed_ms >= 0
    assert r.referencia == "hola"


def test_medir_motores_todos_los_motores(tmp_path: Path) -> None:
    audio = tmp_path / "a.wav"
    audio.write_bytes(b"RIFF")
    muestras = [Muestra(ruta=str(audio), idioma="es")]
    motores = {
        "faster-whisper": lambda ruta, idioma: "hola",
        "moonshine": lambda ruta, idioma: "hola",
    }
    resultados = medir_motores(motores, muestras)
    assert {r.engine for r in resultados} == {"faster-whisper", "moonshine"}


def test_medir_motores_usa_reloj_inyectable(tmp_path: Path) -> None:
    audio = tmp_path / "a.wav"
    audio.write_bytes(b"RIFF")
    muestras = [Muestra(ruta=str(audio), idioma="es")]

    def transcribir(ruta: Path, idioma: str) -> str:
        return "hola"

    reloj = iter([0.0, 0.5])

    def clock() -> float:
        return next(reloj)

    resultados = medir_motores({"faster-whisper": transcribir}, muestras, clock=clock)
    assert resultados[0].elapsed_ms == 500.0
