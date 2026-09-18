# ADR-005 - Local, no remoto (argos offline; teleprompter en localhost)

- **Estado:** Aceptado (2026-08-20, paso 2) — **reconstruido el 2026-09-18** desde los commits y el código.
- **Contexto:** la traducción podía hacerse con una API en la nube (DeepL, Google) o localmente; el teleprompter podía desplegarse en un servidor o correr en la propia máquina.
- **Decisión:** **todo local**. Traducción con **argos-translate offline en CPU** (no DeepL/Google), y el teleprompter corriendo en `localhost`. El proyecto **no publica un demo gestionado**: quien quiera usarlo clona el repo y lo corre local (0 ms de red).
- **Por qué:**
  - **Privacidad real:** el audio y el texto de la entrevista no salen de la máquina. Cero datos a terceros.
  - **Latencia:** 0 ms de red; con AVX2 en CPU y CUDA en GPU, la latencia local gana a la geografía de una API.
  - **Coste y disponibilidad:** sin API keys, sin cuotas, sin cortes.
- **Consecuencias:**
  - `requirements.txt`: `argostranslate==1.11.0`; `ARGOS_COMPUTE_TYPE=default` es **obligatorio** (sin él, argos es→en produce basura — bug documentado con el guard en `argos.py` y la nota del harness del ADR-014, sin test propio).
  - El proyecto es **local-only por diseño**: el pipeline corre en la máquina del usuario; hubo una demo web del UI que quedó fuera del alcance y se eliminó (2026-09-18).
  - Los benchmarks de latencia del ADR-014 incluyen `0 ms` de red como parte del contexto.
- **Trazabilidad:** commit `700d2c4` (paso 2): el código de la época decía «DECISIÓN (ADR-005): argos-translate, no DeepL/Google». Código vigente: `traduccion/argos.py` (cabecera con ADR-005), `tests/test_argos.py`.
