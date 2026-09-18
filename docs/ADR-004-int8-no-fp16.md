# ADR-004 - INT8, no FP16 (TU117 sin Tensor Cores)

- **Estado:** Aceptado (2026-08-13, primer commit del repo) — **reconstruido el 2026-09-18** desde el cuerpo del commit y el código (la era inicial no escribió el archivo; el propio código lo citaba).
- **Contexto:** qué precisión usar para la inferencia en la GPU de referencia (GTX 1650 Ti, Turing **TU117**, compute capability 7.5).
- **Decisión:** **INT8** para whisper (`compute_type="int8"` en el ASR de micrófono, `int8_float16` en los ASR de GPU). **FP16, no.**
- **Por qué (medido en el primer commit):** la TU117 **no tiene Tensor Cores**: FP16 se emula por driver y sale **más lento que FP32**; INT8 corre en los cores enteros y es lo que aprovecha el hardware de esta gama.
- **Consecuencias:**
  - `audio/captura.py` (`compute_type="int8"`), `flujo/adaptadores.py` (`int8` / `int8_float16`), y la tabla del stack del README lo declaran («TU117 sin Tensor Cores → FP16 emulado, INT8 en cores enteros»).
  - Verificado en hardware real: `docs/smoke-windows.txt` registra `WhisperModel('tiny', device='cuda', compute_type='int8_float16')`.
  - Los presupuestos de VRAM/RAM del ADR-014 se calcularon con esta elección.
- **Trazabilidad:** commit `60d0371` (primer commit: «INT8 en vez de FP16: la GTX 1650 Ti (Turing TU117) no tiene Tensor Cores, FP16 se emula por driver y sale más lento que FP32. Medido: … compute capability 7.5»). El antiguo `paso1_verificar.py:107` decía: «Está decidido en el ADR-004 del README de este repo» (el archivo del ADR nunca se escribió; este lo reconstruye).
