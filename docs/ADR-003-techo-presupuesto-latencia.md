# ADR-003 - Techo de 1,5-2 s: se recorta calidad, nunca latencia

- **Estado:** Aceptado (2026-09-01, paso 3) — **reconstruido el 2026-09-18** desde los commits, el código y las citas de los ADRs posteriores.
- **Contexto:** una conversación de entrevista tiene un ritmo; pasados ~2 s de retraso, el diálogo se rompe (el interlocutor habla encima, se pierde el hilo). El presupuesto de latencia es, por tanto, un requisito de primer orden, no un detalle de optimización.
- **Decisión:** **techo de 1,5-2 s end-to-end**. Si el presupuesto no cabe, se recorta **calidad** (modelo más pequeño, voz genérica en vez de clonada, menos contexto), **nunca latencia**. El presupuesto se **deriva** de las etapas medidas: `2000 ms − (ASR + traducción + ruteo)` — nunca se declara una constante inventada (regla que el ADR-014 convirtió en gate).
- **Por qué:** la calidad degradada es tolerable en una entrevista (una voz genérica comunica igual); el silencio de 5 s no. La degradación es una decisión de producto, no un accidente.
- **Consecuencias:**
  - `src/traductor/latencia/presupuesto.py` (`cabe_en_presupuesto`, `degradar_configuracion`) y `latencia/medidor.py` (reloj inyectable, p50/p95 honesto: **nunca se reporta p95 con n<20**).
  - Es el ADR más citado del repo: ADR-010/011/012/014/015 y el README lo referencian como el presupuesto que gobierna cada decisión de motor.
  - El gate final lo mide el ADR-014: pipeline end-to-end **1469.3 ms** (peor caso, 4 corridas) < 2000 ms.
- **Trazabilidad:** commits `20d683e` (paso 3, «medidor latencia inyectable»), `163ef43` («medidor blindado», review), `f389cf9`. Código: `latencia/presupuesto.py`, `latencia/medidor.py` (cita ADR-003 en la línea 62). Tests: `tests/test_presupuesto.py` (13), `tests/test_medidor.py` (13).
