# ADR-007 - Teleprompter primero, TTS opcional

- **Estado:** Aceptado (2026-09-02, paso 5) — **reconstruido el 2026-09-18** desde los commits y el código.
- **Contexto:** el objetivo final incluye hablar con la voz clonada en inglés (TTS + clonación), pero esa pieza es cara y larga (elección de motor, entrenamiento/enrolamiento, gates — toda la Fase 2). En una entrevista real, lo que desbloquea valor inmediato es **entender** al entrevistador y **ser entendido** por texto.
- **Decisión:** entregar **primero el teleprompter** (subtítulos ES+EN en `localhost`, con la fuente de cada voz), con el TTS **opcional y posterior**. El UI no lleva auth y no sintetiza voz en el navegador: es una vista local del flujo.
- **Por qué:**
  - **Semanas vs meses:** el teleprompter es un entregable útil en semanas; el TTS clonado llegó tras la Fase 2 completa.
  - **Honestidad en la entrevista:** el texto en pantalla es verificable; puedes corregir la respuesta antes de decirla, con o sin voz sintética.
- **Consecuencias:**
  - El entregable (FastAPI + WebSocket, sin auth, sin TTS en browser) está implementado en `src/traductor/ui/` y cubierto por `tests/test_ui.py`.
  - El flujo degrada a «solo subtítulos» cuando el TTS falla (ADR-008): el teleprompter es el nivel 4 de la escalera.
  - El pipeline real corre local (ADR-005); el proyecto **no publica un demo gestionado** — quien quiera usarlo clona el repo y lo corre en `localhost`.
- **Trazabilidad:** commits `af260eb` (propuesta del paso 5: teleprompter), `f0e7eb9` (implementación + tests), `4edac23` (merge del PR #8). Código: `src/traductor/ui/app.py`, `templates/teleprompter.html`, `static/style.css`; `tests/test_ui.py` (7 tests).
