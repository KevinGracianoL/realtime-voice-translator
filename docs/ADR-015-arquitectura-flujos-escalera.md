# ADR-015 - Arquitectura por flujos y escalera de presupuesto (Propuesto)

- **Estado:** Aceptado (2026-09-15) — dirección de arquitectura decidida en el PR #16; **ambos flujos implementados**: `outgoing_es_to_en` (PRs #22/#24) e `incoming_en_to_es` (PR #25, subtítulos primero; el clon dinámico queda como extensión opcional pendiente de los controles del ADR-015).
- **Contexto:** el prototipo lineal (audio → VAD → ASR → traducción → TTS → audio) no distingue las dos direcciones reales de la llamada, ni protege contra el artefacto que sale al micrófono virtual. La decisión de producción: **el requisito importante es que la otra persona oiga la VOZ de Kevin hablando inglés**; la voz del entrevistador en español arranca por subtítulos y el clon dinámico es opcional.
- **Decisión propuesta:** sustituir el flujo lineal por dos flujos explícitos con workers persistentes separados:

  - **`outgoing_es_to_en`** (hablar): micrófono → Moonshine Small ES (CPU) → Argos ES→EN (CPU) → teleprompter → **XTTS con perfil de Kevin** (GPU) → VB-CABLE → Meet/Zoom.
  - **`incoming_en_to_es`** (escuchar): audio remoto → Moonshine Small EN (CPU) → Argos EN→ES → **subtítulos locales**; clon de la voz del entrevistador solo si las muestras recogidas pasan los controles.

- **Reglas de flujo:**
  - Los **parciales** del ASR solo aparecen en pantalla; traducción y TTS reciben únicamente **segmentos finales**.
  - **Colas de tamaño 1** por dirección: una respuesta atrasada se descarta, no se pronuncia.
  - **Cancelación:** una solicitud nueva cancela la antigua (interrupción de síntesis al empezar un nuevo turno).
  - El audio sintetizado **no vuelve a entrar al micrófono**; botón de corte inmediato; fallback a texto o voz fija.
  - **Timestamps por etapa** (reloj inyectable, patrón de `latencia.medidor`).
    - **Precarga obligatoria en el arranque (requisito BLOQUEANTE, evidencia ADR-014):** el modelo spacy mwt que argos descarga on-demand debe estar precargado y la traduccion es->en debe validarse sin red ANTES de aceptar una llamada (1 de 20 corridas del harness murio por esa descarga). Un worker de traduccion que no pasa el arranque offline NO entra al flujo.
  - **Watchdog por worker** y degradación automática **sin reiniciar la llamada**.
  - Distribución de hardware: Ryzen (Moonshine/Argos/VAD/app) · GTX 1650 Ti (exclusivamente XTTS) · Radeon integrada (navegador/cámara/Meet) · RAM (modelos y workers).
- **Escalera de presupuesto (degradación sin reinicio):**
  1. Completo: voz de Kevin en inglés + voz del entrevistador en español.
  2. **Recomendado:** voz de Kevin en inglés + subtítulos en español.
  3. Respaldo: voz inglesa genérica (Supertonic/Piper) + subtítulos.
  4. Mínimo: subtítulos en ambos sentidos.
  - Nota de VRAM: al bajar de nivel, `cerrar()` del backend suelta el modelo pero el **allocator de torch puede retener el pool** (la VRAM no vuelve al driver automáticamente). Si la bajada necesita VRAM libre, usar `torch.cuda.empty_cache()` (best-effort) o reiniciar el worker del TTS.
- **Validación de artefactos (antes de enviar audio al micrófono virtual):**
  - **ASR de retorno** sobre el audio sintetizado; comparar lo reconocido con la traducción original; detectar palabras añadidas, omitidas o repetidas.
  - Rechazar **clipping**, **silencios anómalos** y **duración absurda**.
  - Verificar que la **cola siga vigente** (no una respuesta antigua).
  - Si el audio no pasa o la validación tarda demasiado: voz neutra, subtítulos o cancelar el turno. **Nunca reproducir audio sospechoso solo para mantener la clonación.**
- **Presupuesto de latencia (después de terminar de hablar):** cierre de turno ≤350 ms · ASR ≤300 ms · traducción ≤200 ms · primer fragmento XTTS ≤350 ms · ruteo ≤100 ms → **~1.3 s**, con margen dentro del objetivo de 1.5–2 s (ADR-003).
- **Voz del entrevistador (clon dinámico):** requiere ~15–30 s útiles de su voz (sin solapamiento, sin música/ruido, con autorización), calculadas una vez; Meet/Zoom ya comprime y altera el audio (el clon puede copiar esas imperfecciones) y no hay prueba previa con cientos de frases. Por eso: subtítulos primero → se acumulan fragmentos limpios → perfil → prueba interna → se habilita solo si pasa los gates → ante cualquier fallo, vuelve a subtítulos. Perfil temporal cifrado/protegido localmente con borrado automático al terminar.
- **Consecuencias:**
  - El flujo ES→EN (el requisito principal) se integra primero; el EN→ES con clon remoto después.
  - Los contratos del ADR-011 y el worker del ADR-013 son las interfaces intercambiables que esto necesita (cambiar ASR/TTS sin reconstruir).
  - Este ADR define el destino; los PRs de flujo se escriben contra él y contra los ADR-011/013/014.