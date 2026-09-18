# ADR-002 - Audio virtual a nivel de sistema operativo

- **Estado:** Aceptado (2026-09-01, paso 4) — **reconstruido el 2026-09-18** desde los commits y el código (la era inicial no escribió el archivo).
- **Contexto:** el traductor necesita que su salida de audio (la voz en inglés del TTS, y los subtítulos por separado) llegue a **cualquier** plataforma de videollamada (Meet, Zoom, Teams) sin integraciones por plataforma.
- **Decisión:** exponer **dispositivos de audio virtuales a nivel de sistema operativo** (VB-CABLE; más tarde VoiceMeeter para el segundo tubo) y enrutar por **nombre de dispositivo**, de forma determinista. Nada de APIs de plataforma ni de compartir pantalla con el audio del navegador como mecanismo principal.
- **Por qué:** un dispositivo virtual funciona con **cualquier** aplicación que vea micrófonos/altavoces del sistema — coste cero de integración y cero dependencia de que una plataforma abra su API de audio.
- **Consecuencias:**
  - `src/traductor/audio/virtual.py` detecta y clasifica dispositivos (virtual vs físico) y elige la ruta determinista por nombre; cubierto por `tests/test_virtual.py`.
  - La Fase 5 añade el **segundo tubo** (VoiceMeeter) porque un solo cable mezclaba el TTS con la voz del entrevistador (retroalimentación física, ver README «Dos tubos»).
  - Los índices de pyaudio no son estables (se re-enumeran al conectar hardware): todo se busca por nombre (`_buscar_device`), nunca por índice fijo.
- **Trazabilidad:** commit `a05e6a4` (paso 4, «audio virtual Windows — ruta determinista»), rama `paso4-audio-virtual`. Código: `audio/virtual.py` (cita «ADR-002» en su cabecera), `tests/test_virtual.py` (14 tests). Citado por el antiguo ADR-009 («la misma idea del ADR-002: enrutar por dispositivo de audio a nivel de SO, no por una API que interpreta»).
