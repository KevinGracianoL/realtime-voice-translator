"""Filtro de alucinaciones de whisper (puro y testeable).

faster-whisper (y whisper en general) ALUCINA cuando la entrada es silencio,
ruido de fondo o un fragmento demasiado corto: en vez de devolver vacío,
"rellena" con frases de su corpus de entrenamiento — subtítulos de YouTube,
créditos, muletillas. En este proyecto eso se ve como "whisper habla por mí y
dice frases raras" (bug reportado): el VAD por energía del cable deja pasar un
chunk que supera el umbral de RMS pero que NO es voz real, y whisper lo
transcribe como una de estas frases fantasma.

Este módulo NO reemplaza el `vad_filter` interno de faster-whisper (que se
activa en la llamada `transcribe`, ver `adaptadores.py`): es la SEGUNDA capa,
pura y sin hardware, que descarta las frases fantasma que sí llegan a pasar el
VAD. Al ser una función pura sobre texto se testea con arrays sintéticos y se
cubre al 100% en CI, a diferencia del código que toca el modelo.

La lista se curó de las alucinaciones documentadas del modelo `tiny` en inglés
(las que más aparecen con este cable): créditos de subtítulos, agradecimientos
de cierre de video y muletillas de un solo token. La comparación es sobre el
texto NORMALIZADO (minúsculas, sin puntuación ni espacios de más).
"""

from __future__ import annotations

# Frases fantasma EXACTAS (ya normalizadas): las alucinaciones de cierre de
# video/subtítulos que el modelo tiny emite sobre silencio o eco.
_FRASES_FANTASMA: frozenset[str] = frozenset(
    {
        "thank you",
        "thank you very much",
        "thanks for watching",
        "thank you for watching",
        "thank you for watching this video",
        "thanks for watching this video",
        "please subscribe",
        "please subscribe to my channel",
        "like and subscribe",
        "dont forget to subscribe",
        "see you next time",
        "see you in the next video",
        "bye",
        "bye bye",
        "goodbye",
        "you",
        "the",
        "so",
        "okay",
        "ok",
        "yeah",
        "hmm",
        "mm",
        "mm hmm",
        "uh",
        "um",
        "subtitles by the amaraorg community",
        "subtitles by the amara org community",
        "transcription by castingwords",
        "im not sure",
        "i dont know",
    }
)

# Tokens sueltos (un solo token) que casi siempre son ruido/relleno del modelo.
_TOKENS_RELLENO: frozenset[str] = frozenset(
    {"you", "the", "so", "okay", "ok", "yeah", "hmm", "mm", "uh", "um", "a", "i", "it"}
)


def _normalizar(texto: str) -> str:
    """Minúsculas, sin puntuación, espacios colapsados (igual que los flujos)."""
    return " ".join("".join(c for c in texto.lower() if c.isalnum() or c.isspace()).split())


def es_alucinacion(texto: str, *, min_tokens: int = 1) -> bool:
    """True si `texto` es (con alta probabilidad) una alucinación de whisper.

    Reglas, en orden:
    1. Vacío o solo puntuación/espacios tras normalizar → descartar.
    2. Coincide EXACTO con una frase fantasma conocida → descartar.
    3. Es un único token de relleno ("you", "the", "so"...) → descartar.
    4. Menos de `min_tokens` tokens → descartar (default 1 = no filtra por
       longitud; una respuesta real de una palabra como "Correct"/"Exactly"
       pasa. Un caller puede subirlo para exigir intervenciones más largas).
    5. Repetición patológica: un mismo token ocupa > 60% del texto y aparece
       3+ veces (p. ej. "you you you you") → descartar.

    Un texto que pasa las cinco reglas se considera habla real y se muestra.
    """
    norm = _normalizar(texto)
    if not norm:
        return True
    if norm in _FRASES_FANTASMA:
        return True
    tokens = norm.split()
    if len(tokens) == 1 and tokens[0] in _TOKENS_RELLENO:
        return True
    if len(tokens) < min_tokens:
        return True
    conteo = {t: tokens.count(t) for t in set(tokens)}
    veces = max(conteo.values())
    return veces >= 3 and veces / len(tokens) > 0.6
