"""Tests del filtro de alucinaciones de whisper (puro, sin hardware)."""

from traductor.asr.alucinacion import es_alucinacion


def test_vacio_es_alucinacion() -> None:
    assert es_alucinacion("") is True
    assert es_alucinacion("   ") is True
    assert es_alucinacion("...!?") is True


def test_frases_fantasma_conocidas() -> None:
    assert es_alucinacion("Thank you.") is True
    assert es_alucinacion("Thanks for watching!") is True
    assert es_alucinacion("Please subscribe to my channel") is True
    assert es_alucinacion("Subtitles by the Amara.org community") is True


def test_token_de_relleno_suelto() -> None:
    assert es_alucinacion("You") is True
    assert es_alucinacion("the") is True
    assert es_alucinacion("So") is True
    # tokens de relleno que NO están en la lista de frases fantasma exactas
    assert es_alucinacion("it") is True
    assert es_alucinacion("a") is True


def test_fragmento_demasiado_corto() -> None:
    # con min_tokens explícito, un fragmento corto se descarta
    assert es_alucinacion("hello", min_tokens=2) is True
    # con el default (min_tokens=1) una palabra real pasa
    assert es_alucinacion("correct") is False


def test_repeticion_patologica() -> None:
    assert es_alucinacion("you you you you") is True
    assert es_alucinacion("the the the the the") is True


def test_frase_repetida_ngramas() -> None:
    """Bug real de la demo: whisper repite la MISMA frase sobre audio mezclado."""
    catch = "I'm going to catch you"
    assert es_alucinacion(f"{catch}, {catch}, {catch}") is True
    assert es_alucinacion("okay so okay so okay so") is True
    assert es_alucinacion("no no no") is True
    # n-grama de longitud 1 repetido 3x con basura alrededor (no pasa por la
    # regla del token dominante: 3/5 = 0.6 <= 0.6): depende SOLO de la 2a capa
    assert es_alucinacion("no no no yes yes") is True


def test_frase_repetida_robusta_a_basura() -> None:
    """El bucle de whisper con basura alrededor: 3 copias + cola distinta.

    DeepSeek midió 'I don't know what you're talking about' x3 con tiny sobre
    un fragmento de 12.5 s. El filtro debe atraparlo aunque whisper varíe el
    final (no exige que TODO el texto sea copias exactas)."""
    frase = "I don't know what you're talking about"
    assert es_alucinacion(f"{frase} {frase} {frase}") is True
    assert es_alucinacion(f"{frase} {frase} {frase} yeah now") is True
    # con solo DOS copias no se marca (puede ser énfasis real): exige 3+
    assert es_alucinacion(f"{frase} {frase}") is False


def test_frase_repetida_no_afecta_habla_real() -> None:
    """Una intervención real NO es k copias exactas de un bloque."""
    assert es_alucinacion("tell me tell me about your last project") is False
    assert es_alucinacion("what what is your experience") is False
    assert es_alucinacion("very very good candidate for the role") is False


def test_habla_real_pasa() -> None:
    assert es_alucinacion("Tell me about your experience with distributed systems") is False
    assert es_alucinacion("What is your greatest weakness?") is False
    assert es_alucinacion("Why do you want to work here") is False


def test_frase_con_thank_you_dentro_no_es_fantasma() -> None:
    # "thank you" EXACTO es fantasma, pero dentro de una frase real no:
    assert es_alucinacion("Thank you for taking the time to interview me today") is False


def test_min_tokens_configurable() -> None:
    assert es_alucinacion("go now", min_tokens=3) is True
    assert es_alucinacion("go right now", min_tokens=3) is False


def test_repeticion_frontera_veces_exactas() -> None:
    # veces == 3 es el MÍNIMO de la regla: los mutantes `> 3` y `>= 4`
    # (veces >= 3 -> veces > 3 / veces >= 4) caen aquí
    assert es_alucinacion("you you you") is True
    assert es_alucinacion("you you you the") is True


def test_repeticion_veces_tres_sin_consecutivas() -> None:
    # veces == 3 y > 60% SIN 3+ consecutivas: la 2ª capa NO lo marca, así que
    # la regla del token dominante es la que decide — caza `>= 3`->`>= 4` y
    # `> 0.6`->`> 1.6` (con consecutivas la 2ª capa los enmascararía)
    assert es_alucinacion("you you the you") is True


def test_repeticion_frontera_60_porciento_exacto() -> None:
    # 3/5 == 0.6 EXACTO sin repetición CONSECUTIVA: NO es patológico (regla
    # estricta `> 0.6`); caza los mutantes `>= 0.6` y `/` -> `*`. (Con 3
    # consecutivas la 2ª capa del n-grama sí lo marca: ver
    # test_frase_repetida_ngramas.)
    assert es_alucinacion("you there you here you") is False
    assert es_alucinacion("you the you me") is False
