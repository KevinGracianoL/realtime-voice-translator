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


def test_repeticion_frontera_60_porciento_exacto() -> None:
    # 3/5 == 0.6 EXACTO: no es patológico (regla estricta `> 0.6`); los
    # mutantes `>= 0.6` y `/` -> `*` caen aquí
    assert es_alucinacion("you you you the so") is False
