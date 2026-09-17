"""Genera la voz del ENTREVISTADOR para la demo (preguntas cortas).

Dos preguntas cortas separadas por 2 s de silencio: el VAD del incoming
(`silencio_cierre_s=1.0`) cierra un turno entre frases y whisper tiny
transcribe fragmentos de ~3-4 s, el rango donde NO alucina. El WAV anterior
era 19.8 s de frases pegadas -> el VAD acumulaba hasta el corte de 4 s con la
frase a medias y tiny entraba en bucle (bug de la demo del video).
"""

from pathlib import Path

import numpy as np
import soundfile as sf
from huggingface_hub import hf_hub_download
from TTS.api import TTS

SALIDA = Path("scripts/audio/entrevistador_en.wav")
PAUSA_S = 2.0
RATE = 24000

FRASES = (
    "Welcome! Tell me about a hard bug you solved.",
    "What would you improve in that system today?",
)

tts = TTS("tts_models/multilingual/multi-dataset/xtts_v2", gpu=True)
voz = hf_hub_download("coqui/XTTS-v2", "samples/en_sample.wav", repo_type="model")

bloques: list[np.ndarray] = []
for i, frase in enumerate(FRASES):
    tmp = SALIDA.with_name(f"entrevistador_{i}.wav")
    tts.tts_to_file(text=frase, file_path=str(tmp), speaker_wav=[voz], language="en")
    muestras, sr = sf.read(tmp)
    bloques.append(muestras)
    tmp.unlink(missing_ok=True)
    if i < len(FRASES) - 1:
        bloques.append(np.zeros(int(PAUSA_S * sr), dtype=muestras.dtype))

final = np.concatenate(bloques)
sf.write(SALIDA, final, sr)
duracion = len(final) / sr
print(f"entrevistador -> {SALIDA} ({SALIDA.stat().st_size} bytes, {duracion:.1f}s)")
