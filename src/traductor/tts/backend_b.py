"""Backend B: Supertonic 3 (base EN, CPU) + OpenVoice V2 (timbre de Kevin).

Implementa `TTSBackend` (ADR-011), candidato B del ADR-011/014. Pipeline por
turno: Supertonic 3 (ONNX, CPU) genera el inglés limpio y OpenVoice V2
convierte el timbre al de `perfil`. Ambos modelos corren en CPU: la VRAM queda
libre para el ASR (gate de co-residencia del ADR-014).

Los imports de motor son lazy: en CI `supertonic`/`openvoice` no están
instalados y `verificar_salud()` lo reporta SIN cargar modelos. Las líneas que
tocan las APIs reales van `# pragma: no cover`: se cubren con fakes fieles en
los tests y el motor real corre en el harness del ADR-014 (máquina objetivo).

`perfil.muestras` son las referencias de voz pre-enroladas (ADR-011): OpenVoice
extrae el timbre (`target_se`) una vez por perfil y lo cachea. El `src_se`
(timbre de la voz base M1) se extrae del propio audio de Supertonic, también
una vez — el conversor necesita saber de qué timbre parte.
"""

from __future__ import annotations

import shutil
import tempfile
from pathlib import Path
from typing import Any

from traductor.tts.modelos import AudioResult, Salud, VoiceProfile

VOZ_SUPERTONIC = "M1"  # voz fija del generador base (10 presets: M1-M5, F1-F5)
TEXTO_PROBE = (
    "This is a voice profile probe. It contains enough speech for the style extractor to work."
)


class BackendB:
    """TTSBackend: Supertonic 3 + OpenVoice V2. `idioma_salida`: "en" (ADR-015)."""

    def __init__(
        self,
        *,
        dir_checkpoints: str,
        voz: str = VOZ_SUPERTONIC,
        device: str = "cpu",
        idioma_salida: str = "en",
    ) -> None:
        self._dir_checkpoints = Path(dir_checkpoints)
        config = self._dir_checkpoints / "converter" / "config.json"
        ckpt = self._dir_checkpoints / "converter" / "checkpoint.pth"
        if not config.is_file() or not ckpt.is_file():
            raise RuntimeError(
                f"pesos de OpenVoice V2 incompletos en {self._dir_checkpoints}: "
                "faltan converter/config.json o converter/checkpoint.pth "
                "(define OPENVOICE_CHECKPOINTS_V2)"
            )
        self._voz = voz
        self._device = device
        self._idioma_salida = idioma_salida
        self._tts: Any | None = None
        self._estilo: Any | None = None
        self._conversor: Any | None = None
        self._src_se_cache: Any | None = None
        self._se_por_perfil: dict[tuple[str, tuple[str, ...]], Any] = {}

    def _cargar_supertonic(self) -> Any:  # pragma: no cover - requiere supertonic
        """Carga el generador base una sola vez (lazy). Excepciones: RuntimeError."""
        if self._tts is None:
            try:
                from supertonic import TTS
            except ImportError as exc:
                raise RuntimeError(
                    "supertonic no instalado: pip install supertonic "
                    "(venv propio del TTS, ver ADR-011/014)"
                ) from exc
            try:
                self._tts = TTS(auto_download=True)
                self._estilo = self._tts.get_voice_style(voice_name=self._voz)
            except Exception as exc:
                raise RuntimeError(f"Supertonic 3 no cargó: {exc}") from exc
        return self._tts

    def _cargar_conversor(self) -> Any:  # pragma: no cover - requiere openvoice
        """Carga el convertidor de timbre una sola vez (lazy). Excepciones: RuntimeError.

        El repo de OpenVoice no permite desactivar el watermark en el
        constructor (r1 PR #18): se anula tras cargar (no necesitamos la
        marca "@MyShell" en el audio del usuario).
        """
        if self._conversor is None:
            try:
                from openvoice.api import ToneColorConverter
            except ImportError as exc:
                raise RuntimeError(
                    "openvoice no instalado: pip install -e <repo OpenVoice> "
                    "--no-deps + pesos en checkpoints_v2 (ver ADR-014)"
                ) from exc
            try:
                conv = ToneColorConverter(
                    str(self._dir_checkpoints / "converter" / "config.json"),
                    device=self._device,
                )
                conv.watermark_model = None
                conv.load_ckpt(str(self._dir_checkpoints / "converter" / "checkpoint.pth"))
            except Exception as exc:
                raise RuntimeError(f"OpenVoice V2 no cargó: {exc}") from exc
            self._conversor = conv
        return self._conversor

    def _extraer_se(self, ruta: Path, conv: Any, dir_trabajo: Path) -> Any:  # pragma: no cover
        """Timbre de un WAV vía OpenVoice (VAD silero: rápido, no transcribe)."""
        from openvoice import se_extractor

        se, _ = se_extractor.get_se(str(ruta), conv, target_dir=str(dir_trabajo), vad=True)
        return se

    def _src_se(self, conv: Any, dir_trabajo: Path) -> Any:  # pragma: no cover - una vez
        """Timbre de la voz base M1, extraído del propio audio de Supertonic."""
        if self._src_se_cache is None:
            tts = self._cargar_supertonic()
            wav, _ = tts.synthesize(TEXTO_PROBE, voice_style=self._estilo, lang=self._idioma_salida)
            probe = dir_trabajo / "src_probe.wav"
            tts.save_audio(wav, str(probe))
            self._src_se_cache = self._extraer_se(probe, conv, dir_trabajo)
        return self._src_se_cache

    def _target_se(self, perfil: VoiceProfile, conv: Any, dir_trabajo: Path) -> Any:
        """Timbre de `perfil`: cacheado por (id, muestras).

        La clave incluye las muestras: si el perfil se re-enrola con
        grabaciones nuevas (worker de vida larga, ADR-013), el timbre se
        recalcula en vez de devolver el viejo en silencio. Una muestra: su
        timbre directo; varias: media de embeddings torch (ADR-011 pre-enrola
        3-5 grabaciones) — se cubre en la máquina objetivo, no en CI.
        """
        clave = (perfil.id, perfil.muestras)
        if clave not in self._se_por_perfil:
            ses = [self._extraer_se(Path(m), conv, dir_trabajo) for m in perfil.muestras]
            if len(ses) == 1:
                self._se_por_perfil[clave] = ses[0]
            else:  # pragma: no cover - media torch: solo con 2+ muestras reales
                import torch

                self._se_por_perfil[clave] = torch.stack(ses).mean(0)
        return self._se_por_perfil[clave]

    def sintetizar(self, texto: str, perfil: VoiceProfile) -> AudioResult:
        """Sintetiza `texto` con el timbre de `perfil.muestras`.

        Supertonic habla el texto (idioma de salida) y OpenVoice convierte el
        timbre al del perfil. Excepciones: RuntimeError si el motor no está
        disponible (ver `verificar_salud`).
        """
        tts = self._cargar_supertonic()
        conv = self._cargar_conversor()
        wav, _ = tts.synthesize(texto, voice_style=self._estilo, lang=self._idioma_salida)
        dir_trabajo = Path(tempfile.mkdtemp(prefix="backend-b-"))
        try:
            src = dir_trabajo / "src.wav"
            tts.save_audio(wav, str(src))
            out = dir_trabajo / "out.wav"
            conv.convert(
                audio_src_path=str(src),
                src_se=self._src_se(conv, dir_trabajo),
                tgt_se=self._target_se(perfil, conv, dir_trabajo),
                output_path=str(out),
            )
            import wave

            with wave.open(str(out)) as w:  # modo lectura por defecto (rb)
                duracion_s = w.getnframes() / w.getframerate()
            return AudioResult(datos=out.read_bytes(), formato="wav", duracion_s=duracion_s)
        finally:
            shutil.rmtree(dir_trabajo, ignore_errors=True)

    def verificar_salud(self) -> Salud:
        """Estado SIN efectos secundarios: no carga los modelos.

        Solo comprueba instalación (supertonic primero) y si ya están cargados
        — no contamina las fotos de VRAM/RAM del harness por orden de llamadas.
        """
        if self._tts is None or self._conversor is None:
            try:
                import supertonic  # noqa: F401 - solo comprueba instalación
            except ImportError as exc:
                return Salud(disponible=False, detalle=f"supertonic no instalado: {exc}")
            try:
                import openvoice  # noqa: F401 - solo comprueba instalación
            except ImportError as exc:
                return Salud(disponible=False, detalle=f"openvoice no instalado: {exc}")
            return Salud(  # pragma: no cover - solo con ambos instalados
                disponible=False, detalle="modelos no cargados: llama a sintetizar() primero"
            )
        return Salud(disponible=True, detalle=f"Supertonic 3 + OpenVoice V2 listos ({self._voz})")

    def cerrar(self) -> None:
        """Suelta los modelos y limpia caches. Idempotente."""
        self._tts = None
        self._estilo = None
        self._conversor = None
        self._src_se_cache = None
        self._se_por_perfil = {}
