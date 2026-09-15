"""Contrato neutral de un backend TTS.

Un `TTSBackend` sintetiza texto a audio para un `VoiceProfile`. El pipeline
depende de ESTE protocolo, no de un proveedor concreto: el worker XTTS y los
gates de latencia (PRs siguientes) se escriben contra él.
"""

from __future__ import annotations

from collections.abc import Iterator
from typing import Protocol, runtime_checkable

from traductor.tts.modelos import AudioResult, Salud, VoiceProfile


@runtime_checkable
class TTSBackend(Protocol):
    """Backend de síntesis de voz. Ciclo de vida explícito: `cerrar()`.

    `sintetizar_stream` NO es parte del contrato base: un backend sin
    streaming se usa solo con `sintetizar` (el worker lo detecta con
    `isinstance(backend, TTSBackendStream)` y cae al camino completo).
    """

    def sintetizar(self, texto: str, perfil: VoiceProfile) -> AudioResult:
        """Sintetiza `texto` con el timbre de `perfil`.

        Raises:
            RuntimeError: si el backend no está disponible (ver `verificar_salud`).
        """
        ...

    def verificar_salud(self) -> Salud:
        """True si el backend está listo para sintetizar (modelo cargado, GPU ok)."""
        ...

    def cerrar(self) -> None:
        """Libera recursos (modelo, VRAM). Idempotente: llamar dos veces no falla."""
        ...


@runtime_checkable
class TTSBackendStream(Protocol):
    """Backend con streaming (ADR-019, fix 1): sintetiza en CHUNKS.

    El primero llega rápido (~0.7 s en XTTS); el worker emite una línea de
    resultado por chunk y un `fin` (protocolo streaming del worker).
    """

    def sintetizar_stream(self, texto: str, perfil: VoiceProfile) -> Iterator[AudioResult]:
        """Sintetiza `texto` por chunks; el generador se cierra al terminar.

        Raises:
            RuntimeError: si el backend no está disponible.
        """
        ...
