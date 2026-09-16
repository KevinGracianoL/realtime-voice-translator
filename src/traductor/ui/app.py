"""Teleprompter en vivo — FastAPI + WebSocket en localhost.

Corre en tu laptop: el pipeline local envía transcripciones vía WS o POST,
el browser muestra ES+EN con auto-scroll inteligente.
"""

from __future__ import annotations

from pathlib import Path

from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel


class Transcripcion(BaseModel):
    en: str
    es: str
    fuente: str = "yo"  # "yo" (mi voz→EN) o "entrevistador" (su voz→ES)


app = FastAPI(title="Teleprompter Traductor")

# Maneja conexiones WS activas
conexiones: set[WebSocket] = set()


@app.get("/", response_class=HTMLResponse)
async def teleprompter() -> str:
    """Sirve el HTML del teleprompter."""
    html_path = Path(__file__).parent / "templates" / "teleprompter.html"
    return html_path.read_text(encoding="utf-8")


@app.websocket("/ws")
async def ws_endpoint(ws: WebSocket) -> None:
    """WebSocket para push en tiempo real."""
    await ws.accept()
    conexiones.add(ws)
    try:
        while True:
            # Mantener conexión abierta; el server hace broadcast, no recibe
            await ws.receive_text()
    except WebSocketDisconnect:
        pass
    finally:
        conexiones.discard(ws)


@app.post("/api/transcripcion")
async def post_transcripcion(data: Transcripcion) -> dict[str, str]:
    """Fallback POST — mismo efecto que WS, para cuando WS no está disponible."""
    await broadcast(data)
    return {"ok": "true"}


async def broadcast(data: Transcripcion) -> None:
    """Envía a todos los WS conectados."""
    mensaje = data.model_dump_json()
    muertos: list[WebSocket] = []
    for ws in conexiones:
        try:
            await ws.send_text(mensaje)
        except Exception:
            muertos.append(ws)
    for ws in muertos:
        conexiones.discard(ws)


# Montar static si existe (para CSS)
static_dir = Path(__file__).parent / "static"
if static_dir.exists():
    app.mount("/static", StaticFiles(directory=static_dir), name="static")
