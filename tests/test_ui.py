"""Tests para UI teleprompter — sin levantar server real."""

from fastapi.testclient import TestClient

from traductor.ui.app import app


def test_root_sirve_html() -> None:
    client = TestClient(app)
    res = client.get("/")
    assert res.status_code == 200
    assert "Teleprompter" in res.text
    assert "feed" in res.text


def test_post_transcripcion_broadcast() -> None:
    client = TestClient(app)
    # Conecta un WS para recibir broadcast
    with client.websocket_connect("/ws") as ws:
        # POST fallback debe hacer broadcast al WS
        res = client.post("/api/transcripcion", json={"en": "hello", "es": "hola"})
        assert res.status_code == 200
        assert res.json() == {"ok": "true"}
        data = ws.receive_text()
        assert '"en": "hello"' in data or '"en":"hello"' in data.replace(" ", "")


def test_post_transcripcion_sin_ws_no_falla() -> None:
    client = TestClient(app)
    res = client.post("/api/transcripcion", json={"en": "hi", "es": "hola"})
    assert res.status_code == 200


def test_post_transcripcion_invalida_422() -> None:
    client = TestClient(app)
    res = client.post("/api/transcripcion", json={"en": "hi"})
    assert res.status_code == 422


def test_fuente_default_es_yo() -> None:
    """Sin 'fuente', el default es 'yo' (retrocompatible con el outgoing)."""
    client = TestClient(app)
    with client.websocket_connect("/ws") as ws:
        client.post("/api/transcripcion", json={"en": "hello", "es": "hola"})
        data = ws.receive_text()
        assert '"fuente":"yo"' in data.replace(" ", "")


def test_fuente_entrevistador_se_propaga() -> None:
    """El flujo incoming manda fuente='entrevistador' y llega al WS."""
    client = TestClient(app)
    with client.websocket_connect("/ws") as ws:
        client.post(
            "/api/transcripcion",
            json={"en": "your experience", "es": "tu experiencia", "fuente": "entrevistador"},
        )
        data = ws.receive_text()
        assert '"fuente":"entrevistador"' in data.replace(" ", "")


def test_broadcast_descarta_ws_muerto() -> None:
    """WS que falla en send_text debe ser removido de conexiones."""
    from unittest.mock import AsyncMock

    from traductor.ui.app import Transcripcion, broadcast, conexiones

    conexiones.clear()
    ws_falla = AsyncMock()
    ws_falla.send_text.side_effect = Exception("desconectado")
    conexiones.add(ws_falla)

    import asyncio

    asyncio.run(broadcast(Transcripcion(en="hi", es="hola")))

    assert ws_falla not in conexiones
    conexiones.clear()
