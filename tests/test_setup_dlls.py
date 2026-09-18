"""Tests de setup_dlls — corren siempre (tmp_path, sin venv real)."""

import runpy
import sys
import time
import types
from pathlib import Path

import pytest

from setup_dlls import dirs_cuda, escribir_keeper, instalar, plan_dlls


class _Handle:
    """Handle falso observable (os.add_dll_directory no existe en Linux)."""

    def __init__(self, path: str) -> None:
        self.path = path


class _FakeOs(types.ModuleType):
    """os real salvo add_dll_directory, que registra y devuelve handle observable."""

    def __init__(self, real_os: types.ModuleType) -> None:
        super().__init__("os")
        self.__dict__["_real_os"] = real_os
        self.registrados: list[str] = []

    def __getattr__(self, name: str) -> object:
        return getattr(self.__dict__["_real_os"], name)

    def add_dll_directory(self, path: str) -> _Handle:
        self.registrados.append(path)
        return _Handle(path)


def _fake_os() -> _FakeOs:
    import os as real_os

    return _FakeOs(real_os)


def test_dirs_cuda_ordena_destino_primero(tmp_path: Path) -> None:
    import os

    bins = dirs_cuda("/nvidia", str(tmp_path))
    assert bins[0] == str(tmp_path)
    esperados = [os.path.join("/nvidia", sub, "bin") for sub in ("cublas", "cudnn", "cuda_runtime")]
    assert bins[1:] == esperados


def test_plan_dlls_destino_y_solo_3_a_ct2(tmp_path: Path) -> None:
    archivos = {"cublas": ["a.dll", "x.txt"], "cudnn": ["b.dll"], "cuda_runtime": ["c.dll"]}
    plan = plan_dlls("/nvidia", str(tmp_path), str(tmp_path / "ct2"), archivos)
    a_dest = sorted(Path(src).name for src, dst in plan if dst == str(tmp_path))
    assert a_dest == ["a.dll", "b.dll", "c.dll"]
    a_ct2 = sorted(Path(src).name for src, dst in plan if dst == str(tmp_path / "ct2"))
    assert a_ct2 == ["cublas64_12.dll", "cublasLt64_12.dll", "cudart64_12.dll"]


def test_escribir_keeper_contenido(tmp_path: Path) -> None:
    keeper, pth = escribir_keeper(str(tmp_path), ["D1", "D2"])
    texto = Path(keeper).read_text(encoding="utf-8")
    compile(texto, keeper, "exec")
    assert "_HANDLES = [os.add_dll_directory(p) for p in _DIRS]" in texto
    assert "'D1'" in texto and "'D2'" in texto
    assert Path(pth).read_text(encoding="utf-8").strip() == "import zz_nvidia_dlls"
    assert Path(keeper).parent == tmp_path
    assert Path(pth).parent == tmp_path


def test_keeper_ejecuta_y_conserva_handles(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """El keeper generado registra cada dir y conserva los handles (también en Linux)."""
    sub = tmp_path / "bin"
    sub.mkdir()
    keeper, _ = escribir_keeper(str(tmp_path), [str(sub)])
    fake_os = _fake_os()
    monkeypatch.setitem(sys.modules, "os", fake_os)
    ns = runpy.run_path(str(keeper))
    assert fake_os.registrados == [str(sub)]
    assert len(ns["_HANDLES"]) == 1
    assert ns["_HANDLES"][0].path == str(sub)


def test_instalar_copia_y_registra_en_arbol_falso(tmp_path: Path) -> None:
    """instalar() end-to-end contra tmp: copias reales + keeper ejecutable."""
    nvidia = tmp_path / "nvidia"
    for sub, files in {
        "cublas": ["a.dll", "cublas64_12.dll", "cublasLt64_12.dll", "x.txt"],
        "cudnn": ["b.dll"],
        "cuda_runtime": ["c.dll", "cudart64_12.dll"],
    }.items():
        bindir = nvidia / sub / "bin"
        bindir.mkdir(parents=True)
        for f in files:
            (bindir / f).write_bytes(b"fake-dll")
    dest = tmp_path / "Scripts"
    dest.mkdir()
    site_packages = tmp_path / "site-packages"
    site_packages.mkdir()
    ct2 = site_packages / "ctranslate2"
    ct2.mkdir()
    keeper, pth = instalar(str(nvidia), str(dest), str(site_packages), str(ct2))
    assert (dest / "a.dll").exists()
    assert not (dest / "x.txt").exists()
    assert (ct2 / "cublas64_12.dll").exists()
    assert not (ct2 / "b.dll").exists()
    assert Path(pth).read_text(encoding="utf-8").strip() == "import zz_nvidia_dlls"
    assert Path(keeper).exists()
    assert Path(keeper).parent == site_packages
    assert Path(pth).parent == site_packages
    assert repr(str(dest)) in Path(keeper).read_text(encoding="utf-8")


def test_copiar_dlls_reemplaza_existente(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Re-ejecución: backup único, copy al destino, limpieza del .tmp."""
    import os
    import shutil

    from setup_dlls import copiar_dlls

    nvidia = tmp_path / "nvidia"
    (nvidia / "cublas" / "bin").mkdir(parents=True)
    (nvidia / "cublas" / "bin" / "cublas64_12.dll").write_bytes(b"origen")
    (nvidia / "cublas" / "bin" / "cublasLt64_12.dll").write_bytes(b"origen")
    (nvidia / "cudnn" / "bin").mkdir(parents=True)
    (nvidia / "cuda_runtime" / "bin").mkdir(parents=True)
    (nvidia / "cuda_runtime" / "bin" / "cudart64_12.dll").write_bytes(b"origen")
    dest = tmp_path / "Scripts"
    dest.mkdir()
    ct2 = tmp_path / "ct2"
    ct2.mkdir()
    (dest / "cublas64_12.dll").write_bytes(b"viejo")
    renames: list[tuple[str, str]] = []
    copias: list[tuple[str, str]] = []
    real_replace = os.replace
    real_copy2 = shutil.copy2

    def spy_replace(a: str, b: str) -> None:
        if a == str(dest / "cublas64_12.dll"):
            renames.append((a, b))
        real_replace(a, b)

    def spy_copy2(a: str, b: str) -> None:
        if b == str(dest / "cublas64_12.dll"):
            copias.append((a, b))
        real_copy2(a, b)

    monkeypatch.setattr(os, "replace", spy_replace)
    monkeypatch.setattr(shutil, "copy2", spy_copy2)
    copiar_dlls(str(nvidia), str(dest), str(ct2))
    assert (dest / "cublas64_12.dll").read_bytes() == b"origen"
    assert renames == [(str(dest / "cublas64_12.dll"), str(dest / "cublas64_12.dll.tmp"))]
    assert len(copias) == 1
    assert not (dest / "cublas64_12.dll.tmp").exists()
    assert (ct2 / "cublas64_12.dll").exists()


def test_copiar_dlls_sin_destino_previo(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Primera corrida: sin viejo no hay backup; copia directa al destino."""
    import os

    from setup_dlls import copiar_dlls

    nvidia = tmp_path / "nvidia"
    (nvidia / "cublas" / "bin").mkdir(parents=True)
    (nvidia / "cublas" / "bin" / "cublas64_12.dll").write_bytes(b"origen")
    (nvidia / "cublas" / "bin" / "cublasLt64_12.dll").write_bytes(b"origen")
    (nvidia / "cudnn" / "bin").mkdir(parents=True)
    (nvidia / "cuda_runtime" / "bin").mkdir(parents=True)
    (nvidia / "cuda_runtime" / "bin" / "cudart64_12.dll").write_bytes(b"origen")
    dest = tmp_path / "Scripts"
    dest.mkdir()
    ct2 = tmp_path / "ct2"
    ct2.mkdir()
    renames: list[tuple[str, str]] = []
    real_replace = os.replace

    def spy_replace(a: str, b: str) -> None:
        if a == str(dest / "cublas64_12.dll"):
            renames.append((a, b))
        real_replace(a, b)

    monkeypatch.setattr(os, "replace", spy_replace)
    copiar_dlls(str(nvidia), str(dest), str(ct2))
    assert (dest / "cublas64_12.dll").read_bytes() == b"origen"
    assert renames == []
    assert not (dest / "cublas64_12.dll.tmp").exists()


def test_copiar_dlls_reintenta_y_acierta(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Lock transitorio (AV): copy2 falla 2 veces y luego pasa, con backoff."""
    import shutil

    from setup_dlls import copiar_dlls

    nvidia = tmp_path / "nvidia"
    (nvidia / "cublas" / "bin").mkdir(parents=True)
    (nvidia / "cublas" / "bin" / "cublas64_12.dll").write_bytes(b"origen")
    (nvidia / "cublas" / "bin" / "cublasLt64_12.dll").write_bytes(b"origen")
    (nvidia / "cudnn" / "bin").mkdir(parents=True)
    (nvidia / "cuda_runtime" / "bin").mkdir(parents=True)
    (nvidia / "cuda_runtime" / "bin" / "cudart64_12.dll").write_bytes(b"origen")
    dest = tmp_path / "Scripts"
    dest.mkdir()
    ct2 = tmp_path / "ct2"
    ct2.mkdir()
    (dest / "cublas64_12.dll").write_bytes(b"bloqueado")
    real_copy2 = shutil.copy2
    intentos: list[str] = []
    duerme: list[float] = []

    def copy_que_se_desbloquea(src: str, dst: str) -> None:
        if dst == str(dest / "cublas64_12.dll"):
            intentos.append(dst)
            if len(intentos) <= 2:
                raise PermissionError(src)
        real_copy2(src, dst)

    def spy_sleep(s: float) -> None:
        duerme.append(s)

    monkeypatch.setattr(shutil, "copy2", copy_que_se_desbloquea)
    monkeypatch.setattr(time, "sleep", spy_sleep)
    copiar_dlls(str(nvidia), str(dest), str(ct2), reintentos=5, espera=0)
    assert (dest / "cublas64_12.dll").read_bytes() == b"origen"
    assert len(intentos) == 3
    assert duerme == [0.0, 0.0]


def test_copiar_dlls_agota_reintentos_y_conserva_vieja(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Lock permanente en copy: error claro y la DLL vieja queda intacta."""
    import shutil

    from setup_dlls import copiar_dlls

    nvidia = tmp_path / "nvidia"
    (nvidia / "cublas" / "bin").mkdir(parents=True)
    (nvidia / "cublas" / "bin" / "cublas64_12.dll").write_bytes(b"origen")
    (nvidia / "cublas" / "bin" / "cublasLt64_12.dll").write_bytes(b"origen")
    (nvidia / "cudnn" / "bin").mkdir(parents=True)
    (nvidia / "cuda_runtime" / "bin").mkdir(parents=True)
    (nvidia / "cuda_runtime" / "bin" / "cudart64_12.dll").write_bytes(b"origen")
    dest = tmp_path / "Scripts"
    dest.mkdir()
    ct2 = tmp_path / "ct2"
    ct2.mkdir()
    (dest / "cublas64_12.dll").write_bytes(b"VIEJA-PERO-FUNCIONA")
    intentos: list[str] = []

    def bloqueado(src: str, dst: str) -> None:
        intentos.append(dst)
        raise PermissionError(src)

    monkeypatch.setattr(shutil, "copy2", bloqueado)
    monkeypatch.setattr(time, "sleep", lambda _: None)
    with pytest.raises(RuntimeError, match="no puedo reemplazar"):
        copiar_dlls(str(nvidia), str(dest), str(ct2), reintentos=3, espera=0)
    assert (dest / "cublas64_12.dll").read_bytes() == b"VIEJA-PERO-FUNCIONA"
    assert len(intentos) == 3


def test_copiar_dlls_copy_sucio_no_pierde_backup(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Doble que escribe basura y luego revienta: el backup no se corrompe (revisión r5)."""
    import shutil

    from setup_dlls import copiar_dlls

    nvidia = tmp_path / "nvidia"
    (nvidia / "cublas" / "bin").mkdir(parents=True)
    (nvidia / "cublas" / "bin" / "cublas64_12.dll").write_bytes(b"origen")
    (nvidia / "cublas" / "bin" / "cublasLt64_12.dll").write_bytes(b"origen")
    (nvidia / "cudnn" / "bin").mkdir(parents=True)
    (nvidia / "cuda_runtime" / "bin").mkdir(parents=True)
    (nvidia / "cuda_runtime" / "bin" / "cudart64_12.dll").write_bytes(b"origen")
    dest = tmp_path / "Scripts"
    dest.mkdir()
    ct2 = tmp_path / "ct2"
    ct2.mkdir()
    (dest / "cublas64_12.dll").write_bytes(b"VIEJA-BUENA")

    def copy_sucio(src: str, dst: str) -> None:
        if dst == str(dest / "cublas64_12.dll"):
            with open(dst, "wb") as fh:
                fh.write(b"TRUNCADA")
        raise PermissionError(src)

    monkeypatch.setattr(shutil, "copy2", copy_sucio)
    monkeypatch.setattr(time, "sleep", lambda _: None)
    with pytest.raises(RuntimeError, match="no puedo reemplazar"):
        copiar_dlls(str(nvidia), str(dest), str(ct2), reintentos=3, espera=0)
    assert (dest / "cublas64_12.dll").read_bytes() == b"VIEJA-BUENA"
    assert not (dest / "cublas64_12.dll.tmp").exists()


def test_copiar_dlls_parcial_no_borrable(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """El parcial que tampoco se puede borrar no revienta el retry (revisión r3 del PR #14).

    `os.remove` del parcial lanza PermissionError (lock del AV): el
    `suppress(PermissionError)` lo absorbe y el flujo llega al error claro.
    Con `suppress(None)` (mutante) el TypeError mataría el retry.
    """
    import os
    import shutil

    from setup_dlls import copiar_dlls

    nvidia = tmp_path / "nvidia"
    (nvidia / "cublas" / "bin").mkdir(parents=True)
    (nvidia / "cublas" / "bin" / "cublas64_12.dll").write_bytes(b"origen")
    (nvidia / "cublas" / "bin" / "cublasLt64_12.dll").write_bytes(b"origen")
    (nvidia / "cudnn" / "bin").mkdir(parents=True)
    (nvidia / "cuda_runtime" / "bin").mkdir(parents=True)
    (nvidia / "cuda_runtime" / "bin" / "cudart64_12.dll").write_bytes(b"origen")
    dest = tmp_path / "Scripts"
    dest.mkdir()
    ct2 = tmp_path / "ct2"
    ct2.mkdir()
    (dest / "cublas64_12.dll").write_bytes(b"VIEJA-BUENA")
    real_remove = os.remove

    def copy_sucio(src: str, dst: str) -> None:
        if dst == str(dest / "cublas64_12.dll"):
            with open(dst, "wb") as fh:
                fh.write(b"TRUNCADA")
        raise PermissionError(src)

    def remove_parcial_bloqueado(path: str) -> None:
        if path == str(dest / "cublas64_12.dll"):
            raise PermissionError(path)
        real_remove(path)

    monkeypatch.setattr(shutil, "copy2", copy_sucio)
    monkeypatch.setattr(os, "remove", remove_parcial_bloqueado)
    monkeypatch.setattr(time, "sleep", lambda _: None)
    with pytest.raises(RuntimeError, match="no puedo reemplazar"):
        copiar_dlls(str(nvidia), str(dest), str(ct2), reintentos=3, espera=0)
    assert (dest / "cublas64_12.dll").read_bytes() == b"VIEJA-BUENA"


def test_copiar_dlls_backup_bloqueado_error_claro(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """El backup inicial falla (proceso con la DLL): error claro, vieja intacta."""
    import os

    from setup_dlls import copiar_dlls

    nvidia = tmp_path / "nvidia"
    (nvidia / "cublas" / "bin").mkdir(parents=True)
    (nvidia / "cublas" / "bin" / "cublas64_12.dll").write_bytes(b"origen")
    (nvidia / "cublas" / "bin" / "cublasLt64_12.dll").write_bytes(b"origen")
    (nvidia / "cudnn" / "bin").mkdir(parents=True)
    (nvidia / "cuda_runtime" / "bin").mkdir(parents=True)
    (nvidia / "cuda_runtime" / "bin" / "cudart64_12.dll").write_bytes(b"origen")
    dest = tmp_path / "Scripts"
    dest.mkdir()
    ct2 = tmp_path / "ct2"
    ct2.mkdir()
    (dest / "cublas64_12.dll").write_bytes(b"VIEJA-PERO-FUNCIONA")

    def bloqueado(a: str, b: str) -> None:
        raise PermissionError(a)

    monkeypatch.setattr(os, "replace", bloqueado)
    with pytest.raises(RuntimeError, match="no puedo reemplazar"):
        copiar_dlls(str(nvidia), str(dest), str(ct2), reintentos=3, espera=0)
    assert (dest / "cublas64_12.dll").read_bytes() == b"VIEJA-PERO-FUNCIONA"
    assert not (dest / "cublas64_12.dll.tmp").exists()


def test_copiar_dlls_restore_falla_error_claro(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Si ni siquiera la restauración del backup pasa, error combinado claro."""
    import os
    import shutil

    from setup_dlls import copiar_dlls

    nvidia = tmp_path / "nvidia"
    (nvidia / "cublas" / "bin").mkdir(parents=True)
    (nvidia / "cublas" / "bin" / "cublas64_12.dll").write_bytes(b"origen")
    (nvidia / "cublas" / "bin" / "cublasLt64_12.dll").write_bytes(b"origen")
    (nvidia / "cudnn" / "bin").mkdir(parents=True)
    (nvidia / "cuda_runtime" / "bin").mkdir(parents=True)
    (nvidia / "cuda_runtime" / "bin" / "cudart64_12.dll").write_bytes(b"origen")
    dest = tmp_path / "Scripts"
    dest.mkdir()
    ct2 = tmp_path / "ct2"
    ct2.mkdir()
    (dest / "cublas64_12.dll").write_bytes(b"VIEJA-BUENA")
    (dest / "cublasLt64_12.dll").write_bytes(b"VIEJA-BUENA")
    (dest / "cudart64_12.dll").write_bytes(b"VIEJA-BUENA")
    real_replace = os.replace

    def replace_con_restore_bloqueado(a: str, b: str) -> None:
        if a.endswith(".tmp"):
            raise PermissionError(a)
        real_replace(a, b)

    def bloqueado(src: str, dst: str) -> None:
        raise PermissionError(src)

    monkeypatch.setattr(os, "replace", replace_con_restore_bloqueado)
    monkeypatch.setattr(shutil, "copy2", bloqueado)
    monkeypatch.setattr(time, "sleep", lambda _: None)
    with pytest.raises(RuntimeError, match="ni restaurar la DLL vieja"):
        copiar_dlls(str(nvidia), str(dest), str(ct2), reintentos=3, espera=0)


def test_copiar_dlls_tmp_viejo_no_se_promueve(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Un .tmp huérfano de otra corrida (sin b.dll) no se promueve si copy falla."""
    import shutil

    from setup_dlls import copiar_dlls

    nvidia = tmp_path / "nvidia"
    (nvidia / "cublas" / "bin").mkdir(parents=True)
    (nvidia / "cublas" / "bin" / "cublas64_12.dll").write_bytes(b"origen")
    (nvidia / "cublas" / "bin" / "cublasLt64_12.dll").write_bytes(b"origen")
    (nvidia / "cudnn" / "bin").mkdir(parents=True)
    (nvidia / "cuda_runtime" / "bin").mkdir(parents=True)
    (nvidia / "cuda_runtime" / "bin" / "cudart64_12.dll").write_bytes(b"origen")
    dest = tmp_path / "Scripts"
    dest.mkdir()
    ct2 = tmp_path / "ct2"
    ct2.mkdir()
    for f in ("cublas64_12.dll", "cublasLt64_12.dll", "cudart64_12.dll"):
        (dest / f"{f}.tmp").write_bytes(b"HUERFANO")

    def bloqueado(src: str, dst: str) -> None:
        raise PermissionError(src)

    monkeypatch.setattr(shutil, "copy2", bloqueado)
    monkeypatch.setattr(time, "sleep", lambda _: None)
    with pytest.raises(RuntimeError, match="no puedo reemplazar"):
        copiar_dlls(str(nvidia), str(dest), str(ct2), reintentos=3, espera=0)
    for f in ("cublas64_12.dll", "cublasLt64_12.dll", "cudart64_12.dll"):
        assert not (dest / f).exists()
        assert (dest / f"{f}.tmp").read_bytes() == b"HUERFANO"


def test_copiar_dlls_reintentos_por_defecto(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Contrato por defecto: 6 intentos con backoff de 1.0 s."""
    import shutil

    from setup_dlls import copiar_dlls

    nvidia = tmp_path / "nvidia"
    (nvidia / "cublas" / "bin").mkdir(parents=True)
    (nvidia / "cublas" / "bin" / "cublas64_12.dll").write_bytes(b"origen")
    (nvidia / "cublas" / "bin" / "cublasLt64_12.dll").write_bytes(b"origen")
    (nvidia / "cudnn" / "bin").mkdir(parents=True)
    (nvidia / "cuda_runtime" / "bin").mkdir(parents=True)
    (nvidia / "cuda_runtime" / "bin" / "cudart64_12.dll").write_bytes(b"origen")
    dest = tmp_path / "Scripts"
    dest.mkdir()
    ct2 = tmp_path / "ct2"
    ct2.mkdir()
    (dest / "cublas64_12.dll").write_bytes(b"bloqueado")
    (dest / "cublasLt64_12.dll").write_bytes(b"bloqueado")
    (dest / "cudart64_12.dll").write_bytes(b"bloqueado")
    intentos: list[str] = []
    duerme: list[float] = []

    def bloqueado(src: str, dst: str) -> None:
        intentos.append(dst)
        raise PermissionError(src)

    def spy_sleep(s: float) -> None:
        duerme.append(s)

    monkeypatch.setattr(shutil, "copy2", bloqueado)
    monkeypatch.setattr(time, "sleep", spy_sleep)
    with pytest.raises(RuntimeError, match="no puedo reemplazar"):
        copiar_dlls(str(nvidia), str(dest), str(ct2))
    assert len(intentos) == 6
    assert duerme == [1.0, 1.0, 1.0, 1.0, 1.0]


def test_copiar_dlls_limpia_tmp_con_reintento(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """El .tmp viejo (todavía escaneado) se borra con reintento, no se abandona."""
    import os

    from setup_dlls import copiar_dlls

    nvidia = tmp_path / "nvidia"
    (nvidia / "cublas" / "bin").mkdir(parents=True)
    (nvidia / "cublas" / "bin" / "cublas64_12.dll").write_bytes(b"origen")
    (nvidia / "cublas" / "bin" / "cublasLt64_12.dll").write_bytes(b"origen")
    (nvidia / "cudnn" / "bin").mkdir(parents=True)
    (nvidia / "cuda_runtime" / "bin").mkdir(parents=True)
    (nvidia / "cuda_runtime" / "bin" / "cudart64_12.dll").write_bytes(b"origen")
    dest = tmp_path / "Scripts"
    dest.mkdir()
    ct2 = tmp_path / "ct2"
    ct2.mkdir()
    (dest / "cublas64_12.dll").write_bytes(b"viejo")
    real_remove = os.remove
    intentos: list[str] = []
    duerme: list[float] = []

    def remove_que_se_desbloquea(path: str) -> None:
        if path == str(dest / "cublas64_12.dll.tmp"):
            intentos.append(path)
            if len(intentos) <= 2:
                raise PermissionError(path)
        real_remove(path)

    def spy_sleep(s: float) -> None:
        duerme.append(s)

    monkeypatch.setattr(os, "remove", remove_que_se_desbloquea)
    monkeypatch.setattr(time, "sleep", spy_sleep)
    copiar_dlls(str(nvidia), str(dest), str(ct2), reintentos=5, espera=0)
    assert not (dest / "cublas64_12.dll.tmp").exists()
    assert len(intentos) == 3
    assert duerme == [0.0, 0.0]


def test_dir_ct2_deriva_del_paquete_real(monkeypatch: pytest.MonkeyPatch) -> None:
    """Sin paquete instalado falla claro (ModuleNotFoundError), no fantasma."""
    import sys
    import types

    from setup_dlls import dir_ct2

    fake = types.ModuleType("ctranslate2")
    fake.__file__ = "/fake/site/ctranslate2/__init__.py"
    monkeypatch.setitem(sys.modules, "ctranslate2", fake)
    assert dir_ct2() == "/fake/site/ctranslate2"


def test_dir_ct2_sin_file_lanza(monkeypatch: pytest.MonkeyPatch) -> None:
    import sys
    import types

    from setup_dlls import dir_ct2

    fake = types.ModuleType("ctranslate2")
    fake.__file__ = None
    monkeypatch.setitem(sys.modules, "ctranslate2", fake)
    with pytest.raises(RuntimeError) as excinfo:
        dir_ct2()
    assert str(excinfo.value) == "ctranslate2 sin __file__: instalación rota"


def test_main_directo_con_overrides(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    """main() directo con overrides: atribuible por mutmut, cero toques reales."""
    import sys
    import types

    from setup_dlls import main

    monkeypatch.setitem(sys.modules, "nvidia", types.ModuleType("nvidia"))

    nvidia = tmp_path / "nvidia"
    (nvidia / "cublas" / "bin").mkdir(parents=True)
    (nvidia / "cublas" / "bin" / "cublas64_12.dll").write_bytes(b"f")
    (nvidia / "cublas" / "bin" / "cublasLt64_12.dll").write_bytes(b"f")
    (nvidia / "cudnn" / "bin").mkdir(parents=True)
    (nvidia / "cuda_runtime" / "bin").mkdir(parents=True)
    (nvidia / "cuda_runtime" / "bin" / "cudart64_12.dll").write_bytes(b"f")
    dest = tmp_path / "Scripts"
    dest.mkdir()
    sp = tmp_path / "sp"
    sp.mkdir()
    ct2 = sp / "ctranslate2"
    ct2.mkdir()
    fake_ct2 = types.ModuleType("ctranslate2")
    fake_ct2.__file__ = str(ct2 / "__init__.py")
    monkeypatch.setitem(sys.modules, "ctranslate2", fake_ct2)
    main(str(nvidia), str(dest), str(sp))
    assert (dest / "cublas64_12.dll").exists()
    assert (ct2 / "cudart64_12.dll").exists()
    out = capsys.readouterr().out
    assert out == f"keeper+pth en {sp / 'zz_nvidia_dlls.py'} y {sp / 'zz_nvidia_dlls.pth'}\n"


def test_main_defaults_usan_fakes(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """main() sin args resuelve todo desde nvidia/exe fakes (rama defaults)."""
    import sys
    import types

    from setup_dlls import main

    nvidia = tmp_path / "nvidia"
    (nvidia / "cublas" / "bin").mkdir(parents=True)
    (nvidia / "cublas" / "bin" / "cublas64_12.dll").write_bytes(b"f")
    (nvidia / "cublas" / "bin" / "cublasLt64_12.dll").write_bytes(b"f")
    (nvidia / "cudnn" / "bin").mkdir(parents=True)
    (nvidia / "cuda_runtime" / "bin").mkdir(parents=True)
    (nvidia / "cuda_runtime" / "bin" / "cudart64_12.dll").write_bytes(b"f")
    fake_nvidia = types.ModuleType("nvidia")
    fake_nvidia.__path__ = [str(nvidia)]
    monkeypatch.setitem(sys.modules, "nvidia", fake_nvidia)
    exe = tmp_path / "Scripts" / "python.exe"
    exe.parent.mkdir(parents=True)
    exe.touch()
    monkeypatch.setattr(sys, "executable", str(exe))
    ct2 = tmp_path / "ctranslate2"
    ct2.mkdir()
    fake_ct2 = types.ModuleType("ctranslate2")
    fake_ct2.__file__ = str(ct2 / "__init__.py")
    monkeypatch.setitem(sys.modules, "ctranslate2", fake_ct2)
    main()
    assert (tmp_path / "Scripts" / "cublas64_12.dll").exists()
    assert (ct2 / "cudart64_12.dll").exists()


def test_guard_main_solo_bajo_main(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """El guard __main__ dispara main() con fakes; importar no ejecuta nada."""
    import runpy
    import sys
    import types

    import setup_dlls

    nvidia = tmp_path / "nvidia"
    (nvidia / "cublas" / "bin").mkdir(parents=True)
    (nvidia / "cublas" / "bin" / "cublas64_12.dll").write_bytes(b"f")
    (nvidia / "cublas" / "bin" / "cublasLt64_12.dll").write_bytes(b"f")
    (nvidia / "cudnn" / "bin").mkdir(parents=True)
    (nvidia / "cuda_runtime" / "bin").mkdir(parents=True)
    (nvidia / "cuda_runtime" / "bin" / "cudart64_12.dll").write_bytes(b"f")
    fake_nvidia = types.ModuleType("nvidia")
    fake_nvidia.__path__ = [str(nvidia)]
    monkeypatch.setitem(sys.modules, "nvidia", fake_nvidia)
    exe = tmp_path / "Scripts" / "python.exe"
    exe.parent.mkdir(parents=True)
    exe.touch()
    monkeypatch.setattr(sys, "executable", str(exe))
    ct2 = tmp_path / "ctranslate2"
    ct2.mkdir()
    fake_ct2 = types.ModuleType("ctranslate2")
    fake_ct2.__file__ = str(ct2 / "__init__.py")
    monkeypatch.setitem(sys.modules, "ctranslate2", fake_ct2)
    raiz = Path(setup_dlls.__file__)
    runpy.run_path(str(raiz), run_name="__main__")
    assert (tmp_path / "Scripts" / "cublas64_12.dll").exists()


def test_keeper_escapa_rutas_raras(tmp_path: Path) -> None:
    """Apóstrofos y no-ASCII no rompen el keeper generado (va en UTF-8)."""
    rara = tmp_path / "O'Brien-é"
    rara.mkdir()
    keeper, _ = escribir_keeper(str(tmp_path), [str(rara)])
    raw = Path(keeper).read_bytes()
    texto = raw.decode("utf-8")
    compile(texto, keeper, "exec")
    assert repr(str(rara)) in texto
    assert "é".encode() in raw
