"""The light base (docs/adr/0001-light-base.md): contracts import with numpy only.

`LIGHT_MODULES` is the one list of modules that must be importable in a venv holding
only numpy (ChimeraX, pvpython, the rig PC, a viewer). Later tasks extend it as new
light modules land. The venv test installs this checkout with `--no-deps` into a
scratch venv with numpy alone, so a heavy import that sneaks in fails with
ImportError (nothing else is installed), and independently the checker asserts none
of `HEAVY` entered `sys.modules`.
"""

from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
import textwrap
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[1]

# Modules that must import with numpy only. Extend this list, nowhere else.
LIGHT_MODULES = [
    "opym",
    "opym.stream",
    "opym.stream.protocol",
    "opym.stream.bfill",
    "opym.stream.outcome",
    "opym.stream.bfill_run",
    "opym.stream.backfill_runs",
    "opym.timeline",
    "opym.registry",
    "opym.metadata",
    "opym.utils",
    "opym.lanes",
    "opym.decon_config",
    "opym.ome_zarr_writer",
    "opym.display",
    "opym.store",
    "opym.stream.live_status",
    "opym.testing",
    "opym.testing.display",
    "opym.testing.store",
    "opym.testing.readers",
    "opym.testing.stream",
]

HEAVY = [
    "scipy",
    "skimage",
    "sklearn",
    "tifffile",
    "imagecodecs",
    "zarr",
    "numcodecs",
    "ome_types",
    "matlab",
    "torch",
    "napari",
    "vispy",
    "dask",
    "tqdm",
    "yaml",
    "msgpack",
    "zmq",
    "ipywidgets",
    "matplotlib",
    "IPython",
]

# Runs inside the scratch venv (`python -I`). Argument: JSON of
# {modules, heavy, extra_path}.
_CHECKER = textwrap.dedent(
    """
    import importlib, json, sys

    cfg = json.loads(sys.argv[1])
    if cfg["extra_path"]:
        sys.path.insert(0, cfg["extra_path"])
    problems = []
    for name in cfg["modules"]:
        try:
            importlib.import_module(name)
        except Exception as exc:  # noqa: BLE001
            problems.append(f"{name}: import failed: {type(exc).__name__}: {exc}")
            continue
        leaked = sorted(h for h in cfg["heavy"] if h in sys.modules)
        if leaked:
            problems.append(f"{name}: pulled in {leaked}")
            break  # the first offender is the useful one; later ones inherit it
    print(json.dumps({"prefix": sys.prefix, "problems": problems}))
    """
)


def run_checker(
    python: Path,
    modules: list[str],
    heavy: list[str] = HEAVY,
    extra_path: Path | None = None,
) -> dict:
    """Imports `modules` in a fresh interpreter; returns {"prefix", "problems"}."""
    cfg = {
        "modules": modules,
        "heavy": heavy,
        "extra_path": str(extra_path) if extra_path else "",
    }
    env = {k: v for k, v in os.environ.items() if not k.startswith("PYTHON")}
    proc = subprocess.run(
        [str(python), "-I", "-c", _CHECKER, json.dumps(cfg)],
        capture_output=True,
        text=True,
        env=env,
        check=False,
    )
    assert proc.returncode == 0, proc.stderr
    return json.loads(proc.stdout.strip().splitlines()[-1])


@pytest.fixture(scope="module")
def light_venv(tmp_path_factory) -> Path:
    """A venv with numpy only, with this checkout installed `--no-deps`."""
    uv = shutil.which("uv")
    if uv is None:
        pytest.skip("uv is not installed: cannot build the numpy-only scratch venv")
    venv = tmp_path_factory.mktemp("light") / "venv"
    env = {k: v for k, v in os.environ.items() if not k.startswith("PYTHON")}
    py = venv / "bin" / "python"
    # (command, True if a failure means "no network/cache" and should skip)
    steps = (
        ([uv, "venv", "--python", sys.executable, str(venv)], True),
        ([uv, "pip", "install", "--python", str(py), "numpy"], True),
        ([uv, "pip", "install", "--python", str(py), "--no-deps", str(REPO)], False),
    )
    for cmd, skippable in steps:
        proc = subprocess.run(cmd, capture_output=True, text=True, env=env, check=False)
        if proc.returncode == 0:
            continue
        detail = f"{cmd}\n{proc.stdout}\n{proc.stderr}"
        offline = any(
            s in proc.stderr
            for s in (
                "No solution found",
                "Failed to fetch",
                "Failed to download",
                "dns error",
            )
        )
        if skippable or offline:
            pytest.skip(
                "cannot build the numpy-only scratch venv (no network or uv cache): "
                + detail
            )
        pytest.fail(detail)
    return py


def test_light_modules_import_with_numpy_only(light_venv):
    result = run_checker(light_venv, LIGHT_MODULES)
    assert Path(result["prefix"]) == light_venv.parent.parent  # really the scratch venv
    assert result["problems"] == []


def test_checker_fails_on_a_heavy_import(light_venv, tmp_path):
    """Negative control: a module that imports a heavy package must fail the checker."""
    ctrl = tmp_path / "ctrl"
    (ctrl / "scipy").mkdir(parents=True)
    (ctrl / "scipy" / "__init__.py").write_text("")  # stand-in; no network needed
    (ctrl / "heavy_probe.py").write_text("import scipy\n")
    result = run_checker(
        light_venv, ["opym", "heavy_probe", "opym.timeline"], extra_path=ctrl
    )
    assert any("heavy_probe" in p and "scipy" in p for p in result["problems"])


def test_checker_fails_on_a_missing_dependency(light_venv, tmp_path):
    """Negative control: a module needing a package absent from the venv fails."""
    ctrl = tmp_path / "ctrl2"
    ctrl.mkdir()
    (ctrl / "needs_yaml.py").write_text("import yaml\n")
    result = run_checker(light_venv, ["needs_yaml"], extra_path=ctrl)
    assert any("needs_yaml" in p and "import failed" in p for p in result["problems"])


def test_import_opym_does_not_import_viewer_stack():
    """In-process-cheap check (no venv): `import opym` / `opym.stream` stay lazy even
    when the viewer packages are installed."""
    env = {k: v for k, v in os.environ.items() if not k.startswith("PYTHON")}
    env["PYTHONPATH"] = str(REPO / "src")
    code = (
        "import sys, opym, opym.stream\n"
        "bad = [m for m in ('ipywidgets','matplotlib','IPython','ipyfilechooser',"
        "'scipy','skimage','tifffile','zmq','numcodecs') if m in sys.modules]\n"
        "assert not bad, bad\n"
        "assert 'opym.batch' not in sys.modules\n"
        "assert 'opym.stream.receiver' not in sys.modules\n"
    )
    proc = subprocess.run(
        [sys.executable, "-c", code],
        capture_output=True,
        text=True,
        env=env,
        check=False,
    )
    assert proc.returncode == 0, proc.stderr


def test_lazy_names_resolve():
    import opym
    import opym.stream

    assert set(opym.__all__) == set(opym._LAZY)
    assert set(opym.stream.__all__) == set(opym.stream._LAZY)
    assert set(opym.__all__) <= set(dir(opym))
    assert set(opym.stream.__all__) <= set(dir(opym.stream))
    # light names resolve without heavy deps; the value is the submodule's own object
    from opym import StatusRegistry, parse_z_step
    from opym.metadata import parse_z_step as direct_z
    from opym.registry import StatusRegistry as direct_reg

    assert StatusRegistry is direct_reg and parse_z_step is direct_z
    from opym.stream import MSG_FRAME, pack_message
    from opym.stream.protocol import MSG_FRAME as direct_frame

    assert MSG_FRAME is direct_frame and callable(pack_message)
    assert opym.timeline.__name__ == "opym.timeline"  # submodule attribute access
    with pytest.raises(AttributeError):
        opym.no_such_name  # noqa: B018


def test_lazy_heavy_names_resolve_when_deps_present():
    pytest.importorskip("zmq")
    pytest.importorskip("numcodecs")
    pytest.importorskip("ipywidgets")
    from opym import run_batch_cropping
    from opym.batch import run_batch_cropping as direct
    from opym.stream import StreamReceiver, main
    from opym.stream.receiver import StreamReceiver as direct_rx

    assert run_batch_cropping is direct
    assert StreamReceiver is direct_rx and callable(main)
