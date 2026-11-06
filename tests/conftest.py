import atexit
import importlib
import os
import shutil
import tempfile
from pathlib import Path

import pytest

# --- Production-path isolation --------------------------------------------
# The suite must never act on the production RAM disks, GPFS or registry. This
# runs at conftest import, before any opym module is imported, so module
# constants computed at import (e.g. `local_gpu_worker.BASE_DIR` from
# `lanes.jobs_dir()`) already see the redirect. The variables are overridden
# even when the caller exported them: a test run must not depend on, or touch,
# the caller's environment. (2026-11: a test module imported outside this
# conftest removed three production live-view caches; see
# docs/review/2026-11-review.md LIVE-07, TEST-01.)
#
# name -> file or directory below the session dir
ISOLATED_ENV_PATHS = {
    "PETAKIT_JOBS_DIR": "petakit_jobs",
    "OPYM_LIVE_VIEW_ROOT": "opym_live_view",
    "OPYM_STREAM_STAGE_ROOT": "opym_stream_stage",
    "OPYM_BACKFILL_REGISTRY_PATH": "registry.sqlite3",
}
# The registry has no default in the source; its production location is set by
# the opym-*.service units (all under /mmfs2, which is also forbidden below).
PRODUCTION_REGISTRY = Path(
    "/mmfs2/scratch/SDSMT.LOCAL/bscott/opym_backfill/registry.sqlite3"
)
FORBIDDEN_PREFIXES = (
    Path("/dev/shm/petakit_jobs"),  # opym.lanes.DEFAULT_JOBS_DIR
    Path("/dev/shm/opym_live_view"),  # opym.stream.live_zarr.DEFAULT_VIEW_ROOT
    Path("/dev/shm/opym_stream_stage"),
    Path("/mmfs1"),
    Path("/mmfs2"),
    PRODUCTION_REGISTRY,
)


def production_path_violations(env=None) -> list[str]:
    """One message per isolated variable in `env` (default os.environ) that
    resolves to or under a production path. Empty when all are safe."""
    env = os.environ if env is None else env
    bad = []
    for name in ISOLATED_ENV_PATHS:
        val = (env.get(name) or "").strip()
        if not val:
            continue
        resolved = Path(val).resolve()
        for prefix in FORBIDDEN_PREFIXES:
            if resolved == prefix or prefix in resolved.parents:
                bad.append(f"{name}={val} resolves under production path {prefix}")
                break
    return bad


def _assert_no_production_paths() -> None:
    bad = production_path_violations()
    if bad:
        raise pytest.UsageError(
            "refusing to run tests against production paths: " + "; ".join(bad)
        )


SESSION_DIR = Path(tempfile.mkdtemp(prefix="opym-pytest-"))
atexit.register(shutil.rmtree, SESSION_DIR, ignore_errors=True)
for _name, _rel in ISOLATED_ENV_PATHS.items():
    os.environ[_name] = str(SESSION_DIR / _rel)
_assert_no_production_paths()
# What the session-level redirect set (the per-test fixture below narrows it).
SESSION_ENV = {_name: os.environ[_name] for _name in ISOLATED_ENV_PATHS}

# matlab.engine must be imported before numpy/zarr: its native engine library
# needs a newer libstdc++ than the one bundled in numpy/zarr's C-extension
# wheels, and whichever loads first wins symbol resolution for the process.
# Broad except: without `module load matlab/R2024b` active, importing this
# doesn't just fail with ImportError -- matlab.engine's own __init__.py
# raises a bare OSError (missing GLIBCXX_3.4.30) that would otherwise abort
# collection of every test in this directory, not just the GPU-marked ones,
# since (unlike a single test module) a conftest.py import failure kills the
# whole pytest session.
try:
    import matlab.engine  # noqa: F401
except Exception:  # noqa: BLE001
    pass

OPYM_SRC_DIR = Path(__file__).resolve().parents[1] / "src" / "opym"
PETAKIT_ROOT = Path(os.environ.get("PETAKIT_ROOT", "/cm/shared/apps_local/petakit5d"))


def pytest_configure(config):
    _assert_no_production_paths()
    config.addinivalue_line(
        "markers",
        "gpu: requires a real CUDA GPU + `module load matlab/R2024b` "
        "(matlab.engine). Not run by default in environments without both.",
    )


def pytest_addoption(parser):
    parser.addoption(
        "--update-golden",
        action="store_true",
        default=False,
        help="Regenerate golden-reference fixtures instead of comparing against them.",
    )


@pytest.fixture
def update_golden(request):
    return request.config.getoption("--update-golden")


@pytest.fixture(scope="module")
def matlab_engine():
    """Shared by every `-m gpu` regression test in this suite -- each
    importing test module gets its own freshly-started engine instance
    (pytest's "module" scope is per requesting module even when the fixture
    itself lives in conftest.py), matching the cost/lifetime this had
    before being deduplicated out of test_gpu_pipeline_regression.py.
    """
    try:
        import matlab.engine
    except Exception as e:  # noqa: BLE001 - ImportError if the package is missing
        # entirely, but a bare OSError (missing GLIBCXX_3.4.30) if it's
        # installed but `module load matlab/R2024b` isn't active -- both
        # mean "can't use MATLAB here," so both should skip, not error.
        pytest.skip(
            "matlab.engine not usable -- run with bioimaging's venv and "
            f"`module load matlab/R2024b` active: {e}"
        )

    try:
        eng = matlab.engine.start_matlab("-nodisplay")
    except Exception as e:  # noqa: BLE001 - report exact MATLAB engine error to the user
        pytest.skip(f"Could not start MATLAB engine (is `module load matlab/R2024b` active?): {e}")

    try:
        eng.gpuDevice(1, nargout=1)
    except Exception as e:  # noqa: BLE001
        eng.quit()
        pytest.skip(f"No usable CUDA GPU for matlab.engine: {e}")

    eng.addpath(str(OPYM_SRC_DIR), nargout=0)
    if eng.exist("XR_deskew_rotate_data_wrapper", "file", nargout=1) == 0:
        setup_m = PETAKIT_ROOT / "setup.m"
        if not setup_m.exists():
            eng.quit()
            pytest.skip(f"PetaKit5D setup.m not found at {setup_m}")
        eng.run(str(setup_m), nargout=0)
    # After setup.m, as run_petakit_server.m does: setup.m prepends
    # PetaKit5D's own directories, so patches/ added before it lost to the
    # shared copies it is meant to shadow.
    eng.addpath(str(OPYM_SRC_DIR / "patches"), "-begin", nargout=0)

    yield eng
    eng.quit()


# Module-level fallbacks used when the variable is unset: (module, attribute,
# builder taking the per-test jobs dir and isolated root). Any code path that
# ignores the environment still lands in tmp.
_FALLBACK_CONSTANTS = (
    ("opym.lanes", "DEFAULT_JOBS_DIR", lambda j, r: str(j)),
    ("opym.stream.receiver", "_DEFAULT_JOBS_DIR", lambda j, r: str(j)),
    ("opym.stream.live_zarr", "DEFAULT_VIEW_ROOT", lambda j, r: r / "opym_live_view"),
    ("opym.petakit", "BASE_DIR", lambda j, r: j),
    ("opym.petakit", "QUEUE_DIR", lambda j, r: j / "queue"),
    ("opym.petakit", "DONE_DIR", lambda j, r: j / "completed"),
    ("opym.petakit", "FAIL_DIR", lambda j, r: j / "failed"),
)


@pytest.fixture(autouse=True)
def _isolate_production_paths(tmp_path, monkeypatch):
    """Every test gets its own jobs dir, live-view root and registry path
    under tmp_path, and the opt-in RAM-disk staging is off.

    Code that falls back to production's /dev/shm/petakit_jobs when
    PETAKIT_JOBS_DIR is unset (e.g. the receiver's per-session profile) must
    never write there from a test. That happened once on 2026-09-24:
    receiver tests appended fake sessions to the live profiling/receiver.jsonl.
    OPYM_STREAM_STAGE_ROOT is unset (not redirected): when set, the receiver
    stages raw stores there and 26 tests that read them from the dataset
    directory fail (TEST-01). A test of staging sets it itself."""
    root = tmp_path / "isolated"
    jobs = tmp_path / "petakit_jobs"
    monkeypatch.setenv("PETAKIT_JOBS_DIR", str(jobs))
    monkeypatch.setenv("OPYM_LIVE_VIEW_ROOT", str(root / "opym_live_view"))
    monkeypatch.setenv("OPYM_BACKFILL_REGISTRY_PATH", str(root / "registry.sqlite3"))
    monkeypatch.delenv("OPYM_STREAM_STAGE_ROOT", raising=False)
    for module, attr, build in _FALLBACK_CONSTANTS:
        try:
            mod = importlib.import_module(module)
        except Exception:  # noqa: BLE001 - optional deps; nothing to patch then
            continue
        if hasattr(mod, attr):
            monkeypatch.setattr(mod, attr, build(jobs, root))
    bad = production_path_violations()
    if bad:
        pytest.fail("production paths in the test environment: " + "; ".join(bad))
