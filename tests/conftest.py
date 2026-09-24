import os
from pathlib import Path

import pytest

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
    eng.addpath(str(OPYM_SRC_DIR / "patches"), nargout=0)
    if eng.exist("XR_deskew_rotate_data_wrapper", "file", nargout=1) == 0:
        setup_m = PETAKIT_ROOT / "setup.m"
        if not setup_m.exists():
            eng.quit()
            pytest.skip(f"PetaKit5D setup.m not found at {setup_m}")
        eng.run(str(setup_m), nargout=0)

    yield eng
    eng.quit()


@pytest.fixture(autouse=True)
def _isolate_petakit_jobs_dir(tmp_path, monkeypatch):
    """Code that falls back to production's /dev/shm/petakit_jobs when
    PETAKIT_JOBS_DIR is unset (e.g. the receiver's per-session profile) must
    never write there from a test. That happened once on 2026-09-24:
    receiver tests appended fake sessions to the live profiling/receiver.jsonl."""
    monkeypatch.setenv("PETAKIT_JOBS_DIR", str(tmp_path / "petakit_jobs"))
