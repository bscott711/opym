import os

import pytest

from opym import utils
from opym.utils import resolve_output_base

# The mirror reproduces an absolute POSIX path under its root (Argus only).
pytestmark = pytest.mark.skipif(os.name != "posix", reason="mirror paths are POSIX paths")


@pytest.fixture
def world(tmp_path, monkeypatch):
    """A raw root, a mirror root, and a set of directories `os.access` reports
    as unwritable (portable: chmod means nothing to os.access on Windows)."""
    monkeypatch.setenv(utils.MIRROR_ROOT_ENV, str(tmp_path / "mirror"))
    read_only: set = set()
    real_access = os.access

    def access(path, mode):
        if mode == os.W_OK and os.path.realpath(path) in read_only:
            return False
        return real_access(path, mode)

    monkeypatch.setattr(utils.os, "access", access)
    raw = tmp_path / "raw"
    raw.mkdir()

    def mirrored(leaf):
        return tmp_path / "mirror" / str(leaf.resolve()).lstrip("/")

    world = type("World", (), {})()
    world.raw, world.read_only, world.mirrored = raw, read_only, mirrored
    return world


def test_a_writable_leaf_is_its_own_output_base(world):
    leaf = world.raw / "day" / "Cell_001"
    leaf.mkdir(parents=True)
    assert resolve_output_base(leaf) == leaf


def test_a_flat_folders_missing_leaf_goes_in_place_when_its_folder_is_writable(world):
    """`<day>/<base>_<channel>.ome.zarr` groups under `<day>/<base>`, which nothing
    creates: its output belongs in place (where the live lane puts it), not the mirror."""
    day = world.raw / "20261023_LCK"
    (day / "WT_Stim_mNG_488nm.ome.zarr").mkdir(parents=True)
    leaf = day / "WT_Stim"
    assert resolve_output_base(leaf) == leaf
    assert not leaf.exists()                                 # resolving never creates anything


def test_a_missing_leaf_under_an_unwritable_folder_is_mirrored(world):
    day = world.raw / "lab_mate"
    day.mkdir()
    world.read_only.add(os.path.realpath(day))
    leaf = day / "Cell_1"
    assert resolve_output_base(leaf) == world.mirrored(leaf)


def test_an_existing_unwritable_leaf_is_mirrored(world):
    leaf = world.raw / "lab_mate" / "Cell_1"
    leaf.mkdir(parents=True)
    world.read_only.add(os.path.realpath(leaf))
    assert resolve_output_base(leaf) == world.mirrored(leaf)


def test_output_already_in_the_mirror_stays_there(world):
    """A dataset processed into the mirror before in-place output existed keeps
    resolving there, so it is neither relocated nor reprocessed."""
    leaf = world.raw / "day" / "Cell_001"
    (world.raw / "day").mkdir()
    world.mirrored(leaf).mkdir(parents=True)
    assert resolve_output_base(leaf) == world.mirrored(leaf)


def test_once_created_in_place_the_leaf_keeps_resolving_in_place(world):
    day = world.raw / "day"
    day.mkdir()
    leaf = day / "WT_Stim"
    base = resolve_output_base(leaf)
    (base / "decon_stage").mkdir(parents=True)               # what the pipeline does next
    assert resolve_output_base(leaf) == leaf
