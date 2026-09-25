# Ruff style: Compliant
"""
Core utilities, definitions, and path helpers for the OPM Cropper.
"""

from __future__ import annotations

import os
import re
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import Literal

import numpy as np

# Env var to override where redirected output lands when the raw acquisition
# directory itself isn't writable (see `resolve_output_base`). Must match
# whatever the opym-dashboard repo's `registry_reader.py` mirrors this as --
# that's a schema contract, not shared code (same reasoning as `STAGES`
# there), so keep both in sync by hand if this ever changes.
MIRROR_ROOT_ENV = "OPYM_OUTPUT_MIRROR_ROOT"
DEFAULT_MIRROR_ROOT = Path("/mmfs2/scratch/SDSMT.LOCAL/bscott/opym_backfill/outputs")


def mirror_root() -> Path:
    return Path(os.environ.get(MIRROR_ROOT_ENV, str(DEFAULT_MIRROR_ROOT)))


def resolve_output_base(leaf_dir: Path) -> Path:
    """Returns `leaf_dir` itself when it's writable -- the normal case, and
    the one every already-processed dataset was written under -- else a
    mirrored location under `mirror_root()` reproducing `leaf_dir`'s full
    absolute path.

    Some raw acquisition directories (e.g. a lab-mate's read-only-to-us
    `jacks.local` upload) can never be written to, so writing output as a
    sibling of the raw file there always fails with `PermissionError`.
    Falling back only when unwritable -- rather than always mirroring --
    keeps the documented "output lives next to the raw data" convention
    (bioimaging/CLAUDE.md) for the common case and never relocates output
    for a dataset that already has it written in place.

    Reproducing the full path (not a hash) under the mirror root keeps the
    result human-navigable: `<mirror_root>/mmfs1/scratch/jacks.local/.../Cell_1/`.
    """
    leaf_dir = Path(leaf_dir)
    if os.access(leaf_dir, os.W_OK):
        return leaf_dir
    return mirror_root() / str(leaf_dir.resolve()).lstrip("/")


class OutputFormat(str, Enum):
    """Defines the allowed output formats."""

    ZARR = "ZARR"
    TIFF_SERIES = "TIFF_SERIES_SPLIT_C"

    def __str__(self):
        return self.value


MicroscopyDataType = Literal["LLSM", "OPM", "UNKNOWN"]


def detect_microscopy_data_type(directory: Path) -> MicroscopyDataType:
    """
    Detects the microscopy data type by inspecting filenames.

    Args:
        directory: The Path object to the data directory.

    Returns:
        "LLSM", "OPM", or "UNKNOWN".
    """
    if not directory.is_dir():
        raise NotADirectoryError(f"Path is not a directory: {directory}")

    # --- MODIFICATION: Check for OPM using the new C..._T... format ---
    if next(directory.glob("*_C[0-9]_T[0-9][0-9][0-9].tif"), None):
        return "OPM"

    # Check for LLSM
    if next(
        directory.glob("*_Cam[AB]_ch[0-9]_stack[0-9][0-9][0-9][0-9]*.tif"),
        None,
    ):
        return "LLSM"

    return "UNKNOWN"


@dataclass(frozen=True)
class DerivedPaths:
    """Holds all paths derived from the base input file."""

    base_file: Path
    metadata_file: Path
    output_dir: Path
    output_log: Path
    sanitized_name: str


def sanitize_filename(name: str) -> str:
    """Removes .ome.tif and replaces spaces."""
    return name.replace(".ome.tif", "").replace(" ", "_")


def derive_paths(base_file: Path, output_format: OutputFormat) -> DerivedPaths:
    """Derives all associated input and output paths from the base file.

    Input paths (`metadata_file`) always stay next to the raw file -- it's
    read-only input, always readable if `base_file` itself is. Only
    `output_dir` goes through `resolve_output_base`, so it redirects to the
    mirror only when `base_file.parent` (the raw acquisition dir) can't
    actually be written to.
    """
    base_name_no_ext = base_file.name.replace(".ome.tif", "")
    sanitized_name = sanitize_filename(base_file.name)
    metadata_file = base_file.parent / (base_name_no_ext + "_metadata.txt")

    if output_format == OutputFormat.ZARR:
        output_dir_name = "processed_ngff"
    else:
        output_dir_name = "processed_tiff_series_split"

    output_dir = resolve_output_base(base_file.parent) / output_dir_name
    output_log = output_dir / (sanitized_name + "_processing_log.json")

    return DerivedPaths(
        base_file=base_file,
        metadata_file=metadata_file,
        output_dir=output_dir,
        output_log=output_log,
        sanitized_name=sanitized_name,
    )


def parse_roi_string(roi_str: str) -> tuple[slice, slice]:
    """
    Parses a CLI string like "y1:y2, x1:x2" into a NumPy slice.
    e.g., "0:512, 0:512" -> (slice(0, 512), slice(0, 512))
    """
    if not re.match(r"^\d+:\d+,\s*\d+:\d+$", roi_str):
        raise ValueError(
            f"Invalid ROI format: '{roi_str}'. Expected 'y_start:y_stop,x_start:x_stop'"
        )

    y_str, x_str = roi_str.split(",")
    y_start, y_stop = map(int, y_str.strip().split(":"))
    x_start, x_stop = map(int, x_str.strip().split(":"))

    return (slice(y_start, y_stop), slice(x_start, x_stop))


def orient_zyx_for_dsr(volume: np.ndarray) -> np.ndarray:
    """
    Reorders a (Z, Y, X) numpy crop into the (ny, nx, nz) layout PetaKit5D's
    deskew/rotate functions expect via their `[ny, nx, nz] = size(vol)`
    convention: ny is the rotation-invariant lateral axis (this galvo-scanned
    OPM's coverslip-long axis, i.e. raw X), nx is the axis that drifts
    frame-to-frame as Z changes (raw Y), and nz is the frame count (raw Z).

    This mirrors the rot90() the legacy MATLAB BigTiff cropper applies by
    default (see run_bigtiff_cropper.m, gated by submit_remote_crop_job's
    rotate=True) before handing data to XR_deskewRotateFrame.m. The
    GPU-unified pipeline instead stages raw (Z, Y, X) numpy crops straight to
    /dev/shm with no equivalent correction, which silently shears/rotates the
    wrong spatial axis -- confirmed by reproducing deskewRotateFrame3D.m's
    own outSize formula against a live job's actual output shape.
    """
    rotated = np.rot90(volume, k=1, axes=(-2, -1))
    return np.moveaxis(rotated, 0, -1)


def orient_zyx_for_decon_tiff(volume: np.ndarray) -> np.ndarray:
    """
    Reorders a (Z, Y, X) numpy crop into the (nz, ny, nx) layout a TIFF must
    be written in so that MATLAB's `readtiff` hands PetaKit5D (ny, nx, nz).

    This is `orient_zyx_for_dsr` MINUS its trailing `moveaxis(0, -1)` -- TIFF
    paging performs that step for you. `tifffile.imwrite` of a numpy (a, b, c)
    array writes `a` pages of `b x c`, and `readtiff` returns
    (rows, cols, pages) == (b, c, a). So writing `orient_zyx_for_dsr`'s own
    output would give `ny` pages and reach MATLAB as (nx, nz, ny) -- wrong.

    Why this matters for deconvolution specifically: decon is a 3D
    convolution, and PetaKit5D's decon path
    (XR_decon_data_wrapper -> XR_RLdeconFrame3D -> RLdecon) has NO
    axis-order parameter -- it convolves the array exactly as stored, before
    XR_deskewRotateFrame's `inputAxisOrder` permute ever happens. Two of its
    internals hard-code dim 3 == scan Z: `psf_gen_new` takes its background
    from `psf(:, :, [1:5, end-4:end])` and FFT-resamples dim 3 from dz_psf to
    dz_data, and `omw_backprojector_generation` with `skewed=true` builds the
    skewed OTF mask as `cat(3, mask_r, mask_c, mask_l)` -- the three lobes
    must separate along dim 3. So a zarr-mirror store presented as (z, y, x)
    cannot be deconvolved in place, and permuting the PSF instead does not
    rescue it: the data itself has to be materialized in this order.

    The rot90 (rather than a plain transpose) is deliberate and load-bearing.
    It matches the legacy MATLAB BigTiff cropper, and -- critically -- the
    measured PSF carries the SAME rot90 (psf_tools/extract_bead_psf.py applies
    `np.rot90(avg_psf, k=1, axes=(1, 2))` with `rotate_90=True` by default).
    A plain transpose, which is what the deskew-only zarr path's
    `inputAxisOrder='zxy'` performs, differs from this by a flip of the ny
    (coverslip) axis -- i.e. it is a mirror image. Pinned by
    tests/test_decon_staging_geometry.py.
    """
    return np.rot90(volume, k=1, axes=(-2, -1))


def _matlab_round(x: float) -> int:
    """MATLAB's round: halves away from zero (Python's rounds to even)."""
    import math

    return int(math.floor(abs(x) + 0.5)) * (1 if x >= 0 else -1)


def dsr_shape_zyx(
    raw_shape_zyx: tuple[int, int, int],
    z_step_um: float,
    xy_pixel_size_um: float = 0.136,
    sheet_angle_deg: float = 60.0,
) -> tuple[int, int, int]:
    """The (Z, Y, X) shape of the deskewed/rotated volume PetaKit5D makes
    from a raw (Z, Y, X) volume -- so the processed OME-Zarr can be created
    before the first timepoint is processed.

    deskewRotateFrame3D (Crop on, no resampling, not objective scan) returns
    `round([ny, (nx-1)cos(t) + (nz-1)zAniso/sin(t), (nx-1)sin(t) - 4])` in
    its (y, x, z) order, where zAniso = sin(t) dz / xy and (ny, nx, nz) is
    the decon input: raw (X, Y, Z) after `orient_zyx_for_decon_tiff`. The
    TIFF pages (and the store) are that result's (z, y, x). The live zarr
    job fails loudly if a real result ever differs.
    """
    import math

    nz_raw, ny_raw, nx_raw = raw_shape_zyx
    ny, nx, nz = nx_raw, ny_raw, nz_raw
    theta = math.radians(sheet_angle_deg)
    z_aniso = math.sin(abs(theta)) * z_step_um / xy_pixel_size_um
    out_y = ny
    out_x = (nx - 1) * math.cos(theta) + (nz - 1) * z_aniso / math.sin(abs(theta))
    out_z = (nx - 1) * math.sin(abs(theta)) - 4
    return (_matlab_round(out_z), _matlab_round(out_y), _matlab_round(out_x))


def write_decon_staged_tiff(volume_zyx: np.ndarray, dst: Path) -> None:
    """Writes one raw (Z, Y, X) volume as a decon-ready staged TIFF at `dst`,
    in the `orient_zyx_for_decon_tiff` orientation PetaKit5D's decon path
    requires (see that function's docstring for why).

    Shared by `bioimaging.backfill.pipeline.build_decon_staging_dir` (batch
    path: materializes a whole zarr-precropped acquisition after the fact)
    and `opym.stream.receiver` (live path: stages each frame as it arrives)
    so the two never drift on the write-then-rename contract below.

    No-ops if `dst` already exists -- a half-written TIFF is not merely
    incomplete, it poisons every subsequent retry, because PetaKit5D's
    `readtiff` raises on it and a naive retry would keep handing it back;
    skip-if-present also makes re-delivery of an already-staged frame (a
    stream client's reconnect resend, or a batch re-run) a safe no-op.
    """
    import os

    import tifffile

    dst = Path(dst)
    if dst.exists():
        return
    oriented = orient_zyx_for_decon_tiff(np.asarray(volume_zyx))
    # Write-then-rename so a crash mid-write never leaves a half-written
    # TIFF at the real destination name (see docstring above).
    tmp = dst.with_name(dst.name + ".tmp")
    # ome=True explicitly: tifffile only auto-writes OME-XML when the path it
    # is given ends in .ome.tif, and `tmp` never does -- so a single-timepoint
    # file named `<store>.ome.tif` used to carry a plain JSON description,
    # which OME-aware readers (ChimeraX) reject outright. Pages are unchanged,
    # so PetaKit5D's `readtiff` sees exactly the same pixels either way.
    tifffile.imwrite(
        tmp, oriented, compression="zlib", ome=True, metadata={"axes": "ZYX"}
    )
    os.replace(tmp, dst)


def scan_channel_patterns(directory: Path) -> str:
    """
    Scans a directory for unique channel identifiers (e.g., _C00, _C01).
    Returns a comma-separated string for UI pre-filling.
    """
    if not directory.is_dir():
        return ""

    patterns = set()
    # Looking for _C followed by digits (standard for deinterlaced OPM)
    # or Cam[AB] (Standard for PetaKit/LLSM)
    file_re = re.compile(r".*?(_C\d+|Cam[AB]).*?", re.IGNORECASE)

    for f in directory.glob("*.tif"):
        match = file_re.search(f.name)
        if match:
            patterns.add(match.group(1))

    # Return as CSV string
    return ", ".join(sorted(list(patterns)))
