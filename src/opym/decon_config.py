# Ruff style: Compliant
"""Decon + deskew/rotate settings shared by every path that produces DSR
output: the backfill (bioimaging.backfill.pipeline) and the live lane
(opym.stream.live). Both build their tickets from `deskew_decon_kwargs`, so
live and batch output cannot drift apart. The live path is checked
bit-identical to the batch path on real data (see run_live_frames.m).

Decon settings are fixed here rather than left to PetaKit5D's defaults, all
four of which fail quietly on these volumes:
  * wienerAlpha defaults to 0.005, visibly over-sharpened. 0.02 was the
    decon-order comparison's value; a low-SNR-focused 22-variant then
    20-variant refinement sweep on Cell_005 (bioimaging README, "Decon
    parameter tuning") picked 0.20, paired with the hann/damp changes below.
    Alone, higher alpha only marginally helped.
  * hannWinBounds defaults to [0.8, 1.0]; lowering the lower bound to 0.4
    (more apodization) is one of the three knobs the sweep's winning "super4"
    combination changed together.
  * dampFactor defaults to 1 (off, decon_lucy_omw_function.m); 2 caps a decon
    value's departure from its own input by 2x, the direct remedy for
    isolated over-sharpened voxel spikes.
  * edgeErosion defaults to 0, which leaves a bright ringing stripe along the
    slab boundary: RLdecon.m applies `edgetaper` per z-PLANE, so the axial
    faces are never tapered and the FFT wraps there. Eroding 3 voxels removes
    it, for ~6% of the imaged slab.
Changing any of these changes what the output looks like, so they belong in
the ticket (and therefore the log) rather than in a MATLAB default.
"""

from __future__ import annotations

import os
from pathlib import Path

DECON_WIENER_ALPHA = 0.20
DECON_OTF_CUM_THRESH = 0.90  # unchanged from the old default; super4 kept it
DECON_HANN_WIN_BOUNDS = [0.4, 1.0]
DECON_DAMP_FACTOR = 2
DECON_EDGE_EROSION = 3

# Linear, not cubic (user decision 2026-09-24). Cubic makes PetaKit5D's
# deskewRotateFrame3D fall back to CPU imwarp: 13.6 s per 111-plane frame on
# 48 threads, and MATLAB has no gpuArray cubic. Linear takes its multithreaded
# mex path, 0.07 s per frame, which is what lets the live lane keep pace with
# acquisition. On real Cell_002 data the two agree to corr 0.9997 (median
# |diff| 1 count, peaks ~3% softer). Output made before the switch stays
# cubic; this is deliberately not part of `decon_params_fingerprint`, so it
# does not trigger reprocessing.
DSR_INTERP_METHOD = "linear"

# This microscope's detection-path pixel size. The stores' own NGFF metadata
# carries only placeholder scales, so it is never read from there.
XY_PIXEL_SIZE_UM = 0.136


def resolve_decon_psf() -> Path | None:
    """The PSF deconvolution should run with, or None for deskew-only.

    Read from the `OPYM_DECON_PSF` environment variable rather than threaded
    through as an argument: the backfill fans datasets out across a process
    pool, and the stream receiver is a separate service; an env var reaches
    every one of them without changing any signature. Unset means no decon
    (`DSR_nodecon`).
    """
    raw = os.environ.get("OPYM_DECON_PSF", "").strip()
    if not raw:
        return None
    psf = Path(raw).expanduser()
    if not psf.is_file():
        raise FileNotFoundError(
            f"OPYM_DECON_PSF points at {psf}, which is not a file. Refusing to "
            "fall back to deskew-only silently -- unset it to run without decon."
        )
    return psf.resolve()


def dsr_dir_name_for(psf: Path | None) -> str:
    """Output directory name for a DSR result, keyed on whether decon ran.

    Deconvolved output goes to a different directory than deskew-only output
    so the two can coexist, and enabling decon never overwrites the existing
    no-decon archive. The bare name `DSR` is not used: the PSF-tuning
    harnesses already write unrelated output under it.
    """
    return "DSR_decon" if psf else "DSR_nodecon"


def decon_params_fingerprint() -> str:
    """A short, deterministic fingerprint of the decon knobs in effect,
    recorded per dataset so a parameter retune (not just a PSF swap) is
    visible to the backfill's provenance check."""
    return (
        f"a{DECON_WIENER_ALPHA}_o{DECON_OTF_CUM_THRESH}_"
        f"h{DECON_HANN_WIN_BOUNDS[0]}-{DECON_HANN_WIN_BOUNDS[1]}_d{DECON_DAMP_FACTOR}"
    )


def deskew_decon_kwargs(psf: Path | None) -> dict:
    """Keyword arguments shared by `opym.petakit.submit_remote_deskew_job` and
    `submit_live_frames_job`. The decon knobs are ignored by a deskew-only
    (`psf=None`) deskew ticket."""
    return {
        "xy_pixel_size": XY_PIXEL_SIZE_UM,
        "dsr_dir_name": dsr_dir_name_for(psf),
        "interp_method": DSR_INTERP_METHOD,
        "wiener_alpha": DECON_WIENER_ALPHA,
        "otf_cum_thresh": DECON_OTF_CUM_THRESH,
        "hann_win_bounds": DECON_HANN_WIN_BOUNDS,
        "damp_factor": DECON_DAMP_FACTOR,
        "edge_erosion": DECON_EDGE_EROSION,
        # Without this the ticket carries gpu_decon:false and PetaKit5D runs the
        # RL iterations on CPU while both cards sit idle.
        "gpu_decon": True,
    }
