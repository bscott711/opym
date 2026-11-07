# Ruff style: Compliant
"""How opym's 16-bit counts are shown: the 8-bit encodings and the channel palette.

One home for what viewers used to re-type (light: stdlib + numpy only).

- sqrt8, the rule every napari/ParaView-style viewer uses: code =
  round(255 * sqrt(min(v, vmax) / vmax)), `vmax` = 2^k - 1 fixed per channel and
  session by the first volume (`sqrt_vmax_for`). The GPU squares the code back
  (`sqrt_decode`), so what is shown, and the contrast limits, are linear counts.
  `DisplayEncoding` is the record of it (`display_C<c>.json` in the buffer dir).
  The MATLAB copies (`run_live_zarr.m sqrtLut`, `opymDisplayVmax.m`) must agree.
- linear8, the ChimeraX rule (ChimeraX has no decode stage, so it stays linear):
  a right shift that leaves `DISPLAY_LEVELS` grey levels below the top.
- the palette: channel colours as the writer records them, the legacy remap
  (stores written before 2026-09-26 name red for channel 1), and parsers.

Golden vectors for consumers' tests: `opym.testing.display`.
"""

from __future__ import annotations

import math
from typing import cast

import numpy as np

# --- sqrt8 ----------------------------------------------------------------------

# The floor of the sqrt curve's range (the dim cells' precision); see
# `sqrt_vmax_for`.
SQRT_VMAX = 4095
_UINT16_TOP = 65535


def sqrt_vmax_for(arr) -> int:
    """The sqrt curve's `vmax` for a channel whose first volume is `arr`:
    2^k - 1 >= 2 * its maximum, within [SQRT_VMAX, 65535]. As
    run_live_zarr.m's displayLut / opymDisplayVmax.m pick it."""
    top = 2 * int(np.max(arr)) + 1 if np.size(arr) else 0
    return int(
        min(
            _UINT16_TOP,
            max(SQRT_VMAX, 2 ** math.ceil(math.log2(max(top, 1))) - 1),
        )
    )


def sqrt_lut(vmax: float) -> np.ndarray:
    """Every uint16 count's sqrt code (65536 uint8)."""
    v = np.arange(65536, dtype=np.float64)
    code = np.rint(255 * np.sqrt(np.minimum(v, vmax) / vmax))
    return code.astype(np.uint8)


def to_8bit_sqrt(arr, vmax: float) -> np.ndarray:
    """`arr`'s (uint16 counts) sqrt codes, uint8: one lookup per value.

    Single thread; the live viewer spreads `DisplayEncoding.lut()` over z-slabs
    itself (`opym.live_view.to_8bit`) and gets the same codes."""
    return np.take(sqrt_lut(vmax), np.asarray(arr), mode="clip")


def sqrt_decode(code, vmax: float) -> np.ndarray:
    """Sqrt codes back to counts (float32), as the GPU decodes them."""
    n = np.asarray(code, dtype=np.float32) / np.float32(255)
    return (n**2 * np.float32(vmax)).astype(np.float32)


def sqrt_texture_clim(clim, vmax: float) -> tuple[float, float]:
    """Contrast limits in counts as the 8-bit sqrt texture's own clim: the GPU
    normalizes the texture by 255 and compares it with the decoded value, v / vmax."""
    return (float(clim[0]) * 255 / vmax, float(clim[1]) * 255 / vmax)


def sqrt_colormap_controls(
    rgb, vmax: float, clim=None, steps: int = 256
) -> tuple[np.ndarray, np.ndarray]:
    """A colormap that shows sqrt codes at linear intensity, for a stock viewer.

    Returns (controls, colors) for a colormap over the codes' normalized range
    (code / 255, i.e. napari contrast limits (0, 255) on a uint8 layer):
    `controls` is `steps` ascending points in [0, 1], `colors` the (steps, 4)
    RGBA floats. Point n shows `rgb` scaled by the linear ramp of the decoded
    count n**2 * vmax across `clim` (counts; default 0..vmax). With `steps` 256
    every code has its own point, so the display is exact for nearest sampling.
    """
    lo, hi = (0.0, float(vmax)) if clim is None else (float(clim[0]), float(clim[1]))
    controls = np.linspace(0.0, 1.0, steps)
    counts = controls**2 * float(vmax)
    level = np.clip((counts - lo) / max(hi - lo, 1e-9), 0.0, 1.0)
    colors = np.ones((steps, 4), dtype=np.float64)
    colors[:, :3] = level[:, None] * np.asarray(rgb, dtype=np.float64)[None, :3]
    return controls, colors


class DisplayEncoding:
    """How a channel's 8-bit volumes hold its counts: v ~= vmax * (code /
    255) ** power. `sqrt()` (power 2) is what servers and viewers write now;
    `linear(shift)` (power 1, vmax = 255 << shift) reads an older server's
    shifted buffers. Recorded per channel in the buffer dir as
    display_C<c>.json (`record`, `from_record`)."""

    def __init__(self, power: int, vmax: float, shift: int | None = None) -> None:
        self.power = power
        self.vmax = float(vmax)
        self.shift = shift
        self._lut: np.ndarray | None = None

    @classmethod
    def sqrt(cls, vmax: float = SQRT_VMAX) -> DisplayEncoding:
        return cls(2, vmax)

    @classmethod
    def sqrt_for(cls, vol) -> DisplayEncoding:
        """The square-root curve for a channel whose first volume is `vol`
        (`sqrt_vmax_for`)."""
        return cls.sqrt(sqrt_vmax_for(vol))

    @classmethod
    def linear(cls, shift: int) -> DisplayEncoding:
        return cls(1, 255 << shift, shift=shift)

    @classmethod
    def from_record(cls, rec) -> DisplayEncoding | None:
        """From a display_C<c>.json record; None if it names neither."""
        try:
            if rec.get("encoding") == "sqrt":
                return cls(2, float(rec["vmax"]))
            if "shift" in rec:
                return cls.linear(int(rec["shift"]))
        except (AttributeError, KeyError, TypeError, ValueError):
            pass
        return None

    def record(self) -> dict:
        if self.power == 2:
            return {"encoding": "sqrt", "vmax": self.vmax}
        return {"shift": self.shift}

    def __eq__(self, other) -> bool:
        return isinstance(other, DisplayEncoding) and (self.power, self.vmax) == (
            other.power,
            other.vmax,
        )

    def __hash__(self) -> int:
        return hash((self.power, self.vmax))

    def __repr__(self) -> str:
        return f"DisplayEncoding(power={self.power}, vmax={self.vmax:g})"

    def lut(self) -> np.ndarray:
        """Every uint16 count's code (65536 uint8), built once."""
        if self._lut is None:
            if self.power == 2:
                self._lut = sqrt_lut(self.vmax)
            else:
                code = np.minimum(np.arange(65536) >> cast(int, self.shift), 255)
                self._lut = code.astype(np.uint8)
        return self._lut

    def decode(self, codes) -> np.ndarray:
        """Codes back to counts (float32), as the GPU decodes them."""
        n = np.asarray(codes, dtype=np.float32) / np.float32(255)
        return (n**self.power * np.float32(self.vmax)).astype(np.float32)

    def texture_clim(self, clim) -> tuple[float, float]:
        """Contrast limits in counts as the 8-bit texture's own clim: the
        GPU normalizes that by 255 and compares it with the decoded value,
        v / vmax."""
        return sqrt_texture_clim(clim, self.vmax)


# --- linear8 (ChimeraX) -----------------------------------------------------------

# Grey levels kept below the top of a channel's display range when 16-bit data
# is shown as 8-bit, linearly (ChimeraX; side by side, 121-138 levels looked
# identical). The one linear rule: chimerax-ngff and argus-desktop import it.
DISPLAY_LEVELS = 120


def display_shift(top, levels: int = DISPLAY_LEVELS) -> int:
    """Right shift leaving `levels` grey levels below `top`: floor(log2(top/levels)).

    Never negative.

    top is the upper end of the display range (the maximum of time point 0). The
    shift is fixed for a channel so intensities stay comparable across time points.
    """
    shift = 0
    while top / 2 ** (shift + 1) >= levels:
        shift += 1
    return shift


def to_8bit_linear(m, shift: int, pool=None) -> np.ndarray:
    """uint8 copy of an unsigned 16-bit block: m >> shift, clipped to 255.

    The work is split over z slabs on `pool` (an executor with `map`; numpy
    releases the GIL on slabs this big), so a full-resolution block takes tens
    of milliseconds instead of a few tenths of a second. The source is not touched.
    """
    out = np.empty(m.shape, np.uint8)

    def one(sl):
        part = m[sl] >> shift if shift else m[sl].copy()
        np.minimum(part, 255, out=part)
        out[sl] = part

    n = m.shape[0]
    slabs = [slice(i, min(i + 16, n)) for i in range(0, n, 16)]
    if pool is None or len(slabs) == 1:
        for sl in slabs:
            one(sl)
    else:
        list(pool.map(one, slabs))
    return out


def eight_bit_ok(dtype) -> bool:
    """Whether a stored type is shown as 8-bit on request: unsigned 16-bit integers."""
    return dtype.kind == "u" and dtype.itemsize == 2


# --- palette ----------------------------------------------------------------------

# Distinct, colour-blind-safe-ish emission colours; index = channel number.
# The second is magenta, not red: two-colour composites are green/magenta.
CHANNEL_COLORS = ("00FF00", "FF00FF", "00B3FF", "FFC400")
# Colours stores name that viewers show as another: FF3D3D is the red that stores
# written before 2026-09-26 name for channel 1. Applied when reading, never by
# rewriting stores on disk.
LEGACY_COLOR_REMAP = {"FF3D3D": "FF00FF"}

# hex colour -> napari colormap name (for the colours in CHANNEL_COLORS).
_NAPARI_COLORMAPS = {
    "00FF00": "green",
    "FF00FF": "magenta",
    "00B3FF": "cyan",
    "FFC400": "yellow",
}


def default_channel_color(i: int) -> str:
    """Channel `i`'s colour when a store names none; wraps past the fourth."""
    return CHANNEL_COLORS[int(i) % len(CHANNEL_COLORS)]


def normalize_color(value) -> str | None:
    """An omero colour as upper-case "RRGGBB" with the legacy remap applied, or None.

    Accepts "RRGGBB", "#rrggbb" or "RRGGBBAA" strings (alpha dropped), and ints:
    0..0xFFFFFF as 0xRRGGBB, anything else as OMERO's signed 32-bit RGBA.
    """
    if isinstance(value, bool):
        return None
    if isinstance(value, (int, np.integer)):
        n = int(value)
        if not 0 <= n <= 0xFFFFFF:
            n = (n & 0xFFFFFFFF) >> 8
        h = f"{n:06X}"
    elif isinstance(value, str):
        h = value.strip().lstrip("#").upper()
        if len(h) not in (6, 8):
            return None
        try:
            int(h, 16)
        except ValueError:
            return None
        h = h[:6]
    else:
        return None
    return LEGACY_COLOR_REMAP.get(h, h)


def color_rgb(hex_color: str) -> tuple[float, float, float]:
    """(r, g, b) floats in 0..1 from "RRGGBB" (as `normalize_color` returns)."""
    h = hex_color.strip().lstrip("#")
    r, g, b = (int(h[i : i + 2], 16) / 255.0 for i in (0, 2, 4))
    return r, g, b


def napari_colormap_name(hex_color) -> str:
    """napari's colormap name for an omero colour: green, magenta, cyan, yellow
    for the palette (legacy red shows as magenta), else "gray"."""
    return _NAPARI_COLORMAPS.get(normalize_color(hex_color) or "", "gray")
