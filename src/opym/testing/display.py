# Ruff style: Compliant
"""Golden vectors for `opym.display`, for consumers' tests (numpy only).

A consumer that imports `opym.display` still keeps a test that its viewer shows
what these vectors say; a consumer that cannot import it checks against them.
Every vector is worked out from the rule stated beside it (e.g. 255 * sqrt(7 /
4095) = 10.5, so code 11), not by calling the function under test.
"""

from __future__ import annotations

import numpy as np

# --- sqrt8 --------------------------------------------------------------------

# Counts covering the ends, a step the rounding must get right, and a value past vmax.
SQRT_COUNTS = np.array([0, 1, 7, 60, 225, 1000, 4095, 8000, 65535], dtype=np.uint16)

# code = round(255 * sqrt(min(v, vmax) / vmax)), rounded half to even (np.rint).
# vmax 4095, the floor of the range: 4095 and above (8000, 65535) are 255.
SQRT_VMAX_4095 = 4095
SQRT_CODES_4095 = np.array([0, 4, 11, 31, 60, 126, 255, 255, 255], dtype=np.uint8)
# vmax 16383, for a channel whose first volume peaked near 8000.
SQRT_VMAX_16383 = 16383
SQRT_CODES_16383 = np.array([0, 2, 5, 15, 30, 63, 127, 178, 255], dtype=np.uint8)

# (first volume's maximum, vmax): 2^k - 1 >= 2 * max + 1, within [4095, 65535].
SQRT_VMAX_CASES = (
    (0, 4095),
    (7, 4095),
    (2047, 4095),
    (2048, 8191),
    (4095, 8191),
    (4096, 16383),
    (8000, 16383),
    (32767, 65535),
    (65535, 65535),
)

# --- linear8 (ChimeraX) --------------------------------------------------------

# (top of the display range, shift) at 120 levels: floor(log2(top / 120)), >= 0.
LINEAR_SHIFT_CASES = (
    (0, 0),
    (100, 0),
    (239, 0),
    (240, 1),
    (479, 1),
    (480, 2),
    (1064, 3),
    (65535, 9),
)

# --- palette -------------------------------------------------------------------

# (as a store names it, normalized "RRGGBB"); None = unusable.
COLOR_CASES = (
    ("00FF00", "00FF00"),
    ("ff00ff", "FF00FF"),
    ("#00B3FF", "00B3FF"),
    ("FFC400FF", "FFC400"),  # alpha dropped
    ("FF3D3D", "FF00FF"),  # legacy red for channel 1 is magenta
    ("ff3d3d", "FF00FF"),
    (0x00FF00, "00FF00"),
    (0xFF3D3D, "FF00FF"),
    ("", None),
    ("FF00F", None),
    ("GGGGGG", None),
    (None, None),
    (True, None),
)
# napari colormap by normalized colour.
NAPARI_COLORMAPS = {
    "00FF00": "green",
    "FF00FF": "magenta",
    "FF3D3D": "magenta",
    "00B3FF": "cyan",
    "FFC400": "yellow",
    "123456": "gray",
}
