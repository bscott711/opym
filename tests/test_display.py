"""opym.display: the sqrt8 rule, the linear8 rule (ChimeraX) and the palette,
against golden vectors worked out by hand (opym.testing.display)."""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor

import numpy as np
import pytest

from opym import display
from opym.testing import display as golden

# --- sqrt8 ---------------------------------------------------------------------


@pytest.mark.parametrize(
    ("vmax", "codes"),
    [
        (golden.SQRT_VMAX_4095, golden.SQRT_CODES_4095),
        (golden.SQRT_VMAX_16383, golden.SQRT_CODES_16383),
    ],
)
def test_sqrt_codes_match_the_golden_vectors(vmax, codes):
    out = display.to_8bit_sqrt(golden.SQRT_COUNTS, vmax)
    assert out.dtype == np.uint8
    np.testing.assert_array_equal(out, codes)
    np.testing.assert_array_equal(display.sqrt_lut(vmax)[golden.SQRT_COUNTS], codes)
    np.testing.assert_array_equal(
        display.DisplayEncoding.sqrt(vmax).lut()[golden.SQRT_COUNTS], codes
    )


def test_sqrt_lut_is_the_whole_uint16_range_and_monotonic():
    lut = display.sqrt_lut(4095)
    assert lut.shape == (65536,) and lut.dtype == np.uint8
    assert lut[0] == 0 and lut[4095] == 255 and lut[65535] == 255
    assert (np.diff(lut.astype(int)) >= 0).all()


@pytest.mark.parametrize(("peak", "vmax"), golden.SQRT_VMAX_CASES)
def test_sqrt_vmax_for(peak, vmax):
    vol = np.zeros((2, 3, 4), np.uint16)
    vol[0, 0, 0] = peak
    assert display.sqrt_vmax_for(vol) == vmax
    assert display.DisplayEncoding.sqrt_for(vol).vmax == vmax


def test_sqrt_vmax_for_an_empty_volume_is_the_floor():
    assert display.sqrt_vmax_for(np.zeros((0, 3, 4), np.uint16)) == display.SQRT_VMAX


def test_sqrt_decode_inverts_the_codes_within_a_step():
    vmax = 4095
    counts = np.arange(0, 4096, dtype=np.uint16)
    back = display.sqrt_decode(display.to_8bit_sqrt(counts, vmax), vmax)
    assert back.dtype == np.float32
    # rounding to a code is off by half a code step, 2 sqrt(v vmax) / 255, at most
    # (plus the first code's width at the bottom)
    step = 2 * np.sqrt(np.maximum(counts, 1).astype(float) * vmax) / 255
    assert (np.abs(back - counts.astype(float)) <= step).all()
    assert display.sqrt_decode([0, 255], vmax).tolist() == [0.0, float(vmax)]


def test_texture_clim_is_counts_over_vmax_times_255():
    assert display.sqrt_texture_clim((0, 4095), 4095) == (0.0, 255.0)
    lo, hi = display.sqrt_texture_clim((100, 2000), 8191)
    assert (lo, hi) == (100 * 255 / 8191, 2000 * 255 / 8191)
    enc = display.DisplayEncoding.sqrt(8191)
    assert enc.texture_clim((100, 2000)) == (lo, hi)


def test_sqrt_colormap_shows_codes_at_linear_intensity():
    vmax = 4095
    controls, colors = display.sqrt_colormap_controls((0.0, 1.0, 0.0), vmax)
    assert controls.shape == (256,) and colors.shape == (256, 4)
    assert controls[0] == 0.0 and controls[-1] == 1.0
    codes = display.to_8bit_sqrt(golden.SQRT_COUNTS, vmax)
    shown = colors[codes.astype(int), 1]  # green channel at each count's code
    decoded = display.sqrt_decode(codes, vmax)
    np.testing.assert_allclose(shown, decoded / vmax, atol=1e-6)
    assert (colors[:, 0] == 0).all() and (colors[:, 3] == 1).all()
    # a contrast window (counts) clips below and above
    _, c2 = display.sqrt_colormap_controls((1, 1, 1), vmax, clim=(1000, 3000))
    assert c2[0, 0] == 0.0 and c2[-1, 0] == 1.0


def test_display_encoding_record_round_trip_and_linear():
    sqrt = display.DisplayEncoding.sqrt(8191)
    assert sqrt.record() == {"encoding": "sqrt", "vmax": 8191.0}
    assert display.DisplayEncoding.from_record(sqrt.record()) == sqrt
    old = display.DisplayEncoding.from_record({"shift": 3, "levels": 120})
    assert old == display.DisplayEncoding.linear(3) and old.vmax == 255 << 3
    assert old.record() == {"shift": 3}
    for bad in ({"nope": 1}, {"encoding": "sqrt"}, None, 5):
        assert display.DisplayEncoding.from_record(bad) is None
    assert display.DisplayEncoding.linear(2).lut()[1000] == 250
    assert display.DisplayEncoding.linear(2).lut()[65535] == 255


# --- linear8 -------------------------------------------------------------------


@pytest.mark.parametrize(("top", "shift"), golden.LINEAR_SHIFT_CASES)
def test_display_shift_leaves_about_120_levels(top, shift):
    assert display.DISPLAY_LEVELS == 120
    assert display.display_shift(top) == shift
    if top >= 240:
        assert 120 <= top / 2**shift < 240


def test_display_shift_with_other_levels():
    assert display.display_shift(1000, levels=250) == 2


def test_to_8bit_linear_is_shift_and_clip():
    rng = np.random.default_rng(0)
    m = rng.integers(0, 65535, (40, 5, 6), dtype=np.uint16)
    for shift in (0, 1, 3, 8):
        ref = np.minimum(m >> shift, 255).astype(np.uint8)
        np.testing.assert_array_equal(display.to_8bit_linear(m, shift), ref)
        with ThreadPoolExecutor(3) as pool:
            np.testing.assert_array_equal(display.to_8bit_linear(m, shift, pool), ref)
    assert display.to_8bit_linear(m, 2).dtype == np.uint8


def test_to_8bit_linear_does_not_touch_the_source():
    m = np.full((20, 3, 3), 60000, np.uint16)
    display.to_8bit_linear(m, 0)
    display.to_8bit_linear(m, 2)
    assert (m == 60000).all()


def test_eight_bit_ok_only_for_unsigned_16():
    assert display.eight_bit_ok(np.dtype("u2"))
    for other in ("u1", "i2", "f4", "u4"):
        assert not display.eight_bit_ok(np.dtype(other))


# --- palette -------------------------------------------------------------------


def test_channel_palette_second_channel_is_magenta_not_red():
    assert display.CHANNEL_COLORS == ("00FF00", "FF00FF", "00B3FF", "FFC400")
    assert display.default_channel_color(1) == "FF00FF"
    assert display.default_channel_color(4) == "00FF00"  # wraps
    assert display.LEGACY_COLOR_REMAP == {"FF3D3D": "FF00FF"}


@pytest.mark.parametrize(("value", "want"), golden.COLOR_CASES)
def test_normalize_color(value, want):
    assert display.normalize_color(value) == want


def test_color_rgb():
    assert display.color_rgb("FF00FF") == (1.0, 0.0, 1.0)
    assert display.color_rgb("#00B3FF") == (0.0, 179 / 255, 1.0)


@pytest.mark.parametrize(("color", "name"), list(golden.NAPARI_COLORMAPS.items()))
def test_napari_colormap_name(color, name):
    assert display.napari_colormap_name(color) == name


def test_napari_colormap_name_unknown_or_missing_is_gray():
    assert display.napari_colormap_name(None) == "gray"
    assert display.napari_colormap_name("nonsense") == "gray"


# --- old paths -----------------------------------------------------------------


def test_old_import_paths_still_work():
    from opym import live_view

    assert live_view.DisplayEncoding is display.DisplayEncoding
    assert live_view.SQRT_VMAX is display.SQRT_VMAX == 4095
    from opym import ome_zarr_writer as w

    assert w.CHANNEL_COLORS is display.CHANNEL_COLORS
