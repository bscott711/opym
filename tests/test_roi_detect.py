import numpy as np

from opym.roi_detect import _clamp_window, auto_detect_rois


def test_clamp_window_shifts_inward_instead_of_shrinking():
    # Regression for the real PDMS_normal_step_003 failure: a bottom-half ROI
    # centered near the frame's far edge used to be silently truncated by
    # `min(dim_max, start + size)` with no compensating shift, producing a
    # window narrower/shorter than `size` and tripping core.py's "ROI shapes
    # do not match" check against the top ROI. The fixed window must always
    # be exactly `size` wide when the frame is at least that big.
    start, stop = _clamp_window(center=2155, size=661, dim_max=2400)
    assert (start, stop) == (1739, 2400)
    assert stop - start == 661

    # Symmetric case: center near the left/top edge.
    start, stop = _clamp_window(center=50, size=200, dim_max=2400)
    assert start == 0
    assert stop - start == 200

    # Frame smaller than the requested window: clamp to the frame, can't
    # recover full size.
    start, stop = _clamp_window(center=50, size=200, dim_max=120)
    assert (start, stop) == (0, 120)

    # Center comfortably inside the frame: untouched, no clamping needed.
    start, stop = _clamp_window(center=1000, size=400, dim_max=2400)
    assert (start, stop) == (800, 1200)


def test_auto_detect_rois_bottom_roi_near_edge_matches_top_shape():
    # Reproduces the real failure shape: a 2400x2400 frame with the bright
    # sample sitting near the bottom edge of the lower camera half, which
    # previously produced a bottom ROI shorter than the top ROI.
    frame = np.zeros((2400, 2400), dtype=np.float32)
    frame[300:450, 400:900] = 1000.0  # top-half blob
    frame[2200:2350, 500:1000] = 1000.0  # bottom-half blob, near the y=2400 edge

    top_roi, bot_roi = auto_detect_rois(frame, master_roi_path=None)

    assert top_roi is not None
    assert bot_roi is not None

    dummy = np.zeros(frame.shape, dtype=np.uint16)
    top_shape = dummy[top_roi[0], top_roi[1]].shape
    bot_shape = dummy[bot_roi[0], bot_roi[1]].shape
    assert top_shape == bot_shape

    # The bottom window must stay within the frame.
    assert bot_roi[0].stop <= frame.shape[0]
    assert bot_roi[1].stop <= frame.shape[1]
