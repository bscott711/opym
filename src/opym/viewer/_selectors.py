# Ruff style: Compliant
"""
Interactive ROI selection tools using matplotlib.
"""

from typing import cast

import matplotlib.patches as patches
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.backend_bases import MouseButton
from matplotlib.widgets import RectangleSelector


class ROISelector:
    """
    An interactive (blocking) ROI selector for Jupyter notebooks.

    Usage:
        selector = ROISelector(
            mip_data,
            vmin,
            vmax,
            require_top=True,
            require_bottom=True,
        )
        # The plot will display.
        rois = selector.get_rois() # Blocks until required ROIs are drawn
    """

    def __init__(
        self,
        mip_data: np.ndarray,
        vmin: float,
        vmax: float,
        require_top: bool = True,
        require_bottom: bool = True,
    ):
        self.mip_data = mip_data
        self.vmin = vmin
        self.vmax = vmax
        self.require_top = require_top
        self.require_bottom = require_bottom
        self.roi_slices: list[tuple[slice, slice]] = []
        self.first_roi_dims: tuple[int, int] | None = None

        # Determine total expected ROIs
        self.expected_count = 0
        if self.require_top:
            self.expected_count += 1
        if self.require_bottom:
            self.expected_count += 1

        if self.expected_count == 0:
            print("⚠️ No ROIs required based on channel selection.")
            return

        self.fig, self.ax = plt.subplots(figsize=(8, 8))

        # Dynamic instructions
        title_parts = []
        if self.require_top:
            title_parts.append("Select TOP (cyan)")
        if self.require_bottom:
            title_parts.append("Select BOTTOM (lime)")

        self.fig.suptitle(" -> ".join(title_parts))

        print("--- 📝 INSTRUCTIONS ---")
        if self.require_top:
            print("1. Click and drag to draw a box around the TOP (cyan) ROI.")
            if self.require_bottom:
                print(
                    "2. Click and drag near the center of the BOTTOM (lime) ROI."
                    " The box size will be matched automatically."
                )
        elif self.require_bottom:
            print("1. Click and drag to draw a box around the BOTTOM (lime) ROI.")

        print("When finished, call the .get_rois() method on this object.")

        self.ax.imshow(self.mip_data, cmap="gray", vmin=self.vmin, vmax=self.vmax)
        self.ax.set_axis_off()

        self.selector = RectangleSelector(
            self.ax,
            self.on_select,
            useblit=True,
            button=[MouseButton.LEFT],
            minspanx=5,
            minspany=5,
            spancoords="pixels",
            interactive=True,
        )
        plt.show()

    def on_select(self, eclick, erelease):
        """Callback for rectangle selector."""
        if len(self.roi_slices) >= self.expected_count:
            print("ℹ️ Required ROIs already selected. Re-run cell to start over.")
            return

        x_start_drawn, y_start_drawn = int(eclick.xdata), int(eclick.ydata)
        x_end_drawn, y_end_drawn = int(erelease.xdata), int(erelease.ydata)

        # Determine if this is the Top or Bottom selection
        is_top_selection = False

        if self.require_top and len(self.roi_slices) == 0:
            is_top_selection = True

        if is_top_selection:
            # --- Defining TOP ROI ---
            y_start, y_end = (
                min(y_start_drawn, y_end_drawn),
                max(y_start_drawn, y_end_drawn),
            )
            x_start, x_end = (
                min(x_start_drawn, x_end_drawn),
                max(x_start_drawn, x_end_drawn),
            )
            height = y_end - y_start
            width = x_end - x_start
            self.first_roi_dims = (height, width)

            rect = patches.Rectangle(
                (x_start, y_start),
                width,
                height,
                linewidth=2,
                edgecolor="cyan",
                facecolor="none",
            )
            self.ax.add_patch(rect)
            print(f"✅ ROI (TOP) selected. Size: ({height}h, {width}w)")

            if self.require_bottom:
                print("  Draw the SECOND (BOTTOM) ROI. Its size will be constrained.")

        else:
            # --- Defining BOTTOM ROI ---
            # Use constrained size if we already defined a top ROI
            if self.first_roi_dims:
                roi_height, roi_width = cast(tuple[int, int], self.first_roi_dims)
                y_center = y_start_drawn + (y_end_drawn - y_start_drawn) / 2
                x_center = x_start_drawn + (x_end_drawn - x_start_drawn) / 2

                y_start = int(y_center - roi_height / 2)
                y_end = y_start + roi_height
                x_start = int(x_center - roi_width / 2)
                x_end = x_start + roi_width
                print(
                    f"✅ ROI (BOTTOM) selected. "
                    f"Size snapped to ({roi_height}h, {roi_width}w)."
                )
            else:
                # This happens if require_top=False. User draws bottom freely.
                y_start, y_end = (
                    min(y_start_drawn, y_end_drawn),
                    max(y_start_drawn, y_end_drawn),
                )
                x_start, x_end = (
                    min(x_start_drawn, x_end_drawn),
                    max(x_start_drawn, x_end_drawn),
                )
                height = y_end - y_start
                width = x_end - x_start
                print(f"✅ ROI (BOTTOM) selected. Size: ({height}h, {width}w)")

            rect = patches.Rectangle(
                (x_start, y_start),
                x_end - x_start,
                y_end - y_start,
                linewidth=2,
                edgecolor="lime",
                facecolor="none",
            )
            self.ax.add_patch(rect)

        y_slice = slice(y_start, y_end)
        x_slice = slice(x_start, x_end)
        self.roi_slices.append((y_slice, x_slice))
        print(f"  Slice: (slice({y_start}, {y_end}), slice({x_start}, {x_end}))")

        if len(self.roi_slices) == self.expected_count:
            self.selector.set_active(False)
            self.fig.suptitle(
                "All required ROIs selected. Call .get_rois() to continue."
            )
            print("\n🎉 Selection complete! You can now call .get_rois()")

    def get_rois(self) -> list[tuple[slice, slice]]:
        """Returns the selected ROIs. Blocks if required ROIs are not yet selected."""
        while len(self.roi_slices) < self.expected_count:
            plt.pause(0.1)  # Wait for user interaction
        return self.roi_slices


def visualize_alignment(
    mip_data: np.ndarray,
    top_roi: tuple[slice, slice],
    bottom_roi: tuple[slice, slice],
    vmin: float,
    vmax: float,
):
    """Displays the two ROIs stacked vertically on the MIP."""

    # --- FIX: Calculate figsize based on ROI aspect ratio ---
    try:
        # Get shape from the first ROI
        y_slice, x_slice = top_roi
        height = y_slice.stop - y_slice.start
        width = x_slice.stop - x_slice.start
        aspect_ratio = height / width
    except Exception:
        # Fallback in case of slice error
        aspect_ratio = 1

    base_width = 7  # Base width in inches
    # Height for both plots, plus 1.0 inch for titles/padding
    fig_height = (base_width * aspect_ratio) * 2 + 1.0

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(base_width, fig_height))
    # --------------------------------------------------------

    fig.suptitle("Final Aligned ROIs")

    # Top ROI
    ax1.imshow(mip_data[top_roi[0], top_roi[1]], cmap="gray", vmin=vmin, vmax=vmax)
    ax1.set_title("Top ROI (C=0)")
    ax1.set_axis_off()

    # Bottom ROI
    ax2.imshow(
        mip_data[bottom_roi[0], bottom_roi[1]], cmap="gray", vmin=vmin, vmax=vmax
    )
    ax2.set_title("Bottom ROI (C=1)")
    ax2.set_axis_off()

    plt.tight_layout()
    plt.show()


def interactive_roi_selector(
    mip_data: np.ndarray,
    vmin: float,
    vmax: float,
    require_top: bool = True,
    require_bottom: bool = True,
) -> ROISelector:
    """
    Creates and displays an interactive ROISelector instance.

    Args:
        mip_data: The 2D (Y, X) MIP array to display.
        vmin: The minimum display value for contrast.
        vmax: The maximum display value for contrast.
        require_top: Whether the user must select a Top ROI.
        require_bottom: Whether the user must select a Bottom ROI.

    Returns:
        An instance of the ROISelector class. Call .get_rois() on
        this object to retrieve the selected ROIs.
    """
    # This will initialize the class, which in turn creates the plot
    return ROISelector(mip_data, vmin, vmax, require_top, require_bottom)
