# Ruff style: Compliant
"""
Interactive viewers for OPM data, designed for use in Jupyter notebooks.
"""

from collections.abc import Callable
from pathlib import Path
from typing import cast

import ipywidgets as widgets
import matplotlib.patches as patches
import matplotlib.pyplot as plt
import numpy as np
import tifffile
import zarr
from IPython.display import display
from matplotlib.backend_bases import MouseButton
from matplotlib.widgets import RectangleSelector
from tqdm.auto import tqdm


def single_channel_viewer(
    get_stack: Callable,
    T_max: int,
    Z_max: int,
    C_max: int,
    Y: int,
    X: int,
):
    """Displays the interactive single-channel viewer."""
    try:
        # --- 1. Plotting Setup ---
        plt.ioff()
        base_width = 8
        aspect = Y / X
        fig_height = base_width * aspect

        fig, ax = plt.subplots(1, 1, figsize=(base_width, fig_height))
        fig.subplots_adjust(left=0, right=1, top=1, bottom=0)
        ax.set_axis_off()

        initial_stack = get_stack(0, 0)
        initial_plane = initial_stack[Z_max // 2, :, :]
        global_min, global_max = initial_stack.min(), initial_stack.max()
        initial_vmin, initial_vmax = np.percentile(initial_stack, (0.1, 99.9))

        img = ax.imshow(
            initial_plane, cmap="gray", vmin=initial_vmin, vmax=initial_vmax
        )

        # --- 2. Create Widgets ---
        t_slider = widgets.IntSlider(
            description="Time:", min=0, max=T_max, step=1, value=0
        )
        c_slider = widgets.IntSlider(
            description="Channel:", min=0, max=C_max, step=1, value=0
        )
        z_slider = widgets.IntSlider(
            description="Z-Slice:",
            min=0,
            max=Z_max,
            step=1,
            value=Z_max // 2,
            layout=widgets.Layout(width="90%"),
        )
        contrast_slider = widgets.FloatRangeSlider(
            description="Contrast:",
            min=global_min,
            max=global_max,
            value=(initial_vmin, initial_vmax),
            step=1.0,
            readout_format=".0f",
            layout=widgets.Layout(width="80%"),
        )
        lock_contrast_checkbox = widgets.Checkbox(
            value=False,
            description="Lock Contrast",
            indent=False,
            layout=widgets.Layout(width="120px"),
        )
        title_label = widgets.Label(value=f"T=0, Z={Z_max // 2}, C=0")

        # --- 3. Define Update Callbacks ---
        def update_plot(change):
            t = t_slider.value
            c = c_slider.value
            z = z_slider.value
            contrast = contrast_slider.value

            is_locked = lock_contrast_checkbox.value
            slider_changed = "owner" in change and (
                change["owner"] is t_slider or change["owner"] is c_slider
            )

            if slider_changed and not is_locked:
                new_stack = get_stack(t, c)
                new_min, new_max = new_stack.min(), new_stack.max()
                new_vmin, new_vmax = np.percentile(new_stack, (0.1, 99.9))

                contrast_slider.unobserve(update_plot, "value")
                contrast_slider.min = new_min
                contrast_slider.max = new_max
                contrast_slider.value = (new_vmin, new_vmax)
                contrast_slider.observe(update_plot, "value")
                contrast = (new_vmin, new_vmax)

            stack = get_stack(t, c)
            plane = stack[z, :, :]

            img.set_data(plane)
            img.set_clim(vmin=contrast[0], vmax=contrast[1])
            title_label.value = f"T={t}, Z={z}, C={c}"
            fig.canvas.draw_idle()

        t_slider.observe(update_plot, "value")
        c_slider.observe(update_plot, "value")
        z_slider.observe(update_plot, "value")
        contrast_slider.observe(update_plot, "value")

        # --- 4. Display the UI ---
        ui = widgets.VBox(
            [
                widgets.HBox([t_slider, c_slider, title_label]),
                z_slider,
                widgets.HBox([contrast_slider, lock_contrast_checkbox]),
                fig.canvas,
            ]
        )
        display(ui)
        plt.ion()

    except Exception as e:
        print(f"❌ An error occurred in single_channel_viewer: {e}")


def composite_viewer(
    get_stack: Callable,
    T_max: int,
    Z_max: int,
    C_max: int,
    Y: int,
    X: int,
):
    """Displays the interactive composite (multi-channel) viewer."""
    try:
        # --- 1. Plotting Setup ---
        plt.ioff()
        base_width = 8
        aspect = Y / X
        fig_height = base_width * aspect

        fig_comp, ax_comp = plt.subplots(1, 1, figsize=(base_width, fig_height))
        fig_comp.subplots_adjust(left=0, right=1, top=1, bottom=0)
        ax_comp.set_axis_off()
        ax_comp.set_facecolor("black")

        initial_img = np.zeros((Y, X, 3), dtype=float)
        img_comp = ax_comp.imshow(initial_img, vmin=0.0, vmax=1.0)

        # --- 2. Define Helper for Normalization ---
        def normalize_plane(plane, vmin, vmax):
            if vmin >= vmax:
                return np.zeros_like(plane, dtype=float)
            norm_plane = (plane.astype(float) - vmin) / (vmax - vmin)
            return np.clip(norm_plane, 0.0, 1.0)

        # --- 3. Create Master Widgets ---
        t_slider_comp = widgets.IntSlider(
            description="Time:",
            min=0,
            max=T_max,
            step=1,
            value=0,
            layout=widgets.Layout(width="40%"),
        )
        z_slider_comp = widgets.IntSlider(
            description="Z-Slice:",
            min=0,
            max=Z_max,
            step=1,
            value=Z_max // 2,
            layout=widgets.Layout(width="40%"),
        )
        refresh_button = widgets.Button(
            description="Refresh Contrast", layout=widgets.Layout(width="140px")
        )
        title_label_comp = widgets.Label(value=f"T=0, Z={Z_max // 2}")

        # --- 4. Create Per-Channel Widgets ---
        color_map_options = {
            "Blue": np.array([0.0, 0.0, 1.0]),
            "Green": np.array([0.0, 1.0, 0.0]),
            "Red": np.array([1.0, 0.0, 0.0]),
            "Purple": np.array([1.0, 0.0, 1.0]),
            "Cyan": np.array([0.0, 1.0, 1.0]),
            "Yellow": np.array([1.0, 1.0, 0.0]),
            "Gray": np.array([1.0, 1.0, 1.0]),
        }

        def create_channel_controls(channel_num, default_color, default_on=True):
            stack = get_stack(0, channel_num)
            c_min, c_max = stack.min(), stack.max()
            c_vmin, c_vmax = np.percentile(stack, (0.1, 99.9))

            checkbox = widgets.Checkbox(
                value=default_on,
                description=f"C{channel_num}",
                indent=False,
                layout=widgets.Layout(width="50px"),
            )
            cmap = widgets.Dropdown(
                options=list(color_map_options.keys()),
                value=default_color,
                layout=widgets.Layout(width="100px"),
            )
            contrast = widgets.FloatRangeSlider(
                description="",
                min=c_min,
                max=c_max,
                value=(c_vmin, c_vmax),
                step=1.0,
                readout_format=".0f",
                layout=widgets.Layout(width="300px"),
            )
            return checkbox, cmap, contrast

        c0_on, c0_cmap, c0_contrast = create_channel_controls(0, "Blue")
        c1_on, c1_cmap, c1_contrast = create_channel_controls(1, "Green")
        c2_on, c2_cmap, c2_contrast = create_channel_controls(2, "Red")
        c3_on, c3_cmap, c3_contrast = create_channel_controls(3, "Purple")

        channel_controls = [
            (c0_on, c0_cmap, c0_contrast),
            (c1_on, c1_cmap, c1_contrast),
            (c2_on, c2_cmap, c2_contrast),
            (c3_on, c3_cmap, c3_contrast),
        ]

        # --- 5. Define Update Callbacks ---
        def update_composite_plot(change=None):
            t = t_slider_comp.value
            z = z_slider_comp.value

            final_image = np.zeros((Y, X, 3), dtype=float)

            for i, (checkbox, cmap_widget, contrast_widget) in enumerate(
                channel_controls
            ):
                if checkbox.value:
                    stack = get_stack(t, i)
                    plane = stack[z, :, :]
                    vmin, vmax = contrast_widget.value
                    norm_plane = normalize_plane(plane, vmin, vmax)
                    color_vector = color_map_options[cmap_widget.value]
                    final_image += norm_plane[..., np.newaxis] * color_vector

            final_image = np.clip(final_image, 0.0, 1.0)
            img_comp.set_data(final_image)
            title_label_comp.value = f"T={t}, Z={z}"

            # --- FIX: Use draw_idle() to match the working viewer ---
            fig_comp.canvas.draw_idle()
            # --- End Fix ---

        def on_refresh_button_clicked(b):
            t = t_slider_comp.value
            print(f"Refreshing contrast ranges for T={t}...")
            for i, (checkbox, cmap_widget, contrast_widget) in enumerate(
                channel_controls
            ):
                stack = get_stack(t, i)
                new_min, new_max = stack.min(), stack.max()
                new_vmin, new_vmax = np.percentile(stack, (0.1, 99.9))

                contrast_widget.unobserve(update_composite_plot, "value")
                contrast_widget.min = new_min
                contrast_widget.max = new_max
                contrast_widget.value = (new_vmin, new_vmax)
                contrast_widget.observe(update_composite_plot, "value")

            update_composite_plot()
            print("Contrast refreshed.")

        # --- 6. Link Callbacks and Display UI ---
        t_slider_comp.observe(update_composite_plot, "value")
        z_slider_comp.observe(update_composite_plot, "value")
        refresh_button.on_click(on_refresh_button_clicked)

        for checkbox, cmap_widget, contrast_widget in channel_controls:
            checkbox.observe(update_composite_plot, "value")
            cmap_widget.observe(update_composite_plot, "value")
            contrast_widget.observe(update_composite_plot, "value")

        ui = widgets.VBox(
            [
                widgets.HBox(
                    [
                        t_slider_comp,
                        z_slider_comp,
                        refresh_button,
                        title_label_comp,
                    ]
                ),
                widgets.HBox([c0_on, c0_cmap, c0_contrast]),
                widgets.HBox([c1_on, c1_cmap, c1_contrast]),
                widgets.HBox([c2_on, c2_cmap, c2_contrast]),
                widgets.HBox([c3_on, c3_cmap, c3_contrast]),
                fig_comp.canvas,
            ]
        )
        display(ui)
        update_composite_plot()
        plt.ion()

    except Exception as e:
        print(f"❌ An error occurred in composite_viewer: {e}")


# --- NEW FUNCTIONS FROM REFACTORING ---


def create_mip(
    file_path: Path | str, t_index: int = 0
) -> tuple[np.ndarray, float, float, zarr.Array, int]:
    """
    Opens a 5D OME-TIF as a lazy Zarr array and computes the
    max intensity projection for a specific timepoint.

    Args:
        file_path: Path to the OME-TIF file.
        t_index: The T index to project.

    Returns:
        A tuple containing:
        - mip_data (np.ndarray): The 2D (Y, X) MIP array.
        - vmin (float): 1st percentile for contrast.
        - vmax (float): 99.9th percentile for contrast.
        - lazy_data (zarr.Array): The opened 5D zarr array.
        - t_max (int): The maximum T index.
    """
    file_path = Path(file_path)
    print(f"Opening {file_path.name} as lazy Zarr array...")

    try:
        # --- FINAL FIX: Use the original, correct Zarr opening logic ---
        store = tifffile.TiffFile(file_path).series[0].aszarr()
        lazy_data = zarr.open(store, mode="r")
        # -------------------------------------------------------------

    except Exception as e:
        print(f"❌ ERROR: Could not open {file_path.name} as zarr.")
        print(f"  Details: {e}")
        raise

    # This check will now validate that lazy_data is the array
    if not isinstance(lazy_data, zarr.Array):
        raise TypeError(f"Expected zarr.Array, but found {type(lazy_data)}.")

    T, Z, C, Y, X = lazy_data.shape
    print(f"Full data shape: {lazy_data.shape}")

    if t_index >= T:
        print(f"Warning: T_INDEX ({t_index}) is out of range. Using T=0.")
        t_index = 0

    print(f"Selecting data for Max Projection: (T={t_index})")

    # Use cast to inform Pylance of the type
    stack_to_project = cast(zarr.Array, lazy_data[t_index, :, :, :, :])

    total_planes = Z * C
    print(f"Calculating Max Projection from {total_planes} (Z*C) planes...")

    # Use np.asarray() to explicitly convert slice to ndarray
    plane_0: np.ndarray = np.asarray(stack_to_project[0, 0, :, :])
    z_mip = np.copy(plane_0)

    with tqdm(total=total_planes, desc="  Projecting") as pbar:
        for z in range(Z):
            for c in range(C):
                if z == 0 and c == 0:
                    pbar.update(1)
                    continue

                plane: np.ndarray = np.asarray(stack_to_project[z, c, :, :])
                np.maximum(z_mip, plane, out=z_mip)
                pbar.update(1)

    print("✅ Max Projection complete.")

    if z_mip.max() > 0:
        vmin, vmax = np.percentile(z_mip, [1, 99.9])
        if vmax <= vmin:
            vmax = z_mip.max()
    else:
        vmin, vmax = 0, 1

    print(f"  MIP display range (vmin, vmax): ({vmin:.0f}, {vmax:.0f})")
    return z_mip, float(vmin), float(vmax), lazy_data, T - 1


class ROISelector:
    """
    An interactive (blocking) ROI selector for Jupyter notebooks.

    Usage:
        selector = ROISelector(mip_data, vmin, vmax)
        # The plot will display. Draw two ROIs.
        rois = selector.get_rois() # This blocks until 2 ROIs are drawn
    """

    def __init__(self, mip_data: np.ndarray, vmin: float, vmax: float):
        self.mip_data = mip_data
        self.vmin = vmin
        self.vmax = vmax
        self.roi_slices: list[tuple[slice, slice]] = []
        self.first_roi_dims: tuple[int, int] | None = None
        self.fig, self.ax = plt.subplots(figsize=(8, 8))
        self.fig.suptitle("Select TOP ROI, then BOTTOM ROI")
        print("--- 📝 INSTRUCTIONS ---")
        print("1. Click and drag to draw a box around the TOP (cyan) ROI.")
        print(
            "2. Click and drag near the center of the BOTTOM (lime) ROI."
            " The box size will be matched automatically."
        )
        print("3. When finished, call the .get_rois() method on this object.")

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
        if len(self.roi_slices) >= 2:
            print("ℹ️ Two ROIs already selected. Re-run cell to start over.")
            return

        x_start_drawn, y_start_drawn = int(eclick.xdata), int(eclick.ydata)
        x_end_drawn, y_end_drawn = int(erelease.xdata), int(erelease.ydata)

        if len(self.roi_slices) == 0:
            # --- This is the FIRST (TOP) ROI ---
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
            print(f"✅ ROI #1 (TOP) selected. Size: ({height}h, {width}w)")
            print("  Draw the SECOND (BOTTOM) ROI. Its size will be constrained.")

        else:
            # --- This is the SECOND (BOTTOM) ROI ---
            roi_height, roi_width = cast(tuple[int, int], self.first_roi_dims)
            y_center = y_start_drawn + (y_end_drawn - y_start_drawn) / 2
            x_center = x_start_drawn + (x_end_drawn - x_start_drawn) / 2

            y_start = int(y_center - roi_height / 2)
            y_end = y_start + roi_height
            x_start = int(x_center - roi_width / 2)
            x_end = x_start + roi_width

            rect = patches.Rectangle(
                (x_start, y_start),
                roi_width,
                roi_height,
                linewidth=2,
                edgecolor="lime",
                facecolor="none",
            )
            self.ax.add_patch(rect)
            print(
                f"✅ ROI #2 (BOTTOM) selected. "
                f"Size snapped to ({roi_height}h, {roi_width}w)."
            )
            print("\n🎉 Both ROIs selected! You can now call .get_rois()")

        y_slice = slice(y_start, y_end)
        x_slice = slice(x_start, x_end)
        self.roi_slices.append((y_slice, x_slice))
        print(f"  Slice: (slice({y_start}, {y_end}), slice({x_start}, {x_end}))")

        if len(self.roi_slices) == 2:
            self.selector.set_active(False)
            self.fig.suptitle("ROIs selected. Call .get_rois() to continue.")

    def get_rois(self) -> list[tuple[slice, slice]]:
        """Returns the selected ROIs. Blocks if 2 ROIs are not yet selected."""
        while len(self.roi_slices) < 2:
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
    mip_data: np.ndarray, vmin: float, vmax: float
) -> ROISelector:
    """
    Creates and displays an interactive ROISelector instance.

    This is a helper function that instantiates the ROISelector class,
    which triggers the interactive matplotlib widget.

    Args:
        mip_data: The 2D (Y, X) MIP array to display.
        vmin: The minimum display value for contrast.
        vmax: The maximum display value for contrast.

    Returns:
        An instance of the ROISelector class. Call .get_rois() on
        this object to retrieve the selected ROIs.
    """
    # This will initialize the class, which in turn creates the plot
    return ROISelector(mip_data, vmin, vmax)
