# Ruff style: Compliant
"""
Interactive viewers for OPM data, designed for use in Jupyter notebooks.
"""

from collections.abc import Callable

import ipywidgets as widgets
import matplotlib.pyplot as plt
import numpy as np
from IPython.display import display


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
            fig_comp.canvas.draw()

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
