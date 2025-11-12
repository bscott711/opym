# Ruff style: Compliant
"""
Interactive ipywidgets viewers for OPM data.
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
        # --- MODIFIED: Set initial aspect ratio ---
        ax.set_aspect(aspect)
        # ---
        fig.subplots_adjust(left=0, right=1, top=1, bottom=0)
        ax.set_axis_off()

        # --- FIX: Check C_max to avoid crash on empty data ---
        c_initial = 0 if C_max >= 0 else -1
        if c_initial == -1:
            print("Warning: No channels found (C_max < 0). Cannot load data.")
            initial_stack = np.zeros((Z_max + 1, Y, X))
        else:
            initial_stack = get_stack(0, c_initial)
        # --- End fix ---

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
            description="Channel:", min=0, max=C_max, step=1, value=c_initial
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
        # --- ADDED: Rotate Checkbox ---
        rotate_checkbox = widgets.Checkbox(
            value=False,
            description="Rotate 90° CCW",
            indent=False,
            layout=widgets.Layout(width="140px"),
        )
        # ---
        title_label = widgets.Label(value=f"T=0, Z={Z_max // 2}, C={c_initial}")

        # --- 3. Define Update Callbacks ---
        def update_plot(change):
            # Do nothing if no channels exist
            if c_slider.value == -1:
                return

            t = t_slider.value
            c = c_slider.value
            z = z_slider.value
            contrast = contrast_slider.value
            is_locked = lock_contrast_checkbox.value
            is_rotated = rotate_checkbox.value  # <-- GET ROTATION STATE

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

            # --- MODIFIED: Handle rotation and aspect ratio ---
            if is_rotated:
                plane = np.rot90(plane, k=1)
                new_aspect = X / Y  # New H/W is X/Y
            else:
                new_aspect = Y / X  # Original H/W is Y/X

            new_fig_height = base_width * new_aspect
            fig.set_figheight(new_fig_height)
            ax.set_aspect(new_aspect)

            img.set_data(plane)
            # --- ADDED: Autoscale axes to fit new data shape ---
            ax.autoscale(enable=True, tight=True)
            # ---

            img.set_clim(vmin=contrast[0], vmax=contrast[1])
            title_label.value = f"T={t}, Z={z}, C={c}"
            fig.canvas.draw_idle()

        t_slider.observe(update_plot, "value")
        c_slider.observe(update_plot, "value")
        z_slider.observe(update_plot, "value")
        contrast_slider.observe(update_plot, "value")
        rotate_checkbox.observe(update_plot, "value")  # <-- LINK ROTATE

        # --- 4. Display the UI ---
        ui = widgets.VBox(
            [
                widgets.HBox([t_slider, c_slider, title_label]),
                z_slider,
                # --- MODIFIED: Add rotate_checkbox ---
                widgets.HBox(
                    [contrast_slider, lock_contrast_checkbox, rotate_checkbox]
                ),
                # ---
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
        # --- MODIFIED: Set initial aspect ratio ---
        ax_comp.set_aspect(aspect)
        # ---
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
        # --- ADDED: Rotate Checkbox ---
        rotate_checkbox_comp = widgets.Checkbox(
            value=False,
            description="Rotate 90° CCW",
            indent=False,
            layout=widgets.Layout(width="140px"),
        )
        # ---
        title_label_comp = widgets.Label(value=f"T=0, Z={Z_max // 2}")

        # --- 4. Create Per-Channel Widgets (DYNAMICALLY) ---
        color_map_options = {
            "Blue": np.array([0.0, 0.0, 1.0]),
            "Green": np.array([0.0, 1.0, 0.0]),
            "Red": np.array([1.0, 0.0, 0.0]),
            "Purple": np.array([1.0, 0.0, 1.0]),
            "Cyan": np.array([0.0, 1.0, 1.0]),
            "Yellow": np.array([1.0, 1.0, 0.0]),
            "Gray": np.array([1.0, 1.0, 1.0]),
        }
        # --- FIX: Define a list of default colors to cycle through ---
        default_color_cycle = [
            "Blue",
            "Green",
            "Red",
            "Purple",
            "Yellow",
            "Cyan",
        ]

        def create_channel_controls(channel_num, default_color, default_on=True):
            # Check if channel exists before loading
            try:
                stack = get_stack(0, channel_num)
                c_min, c_max = stack.min(), stack.max()
                c_vmin, c_vmax = np.percentile(stack, (0.1, 99.9))
                if c_vmin >= c_vmax:
                    c_vmax = c_max
            except Exception:
                # Fallback if get_stack fails (e.g., file not found)
                c_min, c_max = 0, 1
                c_vmin, c_vmax = 0, 1
                default_on = False  # Don't enable a channel that failed to load

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

        # --- FIX: Dynamically create controls based on C_max ---
        channel_controls = []
        channel_ui_elements = []
        for i in range(C_max + 1):
            color = default_color_cycle[i % len(default_color_cycle)]
            # Default first 4 channels on, others off
            default_on = i < 4

            # Create the controls
            (
                on_checkbox,
                cmap_dropdown,
                contrast_slider,
            ) = create_channel_controls(i, color, default_on)

            # Store controls for logic
            channel_controls.append((on_checkbox, cmap_dropdown, contrast_slider))
            # Store HBox for UI
            channel_ui_elements.append(
                widgets.HBox([on_checkbox, cmap_dropdown, contrast_slider])
            )
        # --- End fix ---

        # --- 5. Define Update Callbacks ---
        def update_composite_plot(change=None):
            t = t_slider_comp.value
            z = z_slider_comp.value
            is_rotated = rotate_checkbox_comp.value  # <-- GET ROTATION STATE

            # --- MODIFIED: Handle rotation, aspect ratio, and image shape ---
            if is_rotated:
                new_aspect = X / Y
                final_image = np.zeros((X, Y, 3), dtype=float)
            else:
                new_aspect = Y / X
                final_image = np.zeros((Y, X, 3), dtype=float)

            new_fig_height = base_width * new_aspect
            fig_comp.set_figheight(new_fig_height)
            ax_comp.set_aspect(new_aspect)
            # ---

            # This loop now works dynamically with any number of channels
            for i, (checkbox, cmap_widget, contrast_widget) in enumerate(
                channel_controls
            ):
                if checkbox.value:
                    try:
                        stack = get_stack(t, i)
                        plane = stack[z, :, :]
                        vmin, vmax = contrast_widget.value
                        norm_plane = normalize_plane(plane, vmin, vmax)

                        # --- MODIFIED: Rotate normalized plane if needed ---
                        if is_rotated:
                            norm_plane = np.rot90(norm_plane, k=1)
                        # ---

                        color_vector = color_map_options[cmap_widget.value]
                        final_image += norm_plane[..., np.newaxis] * color_vector
                    except Exception as e:
                        print(f"Error loading C{i} at T{t}: {e}")
                        # Turn off checkbox if it fails
                        checkbox.value = False

            final_image = np.clip(final_image, 0.0, 1.0)
            img_comp.set_data(final_image)

            # --- ADDED: Autoscale axes to fit new data shape ---
            ax_comp.autoscale(enable=True, tight=True)
            # ---

            title_label_comp.value = f"T={t}, Z={z}"
            fig_comp.canvas.draw_idle()

        def on_refresh_button_clicked(b):
            t = t_slider_comp.value
            print(f"Refreshing contrast ranges for T={t}...")

            # This loop is also dynamic
            for i, (checkbox, cmap_widget, contrast_widget) in enumerate(
                channel_controls
            ):
                try:
                    stack = get_stack(t, i)
                    new_min, new_max = stack.min(), stack.max()
                    new_vmin, new_vmax = np.percentile(stack, (0.1, 99.9))
                    if new_vmin >= new_vmax:
                        new_vmax = new_max

                    contrast_widget.unobserve(update_composite_plot, "value")
                    contrast_widget.min = new_min
                    contrast_widget.max = new_max
                    contrast_widget.value = (new_vmin, new_vmax)
                    contrast_widget.observe(update_composite_plot, "value")
                except Exception as e:
                    print(f"Could not refresh C{i}: {e}")
                    checkbox.value = False  # Disable if refresh fails

            update_composite_plot()
            print("Contrast refreshed.")

        # --- 6. Link Callbacks and Display UI ---
        t_slider_comp.observe(update_composite_plot, "value")
        z_slider_comp.observe(update_composite_plot, "value")
        refresh_button.on_click(on_refresh_button_clicked)
        # --- LINK ROTATE ---
        rotate_checkbox_comp.observe(update_composite_plot, "value")
        # ---

        # This loop links all dynamically created controls
        for checkbox, cmap_widget, contrast_widget in channel_controls:
            checkbox.observe(update_composite_plot, "value")
            cmap_widget.observe(update_composite_plot, "value")
            contrast_widget.observe(update_composite_plot, "value")

        # --- MODIFIED: Add rotate_checkbox_comp ---
        master_controls = widgets.HBox(
            [
                t_slider_comp,
                z_slider_comp,
                refresh_button,
                rotate_checkbox_comp,
                title_label_comp,
            ]
        )
        # ---

        # --- FIX: Dynamically build the VBox ---
        ui = widgets.VBox([master_controls, *channel_ui_elements, fig_comp.canvas])
        # --- End fix ---

        display(ui)
        update_composite_plot()
        plt.ion()

    except Exception as e:
        print(f"❌ An error occurred in composite_viewer: {e}")
