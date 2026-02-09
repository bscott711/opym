# Ruff style: Compliant
"""
opym.ui - Reusable UI components for Jupyter Notebooks.
"""

import ipywidgets as widgets

from .utils import OutputFormat


def create_crop_settings_ui(
    n_channels: int,
) -> tuple[
    widgets.VBox, dict[int, widgets.Checkbox], widgets.Dropdown, widgets.Checkbox
]:
    """
    Generates the UI for channel selection and output format configuration.

    Args:
        n_channels (int): Total number of channels in the source file.

    Returns:
        Tuple containing:
            - The main container widget (VBox) to display.
            - A dictionary mapping channel IDs to their Checkbox widgets.
            - The OutputFormat Dropdown widget.
            - The Rotation Checkbox widget.
    """

    # Calculate excitations (assuming 4 channels per excitation for OPM)
    n_excitations = max(1, n_channels // 4)
    # Fallback if channels < 4, strictly speaking logic depends on your cam setup
    if n_channels < 4:
        n_excitations = 1

    checks: dict[int, widgets.Checkbox] = {}
    ui_rows: list[widgets.Widget] = []

    # 1. Build Channel Grid
    for exc in range(n_excitations):
        base_id = exc * 4
        # Create row of 4 checkboxes
        row_widgets = [
            widgets.Checkbox(value=True, description=f"C{base_id} (Bot, Cam 1)"),
            widgets.Checkbox(value=True, description=f"C{base_id + 1} (Top, Cam 1)"),
            widgets.Checkbox(value=True, description=f"C{base_id + 2} (Top, Cam 2)"),
            widgets.Checkbox(value=True, description=f"C{base_id + 3} (Bot, Cam 2)"),
        ]

        # Map IDs and add to layout
        for i, w in enumerate(row_widgets):
            actual_ch_id = base_id + i
            if actual_ch_id < n_channels:
                checks[actual_ch_id] = w

        # Visual Grouping
        ui_rows.append(
            widgets.VBox(
                [
                    widgets.Label(f"<b>Excitation Group {exc + 1}</b>"),
                    widgets.HBox(row_widgets[:2]),
                    widgets.HBox(row_widgets[2:]),
                ]
            )
        )

    # 2. Build Advanced Settings
    fmt_widget = widgets.Dropdown(
        options=[(f.value, f) for f in OutputFormat],
        value=OutputFormat.TIFF_SERIES,
        description="Format:",
        style={"description_width": "initial"},
    )

    rot_widget = widgets.Checkbox(value=True, description="Rotate 90° CCW")

    # 3. Assemble UI (Vertical Layout instead of Accordion)
    settings_box = widgets.VBox([fmt_widget, rot_widget])
    channels_box = widgets.VBox(ui_rows)

    # Create explicit headers since we removed the Accordion titles
    header_channels = widgets.HTML("<h3>Channel Selection</h3>")
    header_settings = widgets.HTML("<h3>Advanced Output Settings</h3>")

    ui = widgets.VBox(
        [
            header_channels,
            channels_box,
            widgets.HTML("<hr>"),  # Visual separator
            header_settings,
            settings_box,
        ]
    )

    return ui, checks, fmt_widget, rot_widget


def create_deskew_ui(
    detected_z: float = 1.0, default_angle: float = 31.8, default_pixel: float = 0.136
) -> tuple[
    widgets.FloatText,
    widgets.FloatText,
    widgets.FloatText,
    widgets.Button,
    widgets.Label,
    widgets.Output,
]:
    """Creates the UI widgets for the deskew step."""

    style = {"description_width": "initial"}

    w_z = widgets.FloatText(
        value=detected_z, description="Z Step (µm):", step=0.1, style=style
    )
    w_angle = widgets.FloatText(
        value=default_angle, description="Angle (deg):", step=0.1, style=style
    )
    w_px = widgets.FloatText(
        value=default_pixel, description="XY Pixel (µm):", step=0.001, style=style
    )

    run_btn = widgets.Button(
        description="Submit Deskew & Update JSON",
        button_style="success",
        icon="play",
        layout=widgets.Layout(width="50%"),
    )

    status_label = widgets.Label(value="Ready to submit.")
    output_area = widgets.Output()

    return w_z, w_angle, w_px, run_btn, status_label, output_area
