from __future__ import annotations

from pathlib import Path
from typing import Any

import ipywidgets as widgets
import matplotlib.pyplot as plt
import numpy as np
import tifffile
from ipyfilechooser import FileChooser
from IPython.display import clear_output, display


class DeconvolutionViewer:
    """
    Interactive widget to compare Raw vs. Deconvolved volumes with
    synchronized slice views and orthogonal projections.
    """

    def __init__(
        self, start_path: str | Path = "/mmfs2/scratch/SDSMT.LOCAL/bscott/DataUpload"
    ) -> None:
        self.raw_vol: np.ndarray | None = None
        self.decon_vol: np.ndarray | None = None
        self._init_ui(start_path)

    def _init_ui(self, start_path: str | Path) -> None:
        self.fc_raw = FileChooser(
            path=str(start_path), title="<b>Select Raw Volume:</b>"
        )
        self.fc_decon = FileChooser(
            path=str(start_path), title="<b>Select Deconvolved Volume:</b>"
        )

        self.btn_load = widgets.Button(
            description="Load & Compare",
            button_style="primary",
            layout=widgets.Layout(width="200px"),
        )

        # Slicing Controls
        self.z_slider = widgets.IntSlider(
            description="Z-Slice:", min=0, max=1, disabled=True
        )
        self.log_scale = widgets.Checkbox(value=True, description="Log Scale")

        self.out_plot = widgets.Output()

        # Events
        self.btn_load.on_click(self._on_load_clicked)
        self.z_slider.observe(self._update_plot, names="value")
        self.log_scale.observe(self._update_plot, names="value")

        self.ui = widgets.VBox(
            [
                widgets.HBox([self.fc_raw, self.fc_decon]),
                widgets.HBox([self.btn_load, self.z_slider, self.log_scale]),
                self.out_plot,
            ]
        )

    def show(self) -> None:
        """Display the viewer widget."""
        display(self.ui)

    def _on_load_clicked(self, _b: widgets.Button) -> None:
        if not (self.fc_raw.selected and self.fc_decon.selected):
            return

        # Load volumes
        self.raw_vol = tifffile.imread(self.fc_raw.selected).astype(np.float32)
        self.decon_vol = tifffile.imread(self.fc_decon.selected).astype(np.float32)

        # Ensure 3D
        if self.raw_vol.ndim > 3:
            self.raw_vol = np.squeeze(self.raw_vol)
        if self.decon_vol.ndim > 3:
            self.decon_vol = np.squeeze(self.decon_vol)

        # Update slider range based on Z-depth
        z_depth = self.raw_vol.shape[0]
        self.z_slider.max = z_depth - 1
        self.z_slider.value = z_depth // 2
        self.z_slider.disabled = False

        self._update_plot()

    def _update_plot(self, *_args: Any) -> None:
        if self.raw_vol is None or self.decon_vol is None:
            return

        z = self.z_slider.value
        use_log = self.log_scale.value

        with self.out_plot:
            clear_output(wait=True)
            fig, axes = plt.subplots(2, 2, figsize=(12, 10))

            # --- Row 1: XY Views ---
            img_raw = self.raw_vol[z]
            img_decon = self.decon_vol[z]

            if use_log:
                img_raw = np.log1p(img_raw)
                img_decon = np.log1p(img_decon)

            axes[0, 0].imshow(img_raw, cmap="magma")
            axes[0, 0].set_title(f"Raw XY (Z={z})")

            axes[0, 1].imshow(img_decon, cmap="magma")
            axes[0, 1].set_title(f"Deconvolved XY (Z={z})")

            # --- Row 2: XZ Orthogonal Projections ---
            proj_raw = np.max(self.raw_vol, axis=1)
            proj_decon = np.max(self.decon_vol, axis=1)

            if use_log:
                proj_raw = np.log1p(proj_raw)
                proj_decon = np.log1p(proj_decon)

            axes[1, 0].imshow(proj_raw, cmap="magma", aspect="auto")
            axes[1, 0].set_title("Raw XZ Projection")

            axes[1, 1].imshow(proj_decon, cmap="magma", aspect="auto")
            axes[1, 1].set_title("Deconvolved XZ Projection")

            for ax in axes.flat:
                ax.axis("off")

            plt.tight_layout()
            plt.show()
