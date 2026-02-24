from __future__ import annotations

import re
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
    Interactive widget to compare DSR (Raw) vs. Deconvolved volumes.
    Includes an alpha-blend slider to visualize the sharpening progression.
    """

    def __init__(
        self, start_path: str | Path = "/mmfs2/scratch/SDSMT.LOCAL/bscott/DataUpload"
    ) -> None:
        self.dsr_vol: np.ndarray | None = None
        self.decon_vol: np.ndarray | None = None
        self.base_dir: Path | None = None
        self._init_ui(start_path)

    def _init_ui(self, start_path: str | Path) -> None:
        self.fc_dir = FileChooser(
            path=str(start_path),
            title="<b>Select Base Directory (e.g., ..._MMStack_Pos0):</b>",
            show_only_dirs=True,
        )
        self.fc_dir.register_callback(self._on_dir_selected)

        self.ch_dropdown = widgets.Dropdown(
            description="Channel:", disabled=True, layout=widgets.Layout(width="200px")
        )
        self.ch_dropdown.observe(self._on_channel_selected, names="value")

        self.log_scale = widgets.Checkbox(value=True, description="Log Scale")

        self.z_slider = widgets.IntSlider(
            description="Z-Slice:",
            min=0,
            max=1,
            disabled=True,
            layout=widgets.Layout(width="400px"),
        )

        self.blend_slider = widgets.FloatSlider(
            description="Progression:",
            min=0.0,
            max=1.0,
            step=0.05,
            value=0.5,
            disabled=True,
            layout=widgets.Layout(width="400px"),
        )

        self.out_plot = widgets.Output()

        self.z_slider.observe(self._update_plot, names="value")
        self.blend_slider.observe(self._update_plot, names="value")
        self.log_scale.observe(self._update_plot, names="value")

        self.ui = widgets.VBox(
            [
                self.fc_dir,
                widgets.HBox([self.ch_dropdown, self.log_scale]),
                widgets.HBox([self.z_slider, self.blend_slider]),
                self.out_plot,
            ]
        )

    def show(self) -> None:
        """Display the viewer widget."""
        display(self.ui)

    def _on_dir_selected(self, chooser: FileChooser) -> None:
        if not chooser.selected:
            return
        self.base_dir = Path(chooser.selected)
        decon_dir = self.base_dir / "matlab_decon"
        dsr_dir = self.base_dir / "DSR"

        if not decon_dir.exists() or not dsr_dir.exists():
            with self.out_plot:
                clear_output(wait=True)
                print(
                    "⚠️ Could not find both 'DSR' and 'matlab_decon' "
                    f"folders in {self.base_dir.name}"
                )
            return

        # Auto-detect channels by scanning the decon folder
        tif_files = list(decon_dir.glob("*_C*_T*.tif"))
        channels: set[str] = set()
        for f in tif_files:
            match = re.search(r"_C(\d+)_", f.name)
            if match:
                channels.add(match.group(1))

        if not channels:
            with self.out_plot:
                clear_output(wait=True)
                print("⚠️ No deconvolved TIFF files found.")
            return

        sorted_channels = sorted(list(channels))

        # Safely update the dropdown options
        self.ch_dropdown.unobserve(self._on_channel_selected, names="value")
        self.ch_dropdown.options = sorted_channels
        self.ch_dropdown.value = sorted_channels[0]
        self.ch_dropdown.disabled = False
        self.ch_dropdown.observe(self._on_channel_selected, names="value")

        # Load the first channel automatically
        self._load_channel_data(sorted_channels[0])

    def _on_channel_selected(self, change: dict[str, Any]) -> None:
        if change["new"]:
            self._load_channel_data(change["new"])

    def _load_channel_data(self, ch_num: str) -> None:
        if self.base_dir is None:
            return

        dsr_dir = self.base_dir / "DSR"
        decon_dir = self.base_dir / "matlab_decon"

        dsr_file = next(dsr_dir.glob(f"*_C{ch_num}_*.tif"), None)
        decon_file = next(decon_dir.glob(f"*_C{ch_num}_*.tif"), None)

        if not dsr_file or not decon_file:
            with self.out_plot:
                clear_output(wait=True)
                print(
                    f"⚠️ Missing TIFF for Channel {ch_num} "
                    "in either DSR or matlab_decon."
                )
            return

        with self.out_plot:
            clear_output(wait=True)
            print(f"⏳ Loading Channel {ch_num}...")

        # Load and ensure they are 3D arrays
        dsr_raw = tifffile.imread(dsr_file).astype(np.float32)
        decon_raw = tifffile.imread(decon_file).astype(np.float32)

        self.dsr_vol = np.squeeze(dsr_raw) if dsr_raw.ndim > 3 else dsr_raw
        self.decon_vol = np.squeeze(decon_raw) if decon_raw.ndim > 3 else decon_raw

        # --- NEW: Center-Crop to Align Shapes ---
        if self.dsr_vol.shape != self.decon_vol.shape:
            min_z = min(self.dsr_vol.shape[0], self.decon_vol.shape[0])
            min_y = min(self.dsr_vol.shape[1], self.decon_vol.shape[1])
            min_x = min(self.dsr_vol.shape[2], self.decon_vol.shape[2])

            def crop_center(img: np.ndarray, tz: int, ty: int, tx: int) -> np.ndarray:
                z, y, x = img.shape
                sz = (z - tz) // 2
                sy = (y - ty) // 2
                sx = (x - tx) // 2
                return img[sz : sz + tz, sy : sy + ty, sx : sx + tx]

            self.dsr_vol = crop_center(self.dsr_vol, min_z, min_y, min_x)
            self.decon_vol = crop_center(self.decon_vol, min_z, min_y, min_x)

            with self.out_plot:
                clear_output(wait=True)
                print(f"ℹ️ Shapes aligned by center-cropping to: {self.dsr_vol.shape}")

        # Update controls
        z_depth = self.dsr_vol.shape[0]
        self.z_slider.max = z_depth - 1
        self.z_slider.value = z_depth // 2
        self.z_slider.disabled = False
        self.blend_slider.disabled = False

        self._update_plot()

    def _update_plot(self, *_args: Any) -> None:
        if self.dsr_vol is None or self.decon_vol is None:
            return

        z = self.z_slider.value
        alpha = self.blend_slider.value
        use_log = self.log_scale.value

        with self.out_plot:
            clear_output(wait=True)
            # Create a 3-panel layout: Raw | Blended Progression | Final
            fig, axes = plt.subplots(2, 3, figsize=(15, 8))

            raw_xy = self.dsr_vol[z]
            dec_xy = self.decon_vol[z]

            raw_xz = np.max(self.dsr_vol, axis=1)
            dec_xz = np.max(self.decon_vol, axis=1)

            # Mathematical interpolation between Raw and Deconvolved
            blend_xy = (raw_xy * (1.0 - alpha)) + (dec_xy * alpha)
            blend_xz = (raw_xz * (1.0 - alpha)) + (dec_xz * alpha)

            if use_log:
                raw_xy = np.log1p(raw_xy)
                dec_xy = np.log1p(dec_xy)
                blend_xy = np.log1p(blend_xy)

                raw_xz = np.log1p(raw_xz)
                dec_xz = np.log1p(dec_xz)
                blend_xz = np.log1p(blend_xz)

            # Row 1: XY Views
            axes[0, 0].imshow(raw_xy, cmap="magma")
            axes[0, 0].set_title(f"DSR (Raw) Z={z}")

            axes[0, 1].imshow(blend_xy, cmap="magma")
            axes[0, 1].set_title(f"Progression ({int(alpha * 100)}% Decon)")

            axes[0, 2].imshow(dec_xy, cmap="magma")
            axes[0, 2].set_title(f"Final Decon Z={z}")

            # Row 2: XZ Projections
            axes[1, 0].imshow(raw_xz, cmap="magma", aspect="auto")
            axes[1, 0].set_title("DSR XZ Projection")

            axes[1, 1].imshow(blend_xz, cmap="magma", aspect="auto")
            axes[1, 1].set_title("Progression XZ")

            axes[1, 2].imshow(dec_xz, cmap="magma", aspect="auto")
            axes[1, 2].set_title("Final Decon XZ Projection")

            for ax in axes.flat:
                ax.axis("off")

            plt.tight_layout()
            plt.show()
