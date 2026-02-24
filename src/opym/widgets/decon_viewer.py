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


def _calc_fwhm(profile: np.ndarray) -> float:
    """Calculate FWHM of a 1D profile in pixels."""
    if profile.size == 0:
        return 0.0
    bg = np.min(profile)
    peak = np.max(profile)
    half_max = bg + (peak - bg) / 2.0

    above_half = np.where(profile >= half_max)[0]
    if len(above_half) > 1:
        return float(above_half[-1] - above_half[0])
    return 0.0


class DeconvolutionViewer:
    """
    Interactive widget to compare DSR (Raw) vs. Deconvolved volumes.
    Includes an alpha-blend slider and live 1D Line Profiles for FWHM.
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

        self.ch_dropdown.unobserve(self._on_channel_selected, names="value")
        self.ch_dropdown.options = sorted_channels
        self.ch_dropdown.value = sorted_channels[0]
        self.ch_dropdown.disabled = False
        self.ch_dropdown.observe(self._on_channel_selected, names="value")

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

        dsr_raw = tifffile.imread(dsr_file).astype(np.float32)
        decon_raw = tifffile.imread(decon_file).astype(np.float32)

        self.dsr_vol = np.squeeze(dsr_raw) if dsr_raw.ndim > 3 else dsr_raw
        self.decon_vol = np.squeeze(decon_raw) if decon_raw.ndim > 3 else decon_raw

        # Center-crop the DSR volume if PetaKit trimmed the boundaries
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
            fig, axes = plt.subplots(3, 3, figsize=(15, 12))

            lin_raw_xy = self.dsr_vol[z]
            lin_dec_xy = self.decon_vol[z]

            # Anchor profiles to the brightest point in the decon slice
            my, mx = np.unravel_index(np.argmax(lin_dec_xy), lin_dec_xy.shape)

            # Calculate FWHMs strictly on linear data
            fwhm_raw_x = _calc_fwhm(lin_raw_xy[my, :])
            fwhm_dec_x = _calc_fwhm(lin_dec_xy[my, :])

            fwhm_raw_y = _calc_fwhm(lin_raw_xy[:, mx])
            fwhm_dec_y = _calc_fwhm(lin_dec_xy[:, mx])

            fwhm_raw_z = _calc_fwhm(self.dsr_vol[:, my, mx])
            fwhm_dec_z = _calc_fwhm(self.decon_vol[:, my, mx])

            # Prepare plotting arrays
            raw_xy = np.copy(lin_raw_xy)
            dec_xy = np.copy(lin_dec_xy)
            raw_xz = np.max(self.dsr_vol, axis=1)
            dec_xz = np.max(self.decon_vol, axis=1)

            blend_xy = (raw_xy * (1.0 - alpha)) + (dec_xy * alpha)
            blend_xz = (raw_xz * (1.0 - alpha)) + (dec_xz * alpha)

            if use_log:
                raw_xy, dec_xy, blend_xy = (
                    np.log1p(raw_xy),
                    np.log1p(dec_xy),
                    np.log1p(blend_xy),
                )
                raw_xz, dec_xz, blend_xz = (
                    np.log1p(raw_xz),
                    np.log1p(dec_xz),
                    np.log1p(blend_xz),
                )

            # Row 1: XY Views
            axes[0, 0].imshow(raw_xy, cmap="magma")
            axes[0, 0].set_title(f"DSR (Raw) Z={z}")

            axes[0, 1].imshow(blend_xy, cmap="magma")
            axes[0, 1].set_title(f"Progression ({int(alpha * 100)}% Decon)")

            axes[0, 2].imshow(dec_xy, cmap="magma")
            axes[0, 2].set_title("Final Decon")

            # Add crosshairs to XY images to show profile origin
            for ax in [axes[0, 0], axes[0, 1], axes[0, 2]]:
                ax.axhline(my, color="white", alpha=0.3, ls="--")
                ax.axvline(mx, color="white", alpha=0.3, ls="--")
                ax.axis("off")

            # Row 2: XZ Projections
            axes[1, 0].imshow(raw_xz, cmap="magma", aspect="auto")
            axes[1, 0].set_title("DSR XZ Projection")

            axes[1, 1].imshow(blend_xz, cmap="magma", aspect="auto")
            axes[1, 1].set_title("Progression XZ")

            axes[1, 2].imshow(dec_xz, cmap="magma", aspect="auto")
            axes[1, 2].set_title("Final Decon XZ Projection")

            for ax in [axes[1, 0], axes[1, 1], axes[1, 2]]:
                ax.axvline(mx, color="white", alpha=0.3, ls="--")
                ax.axis("off")

            # Row 3: 1D Line Profiles (using the plot-transformed data)
            plot_raw_x, plot_dec_x = raw_xy[my, :], dec_xy[my, :]
            plot_raw_y, plot_dec_y = raw_xy[:, mx], dec_xy[:, mx]

            plot_raw_z = self.dsr_vol[:, my, mx]
            plot_dec_z = self.decon_vol[:, my, mx]
            if use_log:
                plot_raw_z, plot_dec_z = np.log1p(plot_raw_z), np.log1p(plot_dec_z)

            # X-Profile
            axes[2, 0].plot(plot_raw_x, label=f"Raw ({fwhm_raw_x:.1f}px)", color="C0")
            axes[2, 0].plot(plot_dec_x, label=f"Decon ({fwhm_dec_x:.1f}px)", color="C1")
            axes[2, 0].plot(blend_xy[my, :], color="gray", alpha=0.5, ls=":")
            axes[2, 0].set_title("X Profile")
            axes[2, 0].legend(fontsize="small")

            # Y-Profile
            axes[2, 1].plot(plot_raw_y, label=f"Raw ({fwhm_raw_y:.1f}px)", color="C0")
            axes[2, 1].plot(plot_dec_y, label=f"Decon ({fwhm_dec_y:.1f}px)", color="C1")
            axes[2, 1].plot(blend_xy[:, mx], color="gray", alpha=0.5, ls=":")
            axes[2, 1].set_title("Y Profile")
            axes[2, 1].legend(fontsize="small")

            # Z-Profile
            blend_z = (plot_raw_z * (1.0 - alpha)) + (plot_dec_z * alpha)
            axes[2, 2].plot(plot_raw_z, label=f"Raw ({fwhm_raw_z:.1f}px)", color="C0")
            axes[2, 2].plot(plot_dec_z, label=f"Decon ({fwhm_dec_z:.1f}px)", color="C1")
            axes[2, 2].plot(blend_z, color="gray", alpha=0.5, ls=":")
            axes[2, 2].set_title("Z Profile")
            axes[2, 2].legend(fontsize="small")

            plt.tight_layout()
            plt.show()
