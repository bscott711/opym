from __future__ import annotations

from pathlib import Path
from typing import Any

import ipywidgets as widgets
import matplotlib.pyplot as plt
import numpy as np
import scipy.ndimage as ndi
import scipy.optimize as optimize
import tifffile
from ipyfilechooser import FileChooser
from IPython.display import clear_output, display
from scipy.signal.windows import tukey
from skimage.filters import threshold_otsu


class PSFAverager:
    """
    Interactive widget for generating a Master PSF by aligning and averaging
    individual bead ROIs.
    """

    def __init__(
        self, start_path: str | Path = "/mmfs2/scratch/SDSMT.LOCAL/bscott/DataUpload"
    ):
        self.px_xy: float = 0.136
        self.px_z: float = 0.100
        self._init_ui(start_path)

    def _init_ui(self, start_path: str | Path) -> None:
        self.folder_chooser = FileChooser(
            path=str(start_path),
            title="<b>Select Folder containing KEPT Bead ROIs:</b>",
            show_only_dirs=True,
        )

        self.strict_mask_check = widgets.Checkbox(
            value=True, description="Hard Crop (Zero outside mask)"
        )
        self.taper_slider = widgets.FloatSlider(
            value=0.1, min=0.0, max=0.5, step=0.05, description="Edge Taper:"
        )

        self.btn_run = widgets.Button(
            description="Generate Master PSF",
            button_style="success",
            layout=widgets.Layout(width="200px"),
        )
        self.out_log = widgets.Output(
            layout={
                "border": "1px solid #ccc",
                "height": "200px",
                "overflow_y": "scroll",
            }
        )
        self.out_plot = widgets.Output()

        self.btn_run.on_click(self.run_average)

        self.ui = widgets.VBox(
            [
                self.folder_chooser,
                widgets.HBox([self.strict_mask_check, self.taper_slider]),
                self.btn_run,
                self.out_log,
                self.out_plot,
            ]
        )

    def show(self) -> None:
        """Display the widget."""
        display(self.ui)

    @staticmethod
    def fit_gaussian_1d(x: np.ndarray, y: np.ndarray) -> float:
        def gauss(x_val: Any, a: float, x0: float, sigma: float, offset: float) -> Any:
            return a * np.exp(-((x_val - x0) ** 2) / (2 * sigma**2)) + offset

        try:
            p0 = [np.max(y), len(y) / 2.0, len(y) / 4.0, 0.0]
            popt, _ = optimize.curve_fit(gauss, x, y, p0=p0)
            return float(2.355 * abs(popt[2]))
        except Exception:
            return 0.0

    @staticmethod
    def get_fallback_mask(vol: np.ndarray) -> np.ndarray:
        smooth = ndi.gaussian_filter(vol, sigma=1.0)
        try:
            thresh = threshold_otsu(smooth)
        except Exception:
            thresh = np.percentile(smooth, 90)
        mask = smooth > thresh
        return ndi.binary_dilation(mask, iterations=2)

    @staticmethod
    def taper_vol(vol: np.ndarray, fraction: float = 0.1) -> np.ndarray:
        d, h, w = vol.shape
        win_z = tukey(d, alpha=fraction)
        win_y = tukey(h, alpha=fraction)
        win_x = tukey(w, alpha=fraction)
        return vol * (
            win_z[:, None, None] * win_y[None, :, None] * win_x[None, None, :]
        )

    def run_average(self, _b: widgets.Button) -> None:
        with self.out_log:
            clear_output()
            if not self.folder_chooser.selected:
                print("⚠️ Please select a folder first.")
                return
            p = Path(self.folder_chooser.selected)

            all_tifs = sorted(list(p.glob("*.tif")) + list(p.glob("*.tiff")))
            # Update this list comprehension to explicitly require "Bead" in the name
            bead_files = [
                f
                for f in all_tifs
                if "Bead" in f.name
                and "Master" not in f.name
                and "Avg" not in f.name
                and "_mask" not in f.name
            ]

            if not bead_files:
                print("❌ No beads found.")
                return

            print(f"--- Averaging {len(bead_files)} Beads ---")

            # Determine Canvas dimensions across all loaded files
            shapes = [tifffile.imread(f).shape for f in bead_files]
            mz = max((s[0] if len(s) == 3 else 1) for s in shapes)
            my = max((s[1] if len(s) == 3 else s[0]) for s in shapes)
            mx = max((s[2] if len(s) == 3 else s[1]) for s in shapes)
            mz += 1 if mz % 2 == 0 else 0
            my += 1 if my % 2 == 0 else 0
            mx += 1 if mx % 2 == 0 else 0
            target_shape = (mz, my, mx)
            center_target = np.array(target_shape) / 2.0

            aligned_sum = np.zeros(target_shape, dtype=np.float32)
            used_count = 0

            for f in bead_files:
                vol = tifffile.imread(f).astype(np.float32)
                if vol.ndim > 3:
                    vol = np.squeeze(vol)

                # 1. LOAD MASK
                mask_path = f.parent / f"{f.stem}_mask.tif"
                if mask_path.exists():
                    mask = tifffile.imread(mask_path) > 0
                    mask_source = "USER"
                else:
                    mask = self.get_fallback_mask(vol)
                    mask_source = "AUTO"

                # 2. Subtract BG (Use mean so noise fluctuates evenly around 0)
                bg_pixels = vol[~mask]
                bg_val = float(np.mean(bg_pixels)) if len(bg_pixels) > 0 else 0.0
                vol_clean = vol - bg_val

                # Create a strictly positive copy just for Peak/COM finding
                vol_pos = np.clip(vol_clean, 0, None)

                # 3. STRICT MASKING (The Cookie Cutter)
                if self.strict_mask_check.value:
                    vol_clean[~mask] = 0.0
                    vol_pos[~mask] = 0.0

                # 4. Find Alignment Point (Peak-Weighted Center of Mass)
                if np.sum(vol_pos) == 0:
                    print(f"⚠️ Skipped {f.name} (Empty after cleaning)")
                    continue

                peak_weighted = vol_pos**4
                if np.sum(peak_weighted) == 0:
                    continue

                com_tuple = ndi.center_of_mass(peak_weighted)
                if not isinstance(com_tuple, tuple):
                    continue
                com = np.array(com_tuple)

                # Soften the edges of individual crops to hide shifting seams
                if self.taper_slider.value > 0:
                    vol_clean = self.taper_vol(vol_clean, self.taper_slider.value)

                # 5. Embed & Shift
                canvas = np.zeros(target_shape, dtype=np.float32)
                z_off = (mz - vol.shape[0]) // 2
                y_off = (my - vol.shape[1]) // 2
                x_off = (mx - vol.shape[2]) // 2

                z_end, y_end = z_off + vol.shape[0], y_off + vol.shape[1]
                x_end = x_off + vol.shape[2]
                canvas[z_off:z_end, y_off:y_end, x_off:x_end] = vol_clean

                com_canvas = com + np.array([z_off, y_off, x_off])
                shift_vec = center_target - com_canvas

                # Shift the canvas (preserving negative noise fluctuations)
                aligned = ndi.shift(canvas, shift_vec, order=3, prefilter=True)
                aligned_sum += aligned
                used_count += 1

                print(f"[{mask_source}] Merged: {f.name}")

            if used_count == 0:
                return

            # Final Average
            master_psf = aligned_sum / used_count

            # NOW we clip the perfectly averaged noise floor to 0
            master_psf[master_psf < 0] = 0

            # Taper Final Volume
            if self.taper_slider.value > 0:
                master_psf = self.taper_vol(
                    master_psf, fraction=self.taper_slider.value
                )

            master_psf /= np.max(master_psf)
            save_path = p / "Master_PSF_Final_Strict.tif"
            tifffile.imwrite(save_path, master_psf.astype(np.float32))
            print(f"\n✅ SAVED: {save_path.name}")

            # --- Visualization ---
            self._render_visualization(master_psf, center_target)

    def _render_visualization(self, psf: np.ndarray, center: np.ndarray) -> None:
        with self.out_plot:
            clear_output(wait=True)

            cz, cy, cx = list(map(int, center))
            prof_z = psf[:, cy, cx]
            prof_x = psf[cz, cy, :]

            fz = self.fit_gaussian_1d(np.arange(len(prof_z)), prof_z) * self.px_z
            fx = self.fit_gaussian_1d(np.arange(len(prof_x)), prof_x) * self.px_xy

            fig = plt.figure(figsize=(10, 6))
            gs = fig.add_gridspec(2, 3)

            ax0 = fig.add_subplot(gs[0, 0])
            ax0.imshow(np.log1p(psf[:, cy, :]), cmap="magma", aspect="auto")
            ax0.set_title(f"XZ View (Log Scale)\nFWHM Z: {fz:.3f} µm")

            ax1 = fig.add_subplot(gs[0, 1])
            ax1.imshow(np.log1p(psf[cz, :, :]), cmap="magma")
            ax1.set_title(f"XY View (Log Scale)\nFWHM X: {fx:.3f} µm")

            ax3 = fig.add_subplot(gs[1, :])
            ax3.plot(prof_z, label="Z Profile", color="blue")
            ax3.plot(prof_x, label="X Profile", color="orange", linestyle="--")
            ax3.set_title("Intensity Profiles")
            ax3.legend()
            ax3.grid(True, alpha=0.3)

            plt.tight_layout()
            plt.show()
