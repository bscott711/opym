from __future__ import annotations

import traceback
from collections.abc import Callable
from pathlib import Path
from typing import Any

import ipywidgets as widgets
import matplotlib.patches as patches
import matplotlib.pyplot as plt
import numpy as np
import scipy.ndimage as ndi
import tifffile
from ipyfilechooser import FileChooser
from IPython.display import clear_output, display
from matplotlib.axes import Axes
from matplotlib.figure import Figure

import opym


class PSFExtractor:
    """
    Interactive widget for extracting Point Spread Functions (PSFs)
    from deskewed bead data via point-and-click with visual verification.
    """

    def __init__(
        self, start_path: str | Path = "/mmfs2/scratch/SDSMT.LOCAL/bscott/DataUpload"
    ):
        self.stack: np.ndarray | None = None
        self.get_stack_fn: Callable[[int, int], np.ndarray] | None = None

        self.t_idx: int = 0
        self.c_idx: int = 0

        self.fig: Figure | None = None
        self.ax: Axes | None = None
        self.img_obj: Any = None
        self.cid_click: Any = None

        self.dsr_path: Path | None = None
        self.done_rois: list[tuple[int, int, int, int]] = []
        self.rot: int = 0

        self._init_ui(start_path)

    def _init_ui(self, start_path: str | Path) -> None:
        self.dsr_chooser = FileChooser(
            path=str(start_path),
            title="<b>Select DSR Folder:</b>",
            show_only_dirs=True,
        )

        self.channel_slider = widgets.IntSlider(
            description="Channel:", min=0, max=0, disabled=True
        )
        self.slider_max_input = widgets.FloatText(
            value=4096.0,
            description="Max Range:",
            layout=widgets.Layout(width="150px"),
            disabled=True,
        )
        self.contrast_slider = widgets.FloatRangeSlider(
            description="Contrast:",
            min=0,
            max=4096,
            step=1,
            value=[0, 4096],
            layout=widgets.Layout(width="400px"),
            disabled=True,
        )

        self.crop_xy_input = widgets.IntText(
            value=32, description="Crop XY:", layout=widgets.Layout(width="150px")
        )
        self.crop_z_input = widgets.IntText(
            value=16, description="Crop Z:", layout=widgets.Layout(width="150px")
        )

        self.btn_load = widgets.Button(description="Load Data", button_style="primary")
        self.btn_rotate = widgets.Button(
            description="Rotate XY 90°",
            button_style="info",
            disabled=True,
            icon="repeat",
        )

        self.log_output = widgets.Output(
            layout={
                "border": "1px solid #ccc",
                "height": "150px",
                "overflow_y": "scroll",
            }
        )
        self.plot_output = widgets.Output()
        self.verify_output = widgets.Output()

        self.btn_load.on_click(self.on_load_click)
        self.btn_rotate.on_click(self.on_rotate_click)
        self.channel_slider.observe(self.on_channel_change, names="value")
        self.slider_max_input.observe(self._on_max_range_change, names="value")
        self.contrast_slider.observe(self.update_display, names="value")

        self.ui = widgets.VBox(
            [
                self.dsr_chooser,
                widgets.HBox(
                    [self.channel_slider, self.slider_max_input, self.contrast_slider]
                ),
                widgets.HBox([self.crop_xy_input, self.crop_z_input]),
                widgets.HBox([self.btn_load, self.btn_rotate]),
                self.log_output,
                widgets.HBox([self.plot_output, self.verify_output]),
            ]
        )

    def show(self) -> None:
        display(self.ui)

    @staticmethod
    def transform_point_inverse(
        sx: int, sy: int, shape: tuple[int, int], k: int
    ) -> tuple[int, int]:
        h, w = shape
        k = k % 4
        if k == 0:
            return sx, sy
        if k == 1:
            return w - 1 - sy, sx
        if k == 2:
            return w - 1 - sx, h - 1 - sy
        if k == 3:
            return sy, h - 1 - sx
        return sx, sy

    @staticmethod
    def transform_point_forward(
        dx: int, dy: int, shape: tuple[int, int], k: int
    ) -> tuple[int, int]:
        h, w = shape
        k = k % 4
        if k == 0:
            return dx, dy
        if k == 1:
            return dy, w - 1 - dx
        if k == 2:
            return w - 1 - dx, h - 1 - dy
        if k == 3:
            return h - 1 - dy, dx
        return dx, dy

    def update_display(self, _change: Any = None) -> None:
        if self.stack is None or self.img_obj is None or self.fig is None:
            return
        vmin, vmax = self.contrast_slider.value
        self.img_obj.set_clim(vmin, vmax)
        self.fig.canvas.draw_idle()

    def _on_max_range_change(self, change: Any) -> None:
        self.contrast_slider.max = change["new"]

    def on_rotate_click(self, _b: widgets.Button) -> None:
        self.rot = (self.rot + 1) % 4
        self.render_xy_view()

    def on_channel_change(self, change: Any) -> None:
        self.c_idx = change["new"]
        if self.get_stack_fn is None:
            return
        with self.log_output:
            print(f"Loading Channel {self.c_idx}...")
            try:
                self.stack = self.get_stack_fn(self.t_idx, self.c_idx)
                self.render_xy_view()
            except Exception as e:
                print(f"Error: {e}")

    def on_load_click(self, _b: widgets.Button) -> None:
        with self.log_output:
            clear_output()
            if not self.dsr_chooser.selected:
                print("⚠️ Please select a folder first.")
                return
            path = Path(self.dsr_chooser.selected)
            self.dsr_path = path
            self.done_rois = []
            self.rot = 0
            print(f"Reading {path.name}...")
            try:
                (get_stack, t_min, _, c_min, c_max, *_) = opym.load_tiff_series(path)
                self.get_stack_fn = get_stack
                self.t_idx = t_min
                self.c_idx = c_min
                self.stack = get_stack(t_min, c_min)

                self.channel_slider.min, self.channel_slider.max = c_min, c_max
                self.channel_slider.value = c_min
                self.channel_slider.disabled = False

                if self.stack is not None:
                    d_max = np.max(self.stack)
                    if d_max <= 255:
                        lim = 255.0
                    elif d_max <= 4095:
                        lim = 4095.0
                    else:
                        lim = 65535.0

                    self.slider_max_input.value = lim
                    self.slider_max_input.disabled = False
                    self.contrast_slider.max = lim
                    p1, p99 = np.percentile(self.stack, [1, 99.9])
                    self.contrast_slider.value = (float(p1), min(float(p99), lim))
                    self.contrast_slider.disabled = False

                self.btn_rotate.disabled = False
                self.render_xy_view()
            except Exception as e:
                print(f"Load Error: {e}")
                traceback.print_exc()

    def _isolate_central_bead(self, vol: np.ndarray) -> np.ndarray:
        bg_val = np.median(vol)
        # Use a high percentile to only threshold the bright cores, avoiding tails
        thresh = np.percentile(vol, 95)
        mask = vol > thresh
        labeled_mask, num_labels = ndi.label(mask)  # type: ignore

        if num_labels <= 1:
            return vol

        center = np.array(vol.shape) / 2.0
        best_label, min_dist = 1, float("inf")

        for i in range(1, num_labels + 1):
            com = ndi.center_of_mass(vol, labeled_mask, i)
            # com is now guaranteed to be evaluated as a tuple by Pylance
            if isinstance(com, tuple):
                dist = sum((c - m) ** 2 for c, m in zip(center, com))
                if dist < min_dist:
                    min_dist = dist
                    best_label = i

        # Identify all unwanted cores and dilate them to capture their faint rings
        outlier_mask = (labeled_mask > 0) & (labeled_mask != best_label)
        outlier_mask = ndi.binary_dilation(outlier_mask, iterations=3)

        return np.where(outlier_mask, bg_val, vol).astype(vol.dtype)

    def show_verification(self, raw_vol: np.ndarray, final_vol: np.ndarray) -> None:
        with self.verify_output:
            clear_output(wait=True)
            fig, axes = plt.subplots(2, 3, figsize=(7, 5))

            axes[0, 0].imshow(np.max(raw_vol, axis=0), cmap="gray")
            axes[0, 0].set_title("Raw XY")
            axes[0, 1].imshow(np.max(raw_vol, axis=1), cmap="gray")
            axes[0, 1].set_title("Raw XZ")
            axes[0, 2].imshow(np.max(raw_vol, axis=2), cmap="gray")
            axes[0, 2].set_title("Raw YZ")

            axes[1, 0].imshow(np.max(final_vol, axis=0), cmap="gray")
            axes[1, 0].set_title("Isolated XY")
            axes[1, 1].imshow(np.max(final_vol, axis=1), cmap="gray")
            axes[1, 1].set_title("Isolated XZ")
            axes[1, 2].imshow(np.max(final_vol, axis=2), cmap="gray")
            axes[1, 2].set_title("Isolated YZ")

            for ax in axes.flat:
                ax.axis("off")

            plt.tight_layout()
            plt.show()

    def on_click_xy(self, event: Any) -> None:
        if self.stack is None or self.dsr_path is None or event.inaxes != self.ax:
            return

        disp_x, disp_y = int(event.xdata), int(event.ydata)
        h, w = self.stack.shape[1], self.stack.shape[2]

        data_x, data_y = self.transform_point_inverse(disp_x, disp_y, (h, w), self.rot)

        xy_half = self.crop_xy_input.value // 2
        z_half = self.crop_z_input.value // 2

        # Enforce strict symmetry. Reject clicks that force an asymmetrical crop.
        if (
            data_x - xy_half < 0
            or data_x + xy_half >= w
            or data_y - xy_half < 0
            or data_y + xy_half >= h
        ):
            with self.log_output:
                print("⚠️ Bead too close to XY edge. Skipped to maintain symmetry.")
            return

        z_profile = self.stack[:, data_y, data_x]
        z_center = int(np.argmax(z_profile))

        if z_center - z_half < 0 or z_center + z_half >= self.stack.shape[0]:
            with self.log_output:
                print("⚠️ Bead too close to Z edge. Skipped to maintain symmetry.")
            return

        x1, x2 = data_x - xy_half, data_x + xy_half
        y1, y2 = data_y - xy_half, data_y + xy_half
        z1, z2 = z_center - z_half, z_center + z_half

        raw_vol = self.stack[z1:z2, y1:y2, x1:x2]
        if raw_vol.size == 0:
            return

        final_vol = self._isolate_central_bead(raw_vol)
        self.show_verification(raw_vol, final_vol)

        ctr = 1
        while True:
            fname = f"PSF_C{self.c_idx}_Bead_{ctr:03d}.tif"
            save_path = self.dsr_path / fname
            if not save_path.exists():
                break
            ctr += 1

        tifffile.imwrite(save_path, final_vol)

        with self.log_output:
            print(f"✅ Saved #{ctr}: {fname} (Center Z={z_center})")

        self.done_rois.append((x1, x2, y1, y2))

        sx1, sy1 = self.transform_point_forward(x1, y1, (h, w), self.rot)
        sx2, sy2 = self.transform_point_forward(x2, y2, (h, w), self.rot)
        if self.ax is not None and self.fig is not None:
            self.ax.add_patch(
                patches.Rectangle(
                    (min(sx1, sx2), min(sy1, sy2)),
                    abs(sx2 - sx1),
                    abs(sy2 - sy1),
                    linewidth=1,
                    edgecolor="red",
                    facecolor="none",
                    linestyle="-",
                )
            )
            self.fig.canvas.draw_idle()

    def render_xy_view(self) -> None:
        if self.stack is None:
            return

        with self.plot_output:
            clear_output(wait=True)
            mip_xy = np.max(self.stack, axis=0)
            mip_disp = np.rot90(mip_xy, k=self.rot)

            self.fig, self.ax = plt.subplots(figsize=(6, 6))
            vmin, vmax = self.contrast_slider.value
            self.img_obj = self.ax.imshow(mip_disp, cmap="gray", vmin=vmin, vmax=vmax)
            self.ax.set_title("Click on a bead to auto-extract!")

            h_data, w_data = mip_xy.shape
            for dx1, dx2, dy1, dy2 in self.done_rois:
                sx1, sy1 = self.transform_point_forward(
                    dx1, dy1, (h_data, w_data), self.rot
                )
                sx2, sy2 = self.transform_point_forward(
                    dx2, dy2, (h_data, w_data), self.rot
                )
                self.ax.add_patch(
                    patches.Rectangle(
                        (min(sx1, sx2), min(sy1, sy2)),
                        abs(sx2 - sx1),
                        abs(sy2 - sy1),
                        linewidth=1,
                        edgecolor="red",
                        facecolor="none",
                        linestyle="-",
                    )
                )

            if self.cid_click is not None:
                self.fig.canvas.mpl_disconnect(self.cid_click)
            self.cid_click = self.fig.canvas.mpl_connect(
                "button_press_event", self.on_click_xy
            )
            plt.show()
