# Ruff style: Compliant
# File: deskewer.py
# Description:
# This module contains a standalone class for performing offline deskewing
# of light-sheet microscopy data, producing a 2D maximum projection.

import math

import numpy as np
import torch


class OfflineDeskewer:
    """
    A class to deskew 3D light-sheet image stacks.

    This class extracts the core geometric transformation logic from the
    original real-time library. It performs a shear transformation followed
    by a maximum intensity projection to correct for the angled imaging
    geometry of an Oblique Plane Microscope (OPM) or Lattice Light-Sheet
    Microscope (LLSM).
    """

    def __init__(
        self,
        sheet_angle_deg: float,
        xy_pixel_pitch_um: float,
        z_step_um: float,
    ):
        """
        Initializes the deskewer with microscope-specific parameters.
        """
        self.sheet_angle_rad = np.deg2rad(sheet_angle_deg)
        self.xy_pixel_pitch_um = xy_pixel_pitch_um
        self.z_step_um = z_step_um

        self.shear_factor = np.cos(self.sheet_angle_rad) * (
            self.z_step_um / self.xy_pixel_pitch_um
        )

        if torch.cuda.is_available():
            self.device = torch.device("cuda:0")
            print("  - Deskewer: Using GPU (CUDA) for processing.")
        else:
            self.device = torch.device("cpu")
            print("  - Deskewer: No GPU found. Using CPU for processing.")

    def deskew_stack(self, raw_stack: np.ndarray) -> np.ndarray:
        """
        Processes a raw 3D image stack to create a 2D deskewed projection.
        """
        if raw_stack.ndim != 3:
            raise ValueError("Input `raw_stack` must be a 3D array.")

        depth, height, width = raw_stack.shape

        new_height = math.ceil(height + self.shear_factor * (depth - 1))

        final_image = torch.zeros(
            (new_height, width), device=self.device, dtype=torch.float32
        )

        for i, slice_2d in enumerate(raw_stack):
            slice_tensor = torch.from_numpy(slice_2d.astype(np.float32)).to(self.device)

            start_h = math.floor(self.shear_factor * i)
            stop_h = start_h + height

            target_region = final_image[start_h:stop_h, :]
            projected_region = torch.maximum(target_region, slice_tensor)
            final_image[start_h:stop_h, :] = projected_region

        return final_image.cpu().numpy()
