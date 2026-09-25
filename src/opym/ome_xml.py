# Ruff style: Compliant
"""OME-XML for opym's processed OME-Zarr (`ome_zarr_writer.create_processed_store`).

The store follows the bioformats2raw layout (version 3), the transitional
OME-NGFF convention for carrying full OME metadata: each image is a
numbered series group, and `OME/METADATA.ome.xml` describes all of them.
Bio-Formats (Fiji, OMERO) reads sizes, physical calibration, time step and
channel names from here; NGFF readers (napari) use each series' own
multiscales attributes, which say the same thing.
"""

from __future__ import annotations

import re
from datetime import datetime

# Excitation wavelength from an opym channel label: "GFP 488" -> 488.
_WAVELENGTH = re.compile(r"(\d{3,4})\s*$")


def _channels(n_c: int, labels: list[str], colors: tuple[str, ...], image: int):
    from ome_types import model

    out = []
    for c in range(n_c):
        label = labels[c] if c < len(labels) else f"C{c}"
        wl = _WAVELENGTH.search(label)
        extra = (
            {
                "excitation_wavelength": float(wl.group(1)),
                "excitation_wavelength_unit": "nm",
            }
            if wl
            else {}
        )
        out.append(
            model.Channel(
                id=f"Channel:{image}:{c}",
                name=label,
                samples_per_pixel=1,
                color=model.Color(f"#{colors[c % len(colors)]}"),
                **extra,
            )
        )
    return out


def processed_ome_xml(
    *,
    name: str,
    n_t: int,
    n_c: int,
    shape_zyx: tuple[int, int, int],
    channel_labels: list[str],
    voxel_um: float,
    time_interval_s: float | None,
    colors: tuple[str, ...],
    acquisition_date: datetime | None = None,
) -> str:
    """Two images: the deconvolved + deskewed volume (series "0") and its
    Z maximum projection (series "1"). Both are uint16, (T, C, Z, Y, X)."""
    from ome_types import model, to_xml

    nz, ny, nx = shape_zyx

    def image(index: int, image_name: str, size_z: int):
        pixels = model.Pixels(
            id=f"Pixels:{index}",
            dimension_order="XYZCT",
            type="uint16",
            big_endian=False,
            size_x=nx,
            size_y=ny,
            size_z=size_z,
            size_c=n_c,
            size_t=n_t,
            physical_size_x=voxel_um,
            physical_size_x_unit="µm",
            physical_size_y=voxel_um,
            physical_size_y_unit="µm",
            physical_size_z=voxel_um,
            physical_size_z_unit="µm",
            **(
                {"time_increment": time_interval_s, "time_increment_unit": "s"}
                if time_interval_s
                else {}
            ),
            channels=_channels(n_c, channel_labels, colors, index),
            metadata_only=model.MetadataOnly(),
        )
        return model.Image(
            id=f"Image:{index}",
            name=image_name,
            acquisition_date=acquisition_date,
            pixels=pixels,
        )

    ome = model.OME(
        creator="opym",
        images=[image(0, name, nz), image(1, f"{name} Z-MIP", 1)],
    )
    return to_xml(ome)
