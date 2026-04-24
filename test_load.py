from pathlib import Path

import tifffile

data_dir = Path(
    "/mmfs2/scratch/SDSMT.LOCAL/bscott/DataUpload/20260304_YGbeads_30PDMS_AH/dhDF_2"
)
files = sorted(data_dir.glob("*.ome.tif"))
print(f"Found {len(files)} ome.tif files")
for f in files:
    try:
        with tifffile.TiffFile(f) as tif:
            print(f"{f.name}:")
            print(f"  is_ome: {tif.is_ome}")
            print(f"  is_micromanager: {tif.is_micromanager}")
            series = tif.series[0]
            print(f"  series shape: {series.shape}")
            print(f"  series axes: {series.axes}")
        break  # Just check the first one
    except Exception as e:
        print(f"Error reading {f.name}: {e}")
