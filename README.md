# OPM Processing Package (`opym`)

`opym` is a Python package for processing dual-camera OPM (Oblique Plane
Microscopy) data. It crops and splits the two camera views, submits deskew /
rotate and (optionally) deconvolution jobs to PetaKit5D running on GPUs, and can
receive an acquisition **as it is being recorded**, process it one timepoint at a
time, and show it live in napari.

It has three parts:

1. **Interactive notebook and CLI:** pick ROIs on a maximum-intensity projection,
   then crop and split one file or a whole directory (`opym`).
2. **GPU job pipeline:** a supervised pool of MATLAB PetaKit5D servers, one per
   GPU, fed from a ticket queue (`opym-serve`). Live work goes ahead of batch work.
3. **Streaming and the live lane:** a receiver for frames streamed from the
   acquisition PC, per-timepoint decon + deskew/rotate, a processed OME-Zarr
   store, and a live viewer (`opym-receive`, `naparym-live`).

Release notes are in [CHANGELOG.md](CHANGELOG.md).

## Installation

This package is managed with `uv` and `pyproject.toml`. It needs Python 3.10 or
newer.

1. Clone the repository:

    ```bash
    git clone https://github.com/bscott711/opym.git
    cd opym
    ```

2. Create and activate a virtual environment:

    ```bash
    python -m venv .venv
    source .venv/bin/activate
    ```

3. Install the package in editable mode:

    ```bash
    uv pip install -e ".[viewer]"
    ```

The `viewer` extra adds `ipywidgets`, `matplotlib`, `ipyfilechooser` and
`ipython`. `opym` imports `ipywidgets` when it loads, so install it even if you
only use the command line. Use `".[viewer,dev]"` to also get the test and lint
tools.

The GPU pipeline additionally needs MATLAB with the PetaKit5D source on a machine
with NVIDIA GPUs (the supervisor starts MATLAB with
`module load matlab/R2024b`), and `naparym-live` needs napari with a Qt backend;
neither is installed by this package.

## Command-line tools

| Command | Entry point | What it does |
|---|---|---|
| `opym` | `opym.cli:main` | Crop and split OME-TIFFs (see Workflow 2). |
| `opym-receive` | `opym.stream.receiver:main` | Receive a frame stream from the acquisition PC. |
| `opym-serve` | `opym.local_gpu_worker:main` | Supervise the PetaKit5D GPU servers. |
| `naparym-live` | `opym.live_view:main` | Follow a live acquisition in napari. |
| `opym-live-trace` | `opym.stream.trace_report:main` | Where each timepoint spent its time, per hop. |
| `opym-stream-watch` | `opym.stream.client:main` | Stream a local zarr store (acquisition-PC side). |

Only `opym` is declared in this package's `pyproject.toml`. The other commands
are declared by the companion `bioimaging` project, which installs `opym` as a
dependency and also provides the bulk backfill driver. The companion also
installs its own command named `opym`, so where both are installed, run this
package's CLI as `python -m opym.cli`. Every command above can likewise be run
from a plain checkout as `python -m <module>`, for example
`python -m opym.stream.receiver`.

## Workflow 1: Interactive ROI Selection (Jupyter)

This is the recommended starting point for any new dataset. Use the included
notebook to find the correct cropping coordinates.

1. Start Jupyter Lab:

    ```bash
    jupyter lab
    ```

2. Open the `OPM_Cropping_Refactored.ipynb` notebook (in `Notebooks/`).

3. **Cell 0 & 1:** Select your base OME-TIF file (e.g., `..._Pos0.ome.tif`) and run
   the cells to generate a Max Intensity Projection (MIP).

4. **Cell 2:** Run to open the interactive ROI selector.
    * Draw a box for the **Top ROI (C=0)**.
    * Click-and-drag near the center of the **Bottom ROI (C=1)**. The box size
      will be matched automatically.

5. **Cell 3 & 4:** Run to trigger the auto-alignment, which fine-tunes the Bottom
   ROI position based on phase cross-correlation. The final aligned ROIs will be
   displayed.

6. **Cell 5:** Saves your selected ROIs to the central `opm_roi_log.json` file in
   your project directory. This file is used by the CLI for batch processing.

7. **Cell 6:** Submits the cropping job for this file to the GPU job queue and
   saves a settings sidecar (`petakit_settings.json`) in the output folder. This
   needs `opym-serve` running.

8. **Cell 7:** Submits the deskew and (optionally) deconvolution job.

9. **Cells 8-10:** Load the result and open the single-channel or composite
   viewer.

10. **Cell 11:** Batch-processes other files using the sidecar from an earlier run
    as the "gold standard" settings template.

## Workflow 2: CLI Batch Processing

Once you have saved your ROIs to the `opm_roi_log.json` file, you can use the
`opym` CLI to process all other files in your dataset (e.g., `..._Pos1.ome.tif`,
`..._Pos2.ome.tif`, etc.).

### Examples

**Process all files using the log:**

This is the most common use case. The command finds every `*.ome.tif` file in the
input directory and takes its ROIs from the matching entry (by file name) in
`opm_roi_log.json`. Files with no entry are skipped.

```bash
opym \
    --input-dir /path/to/my/dataset \
    --format ZARR \
    --roi-from-log opm_roi_log.json
```

**Process a single file with explicit ROIs:**

```bash
opym \
    --input-file /path/to/my/dataset/my_file_Pos0.ome.tif \
    --format TIFF_SERIES_SPLIT_C \
    --top-roi "431:708,557:1671" \
    --bottom-roi "1582:1859,531:1645"
```

CLI options:

```text
opym [OPTIONS] [input_pos]

  input_pos                     Positional input file (alternative to --input-file).
  --input-file PATH             A single 5D OME-TIF file to process.
  --input-dir PATH              A directory of 5D OME-TIF files (requires --roi-from-log).
  --top-roi TEXT                Top ROI as "y_start:y_stop,x_start:x_stop".
  --bottom-roi TEXT             Bottom ROI as "y_start:y_stop,x_start:x_stop".
  --roi-from-log PATH           JSON log file containing ROIs (e.g. opm_roi_log.json).
  -f, --format [ZARR|TIFF_SERIES_SPLIT_C]
                                Output format. TIFF_SERIES_SPLIT_C is required for
                                pypetakit5d. [default: TIFF_SERIES_SPLIT_C]
  --rotate                      Rotate the cropped ROIs 90 degrees counter-clockwise.
  -c, --channels CH [CH ...]    Output channels to save. [default: 0 1 2 3]
  --debug                       Enable deep profiling in MATLAB workers.
  -h, --help                    Show this message and exit.
```

Provide one input (`--input-file`/positional file, or `--input-dir`) and ROIs from
either `--roi-from-log` or `--top-roi`/`--bottom-roi`.

## Streaming an acquisition and the live lane

`opym.stream` bridges an acquisition PC to the GPUs. The acquisition-side client
sends each channel's volume over ZeroMQ, in slabs, while the volume is still being
acquired. Volumes are compressed with blosc on the wire, and one session can use
several connections in parallel.

```text
acquisition PC ──ZeroMQ──> opym-receive ──> raw OME-Zarr (one store per channel)
                               │
                               └─ live ticket per (timepoint, channel)
                                        │
                        opym-serve: one MATLAB PetaKit5D server per GPU
                        (live queue first, backfill queue second)
                                        │  decon -> deskew/rotate, in memory
                                        v
                      processed OME-Zarr  ──>  naparym-live
```

What the receiver does:

* Writes each channel into its own raw store, `<base_name>_<ChannelName>_<Wavelength>.ome.zarr`
  (zarr v2, one z-plane per chunk), under the `raw_root` the client declares. The
  bulk backfill then discovers a finished streamed dataset exactly like one that
  arrived by file transfer.
* Acknowledges frames only once they are durably written, so a client that loses
  its connection can reconnect and resume. It also survives a receiver restart
  mid-run.
* With `OPYM_LIVE_LANE=1` and a PSF set, hands every completed (timepoint, channel)
  volume to the **live lane**, which queues a ticket for the GPU servers ahead of
  any backfill work. While an acquisition is open the receiver holds a lease that
  keeps the backfill off the GPUs, and a backfill server is preempted if live work
  has none. With the one-format lane, a `PREPARE` message sent while the acquisition
  is being set up warms a GPU server in advance.
* Keeps the microscope's session name. If an earlier run in the same folder used
  it, that run is moved (not deleted) to `.superseded/` in that folder, which the
  backfill ignores. Only while that earlier run may still be in use does the new
  one get a `_001` suffix instead.
* Can stage everything on a RAM disk (`OPYM_STREAM_STAGE_ROOT`) and copy each
  finished session to its final location in the background, verified byte for
  byte. It keeps a free-space floor, evicts drained sessions early, and asks the
  client to resend anything it had to hold back.

The wire protocol, for anyone writing a client, is in
[docs/STREAMING_PROTOCOL.md](docs/STREAMING_PROTOCOL.md). The live path from
camera to screen, with measured timings, is in
[docs/live-view-pipeline.md](docs/live-view-pipeline.md). On the test stack a
timepoint is on screen about 0.6 s after its last plane is acquired.

A minimal live setup on the processing machine:

```bash
export OPYM_DECON_PSF=/path/to/psf.tif        # decon on; the live lane needs it
export OPYM_LIVE_LANE=1                       # hand streamed volumes to the live lane
export OPYM_LIVE_FORMAT=zarr                  # the one-format lane (see below)
export OPYM_STREAM_STAGE_ROOT=/dev/shm/opym_stream_stage   # optional RAM-disk staging

opym-serve &        # GPU servers
opym-receive        # binds tcp://127.0.0.1:5555
naparym-live        # opens at once and follows the newest session
```

The receiver binds to loopback by default; the acquisition PC reaches it through
an SSH port forward. An optional direct endpoint
(`OPYM_STREAM_DIRECT_BIND`) is refused unless both `OPYM_STREAM_ALLOW_IPS` and
`OPYM_STREAM_RAW_ROOTS` are set.

`opym-stream-watch LEAF_DIR --raw-root DIR [--connect ADDR] [--poll-interval S]`
is a ready-made client for the acquisition PC. It polls a local pymmcore MDA zarr
store and streams each completed timepoint (default `--connect` is
`tcp://127.0.0.1:5555`). Custom clients can reuse
`opym.stream.client.StreamSender`, which needs no MATLAB or GPU dependencies.

## The processed OME-Zarr store

With `OPYM_LIVE_FORMAT=zarr` the live lane writes one **processed store** per
dataset and nothing else: no intermediate TIFFs. It is an OME-Zarr in
[bioformats2raw](https://github.com/glencoesoftware/bioformats2raw) layout
version 3. It is built on the RAM disk (under `OPYM_LIVE_VIEW_ROOT`) and each
finished timepoint is then copied, unchanged, to
`<dataset>/viewer/<base_name>_dsr.ome.zarr` next to the raw stores:

* Series `0`: the deconvolved, deskewed and rotated image, axes (t, c, z, y, x),
  `uint16`, a three-level pyramid, chunked one timepoint and channel at a time.
* Series `1`: the Z maximum-intensity projection.
* `OME/METADATA.ome.xml`: OME-XML describing both series (written with `ome-types`).
* Channel colours: green for the first channel and magenta for the second, so
  two-colour composites are green/magenta (cyan and yellow for a third and fourth).
* `.opym_live.json`: which (timepoint, channel) pairs are written, so a viewer
  knows what is real and a store is known complete.

Bio-Formats reads the layout and OME-XML directly; readers that do not walk
bioformats2raw series can open the store root, which also carries the
multiscales for series `0`. In Python, `opym.ome_zarr_writer.image_group(store, "0")`
returns the image group of either this layout or the older single-image viewer
store.

Without `OPYM_LIVE_FORMAT=zarr` the receiver uses the earlier TIFF-based live lane.

## Viewing: `naparym-live`

`naparym-live [STORE]` opens napari in 3-D and follows a live acquisition:

* With no argument it opens immediately and follows whichever session is newest,
  switching when a new one starts. With a store path (or a dataset directory) it
  opens that store once. Any finished dataset's `viewer/*_dsr.ome.zarr` works.
* Each channel is shown as soon as it is processed, at full resolution, from
  memory-mapped view buffers written by the GPU server. The display is 8-bit by
  default (`--bits 16` for the stored 16-bit values); the stored data is always
  16-bit.
* When live QC is running, each timepoint's cell box is drawn coloured by its
  verdict.
* Options: `--no-follow`, `--poll SECONDS`, `--title TEXT`, `--cache-gb GB`
  (RAM for scrubbing, default 100), `--bits {8,16}`, and the experimental
  `--vram-gb` / `--vram-reserve-gb` GPU texture cache (off by default). Press `f`
  to toggle following.

For a run that has already finished, the Jupyter viewers in `opym.viewer` and
`opym.widgets` (installed with the `viewer` extra) are the interactive route.

## Environment variables

Read by opym's Python code (defaults in brackets):

| Variable | Meaning |
|---|---|
| `OPYM_DECON_PSF` | PSF file for deconvolution. Unset means deskew only. An invalid path is an error. |
| `OPYM_LIVE_LANE` | `1`, `true` or `yes`: the receiver hands streamed volumes to the live lane (needs `OPYM_DECON_PSF`). |
| `OPYM_LIVE_FORMAT` | `zarr`: the one-format lane. Anything else: the TIFF lane. |
| `OPYM_LIVE_VIEW_ROOT` | Where live view stores and buffers live [`/dev/shm/opym_live_view`]. |
| `OPYM_LIVE_VIEW_BITS` | Bit depth of the view buffers, `8` or `16` [`8`]. |
| `OPYM_LIVE_QC` | `1`: write raw-projection sidecars for an external live QC service. |
| `OPYM_STREAM_STAGE_ROOT` | RAM-disk staging root. Unset: write straight to the declared `raw_root`. |
| `OPYM_STREAM_STAGE_FLOOR_GB` | Free space the staging RAM disk keeps [`20`]. |
| `OPYM_STREAM_DIRECT_BIND` | Extra direct endpoint, e.g. `tcp://<ip>:5556`. |
| `OPYM_STREAM_ALLOW_IPS` | Comma-separated client IPs allowed on the direct endpoint. |
| `OPYM_STREAM_RAW_ROOTS` | Comma-separated folders the direct endpoint may write under. |
| `OPYM_BACKFILL_REGISTRY_PATH` | Backfill status registry (SQLite). A run moved aside by the receiver is forgotten there. |
| `OPYM_OUTPUT_MIRROR_ROOT` | Where output is redirected when a raw directory is not writable. |
| `OPYM_SERVE_SERVERS` | GPU servers as `id:gpu,id:gpu` [`1:0,2:1`]. |
| `OPYM_SERVE_HANG_MIN` | Minutes without output before a backfill claim is flagged hung [`60`]. |
| `OPYM_SERVE_KILL_HUNG` | Kill a flagged server and requeue its ticket [`1`]. |
| `OPYM_SERVE_LIVE_DEADLINE_S` | Seconds a live claim may run before its server is killed and the ticket requeued; `0` turns it off [`30`]. |
| `OPYM_LIVE_PREEMPT` | Preempt a backfill server for waiting live work [`1`]. |
| `OPYM_LIVE_PREEMPT_AFTER_S` | Seconds a queued live ticket waits before a busy backfill server is preempted for it [`30`]. When every GPU is on backfill, one is preempted at once. |
| `PETAKIT_JOBS_DIR` | Queue and status directory shared by the receiver, supervisor and servers [`/dev/shm/petakit_jobs`]. Point a test stack at its own. |

Read by the MATLAB PetaKit5D server: `PETAKIT_ROOT` (the PetaKit5D install),
`OPYM_PYTHON` (the Python interpreter MATLAB calls back into; falls back to
`python` on the path), and `PETAKIT_JOBS_DIR`. The supervisor sets the per-server
`PETAKIT_SERVER_ID`, `PETAKIT_GPU_ID`, `PETAKIT_CPUS` and `PETAKIT_IDLE_TIMEOUT`.

## Development

```bash
uv pip install -e ".[viewer,dev]"
pytest -m "not gpu"     # GPU tests also skip themselves without MATLAB and a GPU
just format             # ruff format + ruff check --fix
```

`scripts/live_bench/` runs an isolated copy of the live stack (its own queues,
port and server names) to measure the live view end to end and to inject faults;
see its README.

## License

MIT. See [LICENSE](LICENSE).
