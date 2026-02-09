"""
opym.batch - Orchestration logic for batch processing workflows.
"""

import time
from pathlib import Path

import ipywidgets as widgets

# Intra-package imports
from . import metadata, petakit


def run_batch_cropping(
    template_folder: Path,
    file_list: list[Path],
    settings: dict,
    log_output: widgets.Output,
    progress_bar: widgets.IntProgress,
    status_label: widgets.Label,
) -> None:
    """
    Orchestrates the submission and monitoring of batch cropping and deskewing jobs.

    Args:
        template_folder: The folder containing the 'petakit_settings.json' source.
        file_list: List of .ome.tif files to process.
        settings: The loaded settings dictionary.
        log_output: Widget to print logs to.
        progress_bar: Widget to update progress.
        status_label: Widget to update text status.
    """

    # --- 1. Extract Settings ---
    channels = settings["channels"]
    rotate = settings["rotate"]
    fmt = settings["format"]

    # Helper to parse string slice "0:100" -> slice(0, 100)
    def _parse_roi(s: str) -> tuple[slice, slice]:
        parts = s.split(",")
        y_parts = [int(v) for v in parts[0].split(":")]
        x_parts = [int(v) for v in parts[1].split(":")]
        return (slice(y_parts[0], y_parts[1]), slice(x_parts[0], x_parts[1]))

    top_roi = _parse_roi(settings["rois"]["top"])
    bot_roi = _parse_roi(settings["rois"]["bottom"])

    # Deskew Defaults
    ds = settings.get("deskew", {})
    default_angle = float(ds.get("sheet_angle_deg", 31.8))
    default_pixel = float(ds.get("xy_pixel_size", 0.136))
    default_z = float(ds.get("z_step_um", 1.0))

    # Store active jobs: (filename, crop_ticket_path, deskew_ticket_path)
    active_jobs: list[tuple[str, Path, Path]] = []

    # --- 2. Submission Phase ---
    with log_output:
        for i, file_path in enumerate(file_list):
            status_label.value = f"Submitting {i + 1}/{len(file_list)}..."
            try:
                # A. Submit Crop
                crop_ticket = petakit.submit_remote_crop_job(
                    base_file=file_path,
                    top_roi=top_roi,
                    bottom_roi=bot_roi,
                    channels=channels,
                    output_format=fmt,
                    rotate=rotate,
                )

                # B. Prepare Deskew Target
                # Logic to clean filename
                if file_path.name.endswith(".ome.tif"):
                    clean_name = file_path.name[:-8]
                elif file_path.name.endswith(".tif"):
                    clean_name = file_path.name[:-4]
                else:
                    clean_name = file_path.stem

                target_dir = file_path.parent / clean_name

                # C. Detect Z-Step (or use default)
                z_step = default_z
                meta_file = file_path.parent / "AcqSettings.txt"
                if meta_file.exists():
                    try:
                        z_step = metadata.parse_z_step(meta_file, default_z)
                    except Exception:  # nosec
                        pass  # Keep default

                # D. Submit Deskew
                deskew_ticket = petakit.submit_remote_deskew_job(
                    input_target=target_dir,
                    z_step_um=z_step,
                    xy_pixel_size=default_pixel,
                    sheet_angle_deg=default_angle,
                    deskew=True,
                    rotate=True,
                )

                active_jobs.append((file_path.name, crop_ticket, deskew_ticket))
                print(f"[{i + 1}] Submitted: {file_path.name}")

            except Exception as e:
                print(f"❌ Failed: {file_path.name} - {e}")
                # We do not raise here, to allow other files to proceed

    # --- 3. Monitoring Phase ---
    if not active_jobs:
        status_label.value = "No jobs were successfully submitted."
        return

    status_label.value = f"Monitoring {len(active_jobs)} jobs in Petakit queue..."
    progress_bar.max = len(active_jobs)
    progress_bar.value = 0

    completed_indices: set[int] = set()

    try:
        while len(completed_indices) < len(active_jobs):
            for idx, (name, ct, dt) in enumerate(active_jobs):
                if idx in completed_indices:
                    continue

                # Check Queue Status
                # Logic: If ticket GONE from queue folder -> Started/Done
                base_queue = dt.parent.parent / "queue"

                # We check if EITHER ticket is still present
                in_queue = (base_queue / dt.name).exists() or (
                    base_queue / ct.name
                ).exists()

                if not in_queue:
                    completed_indices.add(idx)
                    progress_bar.value += 1
                    with log_output:
                        print(f"✅ Finished in Queue: {name}")

            if len(completed_indices) < len(active_jobs):
                time.sleep(2)

        status_label.value = "✅ All batch jobs processed from queue!"
        progress_bar.bar_style = "success"

    except KeyboardInterrupt:
        status_label.value = "🛑 Monitoring stopped by user."
        with log_output:
            print("\nMonitoring stopped. Jobs may still be running on server.")
