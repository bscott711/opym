# Ruff style: Compliant
# Description:
# This module provides the command-line interface for opym.

import sys
from pathlib import Path
from tkinter import filedialog

from . import core, metadata


def main() -> None:
    """Main entry point for the opym command-line interface."""
    # --- Get Input/Output Directories ---
    input_dir_str = sys.argv[1] if len(sys.argv) > 1 else None

    if not input_dir_str:
        print("No input directory specified. Opening file dialog...")
        input_dir_str = filedialog.askdirectory(title="Select Input Directory")
        if not input_dir_str:
            print("No directory selected. Exiting.")
            return

    input_dir = Path(input_dir_str)
    output_dir_str = sys.argv[2] if len(sys.argv) > 2 else None

    if not output_dir_str:
        output_dir = input_dir / "processed"
        print(f"No output directory specified. Defaulting to: {output_dir}\n")
    else:
        output_dir = Path(output_dir_str)

    output_dir.mkdir(exist_ok=True)

    print("--- Universal Light Sheet Processing Pipeline ---")

    # --- Parse Metadata ---
    # Find the settings file, whether it's AcqSettings.txt or Settings.txt
    settings_path = next(input_dir.glob("*ettings.txt"), None)
    if not settings_path:
        print(f"Could not find a settings file in {input_dir}. Exiting.")
        sys.exit(1)

    meta_params = metadata.parse_settings(settings_path)
    if not meta_params:
        print(f"Could not parse {settings_path}. Exiting.")
        sys.exit(1)

    # --- Find Image Files ---
    prefix = meta_params.get("save_name_prefix")
    if not prefix:
        print("Could not find 'save_name_prefix' in metadata. Exiting.")
        sys.exit(1)

    image_files = sorted(input_dir.glob(f"{prefix}*.tif"))
    if not image_files:
        print(f"No TIFF files found with prefix '{prefix}' in {input_dir}. Exiting.")
        sys.exit(1)

    print(f"Found {len(image_files)} TIFF files to process.")

    # --- Process Each File ---
    all_successful = True
    for image_file in image_files:
        print(f"\nProcessing file: {image_file.name}")
        success = core.process_file(
            filepath=image_file,
            output_dir=output_dir,
            params=meta_params,
        )
        if not success:
            all_successful = False
            break

    # --- Final Status Message ---
    if all_successful:
        print("\n--- Pipeline finished successfully! ---")
    else:
        print("\n--- Pipeline finished with errors. ---")


if __name__ == "__main__":
    main()
