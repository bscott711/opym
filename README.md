# **opym**

A GPU-accelerated processing pipeline for Oblique Plane Microscopy (OPM) data.

This project provides tools to de-interlace, deskew, and deconvolve raw OPM TIFF stacks, leveraging cupy for GPU-acceleration of deskewing and the llspy cudaDeconv binary for high-performance deconvolution.

## **Installation**

It is recommended to use uv for environment management.

1. **Create a virtual environment:**  
   uv venv

2. **Activate the environment:**  
   source .venv/bin/activate

3. Install Dependencies & opym:  
   First, install llspy from its local source directory. Then, install opym in editable mode.  
   \# From the directory where you have the llspy source  
   uv pip install \-e path/to/llspy\_source

   \# From the opym project root directory  
   uv pip install \-e .

## **Usage**

The pipeline is run from the command line. You must provide the path to the input file and an output directory. Other parameters are optional and have sensible defaults.

opym process /path/to/your/interlaced\_data.tif /path/to/your/output\_directory \--angle 31.5 \--num-channels 2 \--iterations 10

For a full list of options, run:

opym \--help  
