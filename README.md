# **opym: OPM Processing in Python**

opym (pronounced "opium") is a command-line tool for GPU-accelerated processing of Oblique Plane Microscopy (OPM) data. It provides a streamlined pipeline for de-interlacing, deskewing, and deconvolving raw TIFF stacks.

## **Features**

* **Automated Discovery:** Point it at a directory, and it automatically finds data based on AcqSettings.txt.  
* **GPU Acceleration:** Leverages cupy for fast deskewing and pycudadecon for best-in-class deconvolution performance.  
* **Flexible:** Command-line arguments can override any parameters found in metadata.  
* **User-Friendly:** Provides a graphical folder-picker if no input directory is specified.

## **Installation**

This package is designed to be installed and run using uv and pip. It relies on pycudadecon, which requires a CUDA-enabled NVIDIA GPU and a compatible version of the **NVIDIA CUDA Toolkit**.

Because the pip-installable version of pycudadecon does not include the compiled CUDA engine, a one-time manual download and setup of the libcudaDecon.dll file is required.

### **1\. Create a Virtual Environment with uv**

Navigate to your project directory and create a new virtual environment.

\# Create the environment  
uv venv

\# Activate the environment  
.\\.venv\\Scripts\\Activate.ps1

### **2\. Manually Install the cudaDecon Library**

This is the critical manual step. You need to download the compiled library from the Anaconda distribution channel and place it where your system can find it.

1. **Download the Library:**  
   * Go to the pycudadecon file list on Anaconda: [https://anaconda.org/conda-forge/pycudadecon/files](https://www.google.com/search?q=https://anaconda.org/conda-forge/pycudadecon/files)  
   * Find a recent version that matches your architecture (e.g., win-64) and CUDA toolkit version.  
   * Download the .tar.bz2 file.  
2. **Extract the .dll:**  
   * Use a tool like 7-Zip to open the downloaded .tar.bz2 archive.  
   * Navigate inside to the Library/bin/ directory.  
   * Extract the libcudaDecon.dll file to a known, permanent location on your computer. A good practice is to create a dedicated folder, for example: C:\\tools\\lib.  
3. **Add to System PATH:**  
   * You must add the folder containing libcudaDecon.dll to your system's PATH environment variable so that Windows can find it.  
   * Press the Windows Key, type env, and select "Edit the system environment variables".  
   * Click "Environment Variables...".  
   * Under "System variables", select Path and click "Edit...".  
   * Click "New" and add the path to the folder where you saved the .dll (e.g., C:\\tools\\lib).  
   * Click OK on all windows to save.  
   * **You must restart your terminal for this change to take effect.**

### **3\. Install opym and Dependencies**

After restarting your terminal and reactivating the uv environment, navigate to the root directory of the opym project and run:

uv pip install \-e .

This will install opym, the Python wrapper for pycudadecon, and all other dependencies into your uv environment. The opym command will now be available in your terminal.

## **Usage**

Once installed, you can run the pipeline from your activated uv environment.

**Basic Usage (with folder selection dialog):**

opym

**Recommended Usage (specifying an input directory):**

opym "C:\\path\\to\\your\\data\_folder"  
