# Ruff style: Compliant
# Description:
# This module, adapted from llspy, handles the downloading and installation
# of the cudaDeconv binary required for deconvolution.

import os
import platform
import shutil
import stat
import sys
import tempfile
import zipfile
from urllib.request import urlretrieve

# --- Constants ---
VERSION = "0.5.1"
LIBNAME = "cudaDeconv"
REPO = "Nico-Stien-UCSF/cudaDeconv"
MIN_GLIBC_VERSION = 2.17
DEFAULT_CUDA_VERSION = "11.2"
URL = f"https://github.com/{REPO}/releases/download/v{VERSION}/{LIBNAME}_v{VERSION}_{{platform}}_cuda{{cuda}}.zip"


# --- Helper Functions ---
def get_app_dir(app_name="opym", roaming=True):
    """Get the application data directory."""
    if sys.platform == "win32":
        if roaming:
            return os.path.join(os.environ["APPDATA"], app_name)
        else:
            return os.path.join(os.environ["LOCALAPPDATA"], app_name)
    elif sys.platform == "darwin":
        return os.path.join(os.path.expanduser("~"), "Library", "Application Support", app_name)
    else:
        return os.path.join(os.path.expanduser("~"), ".config", app_name)


def get_lib_dir():
    """Return the directory where the cudaDeconv binary is stored."""
    return os.path.join(get_app_dir(), "lib")


def get_lib_path():
    """Return the full path to the cudaDeconv binary."""
    libdir = get_lib_dir()
    if sys.platform.startswith("win"):
        return os.path.join(libdir, LIBNAME + ".exe")
    else:
        return os.path.join(libdir, LIBNAME)


def get_cuda_version():
    """Attempt to detect the CUDA version."""
    if sys.platform.startswith("linux"):
        try:
            with os.popen("nvcc --version") as p:
                output = p.read()
            version_line = [
                line for line in output.split("\n") if "release" in line
            ]
            if version_line:
                version = version_line[0].split("release ")[1].split(",")[0]
                return version
        except Exception:
            pass
    # Default for Windows/macOS or if detection fails
    return DEFAULT_CUDA_VERSION


# --- Core Functions ---
def check_lib_installed():
    """Check if the cudaDeconv binary is installed and executable."""
    lib_path = get_lib_path()
    return os.path.isfile(lib_path) and os.access(lib_path, os.X_OK)


def install_lib(cuda_version=None):
    """Download and install the cudaDeconv binary."""
    if cuda_version is None:
        cuda_version = get_cuda_version()

    opsys = platform.system().lower()
    if opsys == "darwin":
        opsys = "mac"
    pform = f"{opsys}-{platform.machine()}"

    url = URL.format(platform=pform, cuda=cuda_version.replace(".", ""))
    lib_dir = get_lib_dir()
    os.makedirs(lib_dir, exist_ok=True)
    lib_path = get_lib_path()

    print(f"Downloading cudaDeconv from: {url}")
    try:
        with tempfile.TemporaryDirectory() as tmpdir:
            zip_path = os.path.join(tmpdir, "cudadeconv.zip")
            urlretrieve(url, zip_path)
            with zipfile.ZipFile(zip_path, "r") as zip_ref:
                zip_ref.extractall(tmpdir)
            extracted_bin = os.path.join(tmpdir, os.path.basename(lib_path))
            shutil.move(extracted_bin, lib_path)
        # Make the file executable
        st = os.stat(lib_path)
        os.chmod(lib_path, st.st_mode | stat.S_IEXEC)
        print(f"Successfully installed {LIBNAME} to {lib_path}")
        return True
    except Exception as e:
        print(f"Failed to download or install cudaDeconv: {e}")
        return False

