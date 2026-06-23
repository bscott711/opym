#!/usr/bin/env python3
import tifffile
import numpy as np
import psfmodels as psfm
from scipy.ndimage import affine_transform
from pathlib import Path
import matplotlib.pyplot as plt

def generate_and_save_psf(wavelength_nm, out_path):
    print(f"Generating PSF for {wavelength_nm}nm...")
    # Use a symmetric cube size for the PSF
    z_shape = 63
    # Generate on a much larger padded XY grid to prevent edge artifacts during shear
    padded_nx = 127 
    
    # Generate the unskewed PSF as a purely monotonic 3D Gaussian
    # We use empirical widths measured directly from the Master PSF (at ~525nm).
    # Correcting for Python vs MATLAB dimension reading (Z vs X axis swap):
    base_wvl = 525.0
    scale = wavelength_nm / base_wvl
    
    sigma_x = 0.11 * scale
    sigma_y = 0.29 * scale
    sigma_z = 0.21 * scale
    
    # We evaluate the Gaussian analytically on the sheared coordinate grid.
    # This completely eliminates any interpolation ringing artifacts that `scipy.ndimage.affine_transform`
    # introduces when using cubic splines on a sub-pixel sharp Gaussian.
    
    target_z = 63
    target_y = 63
    target_x = 63
    
    z = (np.arange(target_z) - target_z // 2) * 0.1
    y = (np.arange(target_y) - target_y // 2) * 0.136
    x = (np.arange(target_x) - target_x // 2) * 0.136
    
    ZZ, YY, XX = np.meshgrid(z, y, x, indexing='ij')
    
    # Apply OPM shear analytically.
    # We negate the tangent to reverse the slope and match the Master PSF.
    theta = np.radians(30.0)
    YY_unskewed = YY - ZZ * (-np.tan(theta))
    
    # Evaluate pure 3D Gaussian
    skewed_psf = np.exp(-(XX**2 / (2 * sigma_x**2) + YY_unskewed**2 / (2 * sigma_y**2) + ZZ**2 / (2 * sigma_z**2)))
    psf_data = np.exp(-(XX**2 / (2 * sigma_x**2) + YY**2 / (2 * sigma_y**2) + ZZ**2 / (2 * sigma_z**2)))
    
    # Normalize
    skewed_psf = skewed_psf / np.max(skewed_psf)
    skewed_psf = skewed_psf.astype(np.float32)
    
    out_path.parent.mkdir(parents=True, exist_ok=True)
    tifffile.imwrite(out_path, skewed_psf, imagej=True)
    
    # Generate visualization
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    
    axes[0].imshow(np.max(psf_data, axis=0), cmap='magma')
    axes[0].set_title(f'Unskewed XY MIP ({wavelength_nm}nm)')
    axes[0].axis('off')
    
    axes[1].imshow(np.max(skewed_psf, axis=0), cmap='magma')
    axes[1].set_title('Skewed XY MIP')
    axes[1].axis('off')
    
    axes[2].imshow(np.max(skewed_psf, axis=2), cmap='magma')
    axes[2].set_title('Skewed YZ MIP (showing shear)')
    axes[2].axis('off')
    
    png_path = out_path.with_suffix('.png')
    fig.savefig(png_path, bbox_inches='tight', dpi=150)
    plt.close(fig)
    
    print(f"✅ Saved: {out_path.name} and visualization")

if __name__ == "__main__":
    out_dir = Path("/mmfs2/scratch/SDSMT.LOCAL/bscott/DataUpload/PSF")
    
    emissions = {
        "GFP": 525,
        "Calcein_Violet": 450,
        "mScarlet": 595,
        "CF647": 670,
    }
    
    for name, wvl in emissions.items():
        out_path = out_dir / f"Synthetic_PSF_{name}_{wvl}nm.tif"
        generate_and_save_psf(wvl, out_path)
    
    print("All synthetic PSFs generated successfully!")
