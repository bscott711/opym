# cpp-zarr read-path patch

Shadow-patched copy of `parallelReadZarr` from the shared PetaKit5D install
(`/cm/shared/apps_local/petakit5d`, not writable by this account -- owned by
`software.cluster@jacks.local`). `run_petakit_server.m` adds `linux/` to the
MATLAB path after the shared install's own `setup.m`, so this shadows the
broken shared `parallelReadZarr.mexa64` via normal MATLAB path search order,
same mechanism as `../XR_deskewRotateFrame.m`.

## What's fixed

The shared build can't read a spec-legal `"compressor": null` (uncompressed)
zarr v2 array: `zarr.cpp`'s constructor unconditionally tries both a blosc-
shaped and a gzip-shaped `.at()` on `compressor`, both throw on JSON null,
and the outer catch mislabels it "Metadata is incomplete. Check the .zarray
file". `parallelreadzarr.cpp`'s read loop also has no raw/uncompressed
branch at all -- every chunk goes through blosc2_decompress or zlib inflate
unconditionally. Patched both:
- `src/zarr.cpp`: `compressor.is_null()` -> `cname="raw"` sentinel instead
  of falling through to the generic "metadataIncomplete" catch-all.
- `src/parallelreadzarr.cpp`: `cname=="raw"` -> memcpy chunk bytes straight
  through, bounded to `min(fileLen, sB)` for a short edge chunk.

`src/helperfunctions.cpp` and the three `.h` headers are unmodified copies,
needed only because `zarr.cpp`/`parallelreadzarr.cpp` are compiled as their
own translation units, not linked against the shared install's object files.

## Rebuilding

```bash
CONDA_INC=/cm/shared/apps_local/python/3.12/include   # blosc2.h, uuid/uuid.h
CONDA_LIB=/cm/shared/apps_local/python/3.12/lib        # libblosc2.so
NJSON_INC=<this dir>/thirdparty                        # vendored nlohmann/json.hpp

cd mexSrc
/mmfs2/cm/shared/apps_local/matlab/R2024B/bin/mex -outdir ../linux -output parallelReadZarr.mexa64 \
  CXXOPTIMFLAGS="-DNDEBUG -O2" LDOPTIMFLAGS="-O2 -DNDEBUG" \
  CXXFLAGS='$CXXFLAGS -fopenmp -O2' \
  LDFLAGS="\$LDFLAGS -fopenmp -O2 -Wl,-rpath,$CONDA_LIB" \
  -I"$CONDA_INC" -I"$NJSON_INC" -L"$CONDA_LIB" \
  -lblosc2 -lz -luuid \
  parallelreadzarrmex.cpp ../src/zarr.cpp ../src/helperfunctions.cpp ../src/parallelreadzarr.cpp
```

No `patchelf` ABI workaround needed (unlike the vendor's own
`compile_parallelReadZarr.m`, written for a different MATLAB/GCC pairing):
confirmed this build only requires up to `GLIBCXX_3.4.29`, and MATLAB
R2024b's bundled `libstdc++.so.6.0.30` provides up to `3.4.30`.

The blosc2/uuid headers+libs come from the shared Python 3.12 conda env
(`/cm/shared/apps_local/python/3.12`) -- read-only use, nothing installed or
modified there. `nlohmann/json.hpp` (v3.11.3) is vendored into
`thirdparty/` rather than depending on an unrelated shared package's copy.

## Testing

Needs `LM_LICENSE_FILE=27000@pioneer` set explicitly for a standalone
`matlab -batch` test run from an interactive shell (the production launcher
gets this from `module load matlab/R2024b`, which an interactive shell here
doesn't run by default) -- otherwise MATLAB falls back to a host-locked
license file that doesn't match this machine and fails to start.

Verified against a real `compressor: null` dataset (correct shape/dtype,
full sensible pixel value range) and a synthetic 3D blosc-compressed array
(regression check -- exact expected values, confirming the pre-existing
compressed path is untouched).
