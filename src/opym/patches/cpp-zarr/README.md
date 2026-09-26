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

## N-D arrays (leading index) and opymWriteZarrBlock

PetaKit5D's zarr code is 3-D throughout. The live pipeline's stores are not:
a raw channel store is `(T, Z, Y, X)` and the processed OME-Zarr is
`(T, C, Z, Y, X)`. Both are C order, with chunk size 1 on every leading axis
and `/` as the dimension separator. For such an array, the block at fixed
leading indices is exactly a 3-D array rooted at `<store>/<t>/<c>/`.
`zarr::set_leadingIndex` (in `src/zarr.cpp`) re-roots the zarr object there
and keeps the trailing three axes, so all the 3-D code runs unchanged. It
never rewrites the `.zarray`.

- `parallelReadZarr(store, 'leadingIndex', [t+1])` returns one raw
  timepoint `(Z, Y, X)`. It takes 1-based indices, like `bbox`.
- `opymWriteZarrBlock(store, data, [t+1 c+1])` (`mexSrc/opymwritezarrblockmex.cpp`)
  writes `data` (size = the trailing shape, class = the dtype) through
  upstream's unmodified `parallelWriteZarr` (`src/parallelwritezarr.*`, copied
  from the shared install). It uses the array's own compressor, writes each
  chunk to a temp name then renames it, and skips all-zero chunks.

- `parallelReadZarr(store, 'leadingIndex', [t+1], 'orientForDecon', true)`
  returns `flip(permute(zyx, [3 2 1]), 1)` directly -- the `(X, Y, Z)` array
  the live job deconvolves -- by reversing each C-order row, instead of the
  element-wise transpose to `(Z, Y, X)` that MATLAB then undid (0.16 s ->
  0.02 s for a 161-plane volume on /dev/shm). C-order arrays only.
- `t = opymWriteLiveOutputs(dsr, npyPath, levelStores, mipStore, [t+1 c+1])`
  (`mexSrc/opymwriteliveoutputsmex.cpp`) writes everything the live job
  produces for one deskewed `(Y, X, Z)` volume from a single parallel
  transpose into C order: the live viewer's `.npy` buffer first (published
  before anything is encoded), then every pyramid level (2x block means,
  opym's `downsample2`) and the Z-MIP, compressed with the same blosc call as
  `opymWriteZarrBlock`. After each call a background thread readies the next
  buffer file (allocated, every page mapped) so the next transpose never
  page-faults. Replaced a MATLAB permute + fwrite, a second permute, the
  C-order chunk gather (a cache miss per voxel) and a GPU round trip:
  ~2.6 s -> ~0.3 s per volume.

All are checked against zarr-python in `tests/test_cpp_zarr_nd.py`
(`-m gpu`), and end to end, bit for bit, against the TIFF live path in
`tests/test_live_zarr_equivalence.py`. A 161-plane raw volume takes ~0.4 s
to read from GPFS.

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

The block writer additionally links blosc v1 (upstream's writer calls
`blosc_compress_ctx`):

```bash
/mmfs2/cm/shared/apps_local/matlab/R2024B/bin/mex -outdir ../linux -output opymWriteZarrBlock.mexa64 \
  CXXOPTIMFLAGS="-DNDEBUG -O2" LDOPTIMFLAGS="-O2 -DNDEBUG" \
  CXXFLAGS='$CXXFLAGS -fopenmp -O2' \
  LDFLAGS="\$LDFLAGS -fopenmp -O2 -Wl,-rpath,$CONDA_LIB" \
  -I"$CONDA_INC" -I"$NJSON_INC" -L"$CONDA_LIB" \
  -lblosc -lblosc2 -lz -luuid \
  opymwritezarrblockmex.cpp ../src/zarr.cpp ../src/helperfunctions.cpp \
  ../src/parallelwritezarr.cpp ../src/parallelreadzarr.cpp
```

The live-output writer (blosc v1, and `-march=x86-64-v3`: AVX2, which every
Argus node has):

```bash
/mmfs2/cm/shared/apps_local/matlab/R2024B/bin/mex -outdir ../linux -output opymWriteLiveOutputs.mexa64 \
  CXXOPTIMFLAGS="-DNDEBUG -O3" LDOPTIMFLAGS="-O3 -DNDEBUG" \
  CXXFLAGS='$CXXFLAGS -fopenmp -O3 -march=x86-64-v3' \
  LDFLAGS="\$LDFLAGS -fopenmp -O3 -pthread -Wl,-rpath,$CONDA_LIB" \
  -I"$CONDA_INC" -I"$NJSON_INC" -L"$CONDA_LIB" \
  -lblosc -lz -luuid \
  opymwriteliveoutputsmex.cpp ../src/zarr.cpp ../src/helperfunctions.cpp
```

To run the `-m gpu` tests from a shell: `module load matlab/R2024b`,
`LM_LICENSE_FILE=27000@pioneer`, and
`LD_PRELOAD=/cm/shared/apps_local/matlab/R2024B/sys/os/glnxa64/libstdc++.so.6`
(the MATLAB engine needs `GLIBCXX_3.4.30`, and the system libstdc++ lacks it).

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
