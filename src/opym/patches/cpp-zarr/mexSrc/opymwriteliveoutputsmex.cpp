// t = opymWriteLiveOutputs(dsr, npyPath, levelStores, mipStore, leadingIndex)
//
// Everything the live job writes for one deskewed (t, c) volume, from a
// single transpose. deskewRotateFrame3D returns (Y, X, Z) column-major and
// every output wants C-order (Z, Y, X); run_live_zarr.m used to get there
// with a MATLAB permute + fwrite for the view buffer (~1.2 s), another
// permute for level 0, cpp-zarr's C-order chunk gather (a cache miss per
// voxel) and a GPU round trip for the pyramid (~1.3 s together).
//
// In order:
//  1. npyPath: the view buffer, C-order (Z, Y, X) uint16 .npy v1.0 (the
//     header writeNpyZYX wrote). The transpose goes straight into a memory
//     map of a temporary file, renamed into place before anything else
//     happens, so the viewer never waits on encoding. With npyPath '' the
//     transpose goes to memory and no view buffer is written.
//
//     Mapping a fresh 1 GB file costs ~0.3-0.4 s in page faults on the RAM
//     disk -- more than the transpose itself (~0.02 s). So after each call a
//     background thread readies the next buffer (a hidden .opym_prep_<pid>
//     file in the same directory, allocated and mapped with every page
//     present) while the server waits for its next ticket; the next call of
//     the same shape in the same directory transposes into it and renames it
//     into place. The previous buffer is unmapped by that thread too. Any
//     other call (first of a session, new shape) falls back to a fresh file.
//  2. levelStores{1..n}: level 0 is that volume, level k the 2x block mean of
//     level k-1 (uint32 sum of 8, floor /8, odd trailing plane/row/column
//     dropped -- opym.ome_zarr_writer.downsample2), each at leadingIndex.
//  3. mipStore: the max over Z, as a (1, Y, X) block at leadingIndex.
//
// Chunks are compressed with the call opymWriteZarrBlock / PetaKit's
// parallelWriteZarr makes -- blosc_compress_ctx(clevel, BLOSC_SHUFFLE, 2,
// ..., cname, 0, nthreads), edge chunks full size with the fill value --
// always single-threaded here. Every decoded value is identical to what that
// writer stores; the compressed bytes are too wherever it also ran blosc on
// one thread (level 0 and the MIP). All-zero chunks are skipped (read back
// as fill_value) and each chunk lands under a temporary name and is renamed
// into place. Stores must exist (opym creates them); their .zarray is only
// read.
//
// Returns t = [view_s, write_s]: seconds to the published view buffer, then
// for all the zarr writes.
#include <algorithm>
#include <chrono>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <fcntl.h>
#include <omp.h>
#include <set>
#include <string>
#include <sys/mman.h>
#include <thread>
#include <unistd.h>
#include <vector>
#include "blosc.h"
#include "mex.h"
#include "../src/helperfunctions.h"
#include "../src/zarr.h"

namespace {

using Clock = std::chrono::steady_clock;

struct Block {
    std::string root;  // <store>/<t>/<c>
    uint64_t shape[3];
    uint64_t chunks[3];
    std::string cname;
    int clevel;
    uint16_t fill;
};

Block openBlock(const std::string &store, const std::vector<uint64_t> &leading,
                const uint64_t shape[3])
{
    zarr Zarr;
    try {
        Zarr = zarr(store);
    } catch (const std::string &e) {
        mexErrMsgIdAndTxt("opymLive:zarray", "Cannot open %s (%s)", store.c_str(), e.c_str());
    }
    try {
        Zarr.set_leadingIndex(leading);
    } catch (const std::string &e) {
        mexErrMsgIdAndTxt("opymLive:leadingIndex",
                          "Cannot index %s that way (%s): it needs a C-order array with '/' "
                          "separators, chunk size 1 on every leading axis, and indices in range.",
                          store.c_str(), e.c_str());
    }
    if (Zarr.get_dtype() != "<u2")
        mexErrMsgIdAndTxt("opymLive:dtype", "%s holds %s, not uint16", store.c_str(),
                          Zarr.get_dtype().c_str());
    const std::string cname = Zarr.get_cname();
    if (cname == "raw" || cname == "gzip")
        mexErrMsgIdAndTxt("opymLive:codec", "%s is %s; only blosc arrays are written here",
                          store.c_str(), cname.c_str());
    Block b;
    b.root = Zarr.get_fileName();
    for (int i = 0; i < 3; i++) {
        b.shape[i] = Zarr.get_shape(i);
        b.chunks[i] = Zarr.get_chunks(i);
        if (b.shape[i] != shape[i])
            mexErrMsgIdAndTxt("opymLive:shape",
                              "%s holds blocks of [%llu %llu %llu], not [%llu %llu %llu]",
                              store.c_str(), (unsigned long long)b.shape[0],
                              (unsigned long long)b.shape[1], (unsigned long long)b.shape[2],
                              (unsigned long long)shape[0], (unsigned long long)shape[1],
                              (unsigned long long)shape[2]);
    }
    b.cname = cname;
    b.clevel = (int)Zarr.get_clevel();
    try {
        b.fill = (uint16_t)std::stoi(Zarr.get_fill_value());
    } catch (...) {
        b.fill = 0;
    }
    return b;
}

// Write a C-order (Z, Y, X) uint16 volume as the block's chunks. Returns an
// error message, empty on success.
std::string writeBlock(const Block &b, const uint16_t *vol)
{
    const uint64_t Z = b.shape[0], Y = b.shape[1], X = b.shape[2];
    const uint64_t cz = b.chunks[0], cy = b.chunks[1], cx = b.chunks[2];
    const uint64_t nz = (Z + cz - 1) / cz, ny = (Y + cy - 1) / cy, nx = (X + cx - 1) / cx;
    const uint64_t nChunks = nz * ny * nx;
    const uint64_t n = cz * cy * cx;
    const uint64_t sB = n * sizeof(uint16_t);

    // "/"-separated keys are nested directories; create the parents serially.
    for (uint64_t iz = 0; iz < nz; iz++)
        for (uint64_t iy = 0; iy < ny; iy++) {
            const std::string dir = b.root + "/" + std::to_string(iz) + "/" + std::to_string(iy);
            mkdirRecursive(dir.c_str());
        }

    const std::string suffix = ".opym" + std::to_string(getpid()) + ".tmp";
    std::string err;
    #pragma omp parallel
    {
        std::vector<uint16_t> unc(n);
        std::vector<uint8_t> comp(sB + BLOSC_MAX_OVERHEAD);
        #pragma omp for schedule(dynamic)
        for (int64_t f = 0; f < (int64_t)nChunks; f++) {
            if (!err.empty()) continue;
            const uint64_t iz = f / (ny * nx), iy = (f / nx) % ny, ix = f % nx;
            const uint64_t x0 = ix * cx;
            const uint64_t xn = x0 < X ? std::min(cx, X - x0) : 0;
            bool any = false;
            for (uint64_t lz = 0; lz < cz; lz++) {
                const uint64_t z = iz * cz + lz;
                for (uint64_t ly = 0; ly < cy; ly++) {
                    const uint64_t y = iy * cy + ly;
                    uint16_t *dst = unc.data() + (lz * cy + ly) * cx;
                    uint64_t done = 0;
                    if (z < Z && y < Y && xn) {
                        const uint16_t *src = vol + (z * Y + y) * X + x0;
                        std::memcpy(dst, src, xn * sizeof(uint16_t));
                        done = xn;
                    }
                    std::fill(dst + done, dst + cx, b.fill);
                }
            }
            for (uint64_t k = 0; k < n && !any; k++) any = unc[k] != 0;
            if (!any) continue;  // as parallelWriteZarr's sparse mode: fill_value on read
            const int csize = blosc_compress_ctx(b.clevel, BLOSC_SHUFFLE, sizeof(uint16_t), sB,
                                                 unc.data(), comp.data(), sB + BLOSC_MAX_OVERHEAD,
                                                 b.cname.c_str(), 0, 1);
            const std::string name = b.root + "/" + std::to_string(iz) + "/" +
                                     std::to_string(iy) + "/" + std::to_string(ix);
            const std::string tmp = name + suffix + std::to_string(omp_get_thread_num());
            bool ok = csize > 0;
            if (ok) {
                const int fd = open(tmp.c_str(), O_WRONLY | O_CREAT | O_TRUNC, 0644);
                ok = fd >= 0;
                if (ok) {
                    ok = write(fd, comp.data(), csize) == csize;
                    ok = (close(fd) == 0) && ok;
                }
                ok = ok && rename(tmp.c_str(), name.c_str()) == 0;
            }
            if (!ok) {
                #pragma omp critical
                if (err.empty()) err = "Cannot write chunk " + name;
            }
        }
    }
    return err;
}

std::string npyHeader(uint64_t Z, uint64_t Y, uint64_t X)
{
    // What writeNpyZYX wrote: the whole header a multiple of 64 bytes.
    std::string dict = "{'descr': '<u2', 'fortran_order': False, 'shape': (" + std::to_string(Z) +
                       ", " + std::to_string(Y) + ", " + std::to_string(X) + "), }";
    const size_t pad = (64 - ((10 + dict.size() + 1) % 64)) % 64;
    dict.append(pad, ' ');
    dict.push_back('\n');
    std::string h;
    h.push_back((char)0x93);
    h += "NUMPY";
    h.push_back((char)1);
    h.push_back((char)0);
    const uint16_t len = (uint16_t)dict.size();
    h.push_back((char)(len & 0xff));
    h.push_back((char)(len >> 8));
    return h + dict;
}

std::string str(const mxArray *a, const char *what)
{
    if (!mxIsChar(a)) mexErrMsgIdAndTxt("opymLive:input", "%s must be a string", what);
    if (mxIsEmpty(a)) return std::string();
    char *s = mxArrayToString(a);
    std::string out(expandTilde(s));
    mxFree(s);
    return out;
}

// --- the next view buffer, readied in the background ---------------------
struct Prepared {
    std::string dir, path;
    uint64_t size = 0;
    void *map = nullptr;
};
std::thread prepThread;
Prepared prep;  // written only by prepThread; read after joining it
bool locked = false;

void joinPrep()
{
    if (prepThread.joinable()) prepThread.join();
}

void dropPrep()
{
    joinPrep();
    if (prep.map) munmap(prep.map, prep.size);
    if (!prep.path.empty()) unlink(prep.path.c_str());
    prep = Prepared();
}

void atExit()
{
    dropPrep();
}

// Unmap the buffer just published (in the background too: tearing down a
// 1 GB mapping isn't free), then ready the next one.
void startPrep(const std::string &dir, uint64_t size, void *oldMap, uint64_t oldSize)
{
    if (!locked) {
        mexLock();  // a running thread must never outlive the code it runs
        mexAtExit(atExit);
        locked = true;
    }
    prepThread = std::thread([dir, size, oldMap, oldSize]() {
        if (oldMap) munmap(oldMap, oldSize);
        Prepared p;
        p.path = dir + "/.opym_prep_" + std::to_string(getpid()) + ".npy.tmp";
        const int fd = open(p.path.c_str(), O_RDWR | O_CREAT | O_TRUNC, 0644);
        if (fd < 0) return;
        void *m = MAP_FAILED;
        if (fallocate(fd, 0, 0, (off_t)size) == 0)
            m = mmap(nullptr, size, PROT_READ | PROT_WRITE, MAP_SHARED | MAP_POPULATE, fd, 0);
        close(fd);
        if (m == MAP_FAILED) {
            unlink(p.path.c_str());
            return;
        }
        // Write-fault every page now so the transpose never does.
        const long page = sysconf(_SC_PAGESIZE);
        for (uint64_t off = 0; off < size; off += (uint64_t)page) ((volatile uint8_t *)m)[off] = 0;
        p.dir = dir;
        p.size = size;
        p.map = m;
        prep = p;
    });
}

std::string dirOf(const std::string &path)
{
    const size_t k = path.find_last_of('/');
    return k == std::string::npos ? std::string(".") : path.substr(0, k);
}

}  // namespace

void mexFunction(int nlhs, mxArray *plhs[], int nrhs, const mxArray *prhs[])
{
    if (nrhs != 5)
        mexErrMsgIdAndTxt("opymLive:input",
                          "Usage: t = opymWriteLiveOutputs(dsr, npyPath, levelStores, mipStore, leadingIndex)");
    if (mxGetClassID(prhs[0]) != mxUINT16_CLASS || mxGetNumberOfDimensions(prhs[0]) != 3)
        mexErrMsgIdAndTxt("opymLive:input", "dsr must be a 3-D uint16 array (Y, X, Z)");
    const std::string npyPath = str(prhs[1], "npyPath");
    if (!mxIsCell(prhs[2]) || !mxGetNumberOfElements(prhs[2]))
        mexErrMsgIdAndTxt("opymLive:input", "levelStores must be a non-empty cell array of paths");
    std::vector<std::string> levelStores;
    for (size_t i = 0; i < mxGetNumberOfElements(prhs[2]); i++)
        levelStores.push_back(str(mxGetCell(prhs[2], i), "levelStores{i}"));
    const std::string mipStore = str(prhs[3], "mipStore");
    if (!mxIsDouble(prhs[4]) || !mxGetNumberOfElements(prhs[4]))
        mexErrMsgIdAndTxt("opymLive:input", "leadingIndex must be a non-empty double vector");
    std::vector<uint64_t> leading;
    for (size_t k = 0; k < mxGetNumberOfElements(prhs[4]); k++) {
        const double v = mxGetPr(prhs[4])[k];
        if (v < 1) mexErrMsgIdAndTxt("opymLive:input", "leadingIndex values are 1-based");
        leading.push_back((uint64_t)v - 1);
    }

    const mwSize *d = mxGetDimensions(prhs[0]);
    const uint64_t Y = d[0], X = d[1], Z = d[2];
    const uint16_t *dsr = (const uint16_t *)mxGetData(prhs[0]);

    // Every store's metadata up front: nothing is written unless all of it fits.
    std::vector<Block> levels;
    uint64_t shp[3] = {Z, Y, X};
    for (const std::string &s : levelStores) {
        levels.push_back(openBlock(s, leading, shp));
        shp[0] /= 2; shp[1] /= 2; shp[2] /= 2;
    }
    const uint64_t mipShape[3] = {1, Y, X};
    const Block mip = openBlock(mipStore, leading, mipShape);

    const auto t0 = Clock::now();

    // 1. The view buffer: transpose (Y, X, Z) col-major -> C-order (Z, Y, X)
    // plane by plane, in cache-sized tiles, into the memory map.
    const std::string header = npyHeader(Z, Y, X);
    const uint64_t nVox = Z * Y * X;
    const bool view = !npyPath.empty();
    const uint64_t fileSize = header.size() + nVox * sizeof(uint16_t);
    std::string tmp = npyPath + ".tmp";
    const std::string dir = dirOf(npyPath);
    void *map = nullptr;
    std::vector<uint16_t> mem;
    uint16_t *vol;
    joinPrep();
    if (view && prep.map && prep.dir == dir && prep.size == fileSize) {
        map = prep.map;  // readied after the last call: every page present
        tmp = prep.path;
        prep = Prepared();
        std::memcpy(map, header.data(), header.size());
        vol = (uint16_t *)((uint8_t *)map + header.size());
    } else if (view) {
        dropPrep();
        const int fd = open(tmp.c_str(), O_RDWR | O_CREAT | O_TRUNC, 0644);
        if (fd < 0) mexErrMsgIdAndTxt("opymLive:view", "Cannot write %s", tmp.c_str());
        if (ftruncate(fd, (off_t)fileSize) != 0) {
            close(fd);
            mexErrMsgIdAndTxt("opymLive:view", "Cannot size %s", tmp.c_str());
        }
        map = mmap(nullptr, fileSize, PROT_READ | PROT_WRITE, MAP_SHARED, fd, 0);
        close(fd);
        if (map == MAP_FAILED) mexErrMsgIdAndTxt("opymLive:view", "Cannot map %s", tmp.c_str());
        std::memcpy(map, header.data(), header.size());
        vol = (uint16_t *)((uint8_t *)map + header.size());
    } else {
        mem.resize(nVox);
        vol = mem.data();
    }
    const uint64_t T = 64;
    const int64_t nTy = (int64_t)((Y + T - 1) / T);
    #pragma omp parallel for collapse(2) schedule(static)
    for (int64_t z = 0; z < (int64_t)Z; z++) {
        for (int64_t ty = 0; ty < nTy; ty++) {
            const uint16_t *src = dsr + (uint64_t)z * Y * X;
            uint16_t *dst = vol + (uint64_t)z * Y * X;
            const uint64_t y0 = (uint64_t)ty * T, y1 = std::min(Y, y0 + T);
            for (uint64_t x0 = 0; x0 < X; x0 += T) {
                const uint64_t x1 = std::min(X, x0 + T);
                for (uint64_t y = y0; y < y1; y++)
                    for (uint64_t x = x0; x < x1; x++) dst[y * X + x] = src[y + x * Y];
            }
        }
    }
    if (view && rename(tmp.c_str(), npyPath.c_str()) != 0) {
        munmap(map, fileSize);
        unlink(tmp.c_str());
        mexErrMsgIdAndTxt("opymLive:view", "Cannot publish %s", npyPath.c_str());
    }
    const auto t1 = Clock::now();

    // 2. The pyramid, each level from the one before it.
    std::string err = writeBlock(levels[0], vol);
    std::vector<uint16_t> prev, next;
    const uint16_t *cur = vol;
    uint64_t cZ = Z, cY = Y, cX = X;
    for (size_t k = 1; k < levels.size() && err.empty(); k++) {
        const uint64_t oZ = cZ / 2, oY = cY / 2, oX = cX / 2;
        next.assign(oZ * oY * oX, 0);
        uint16_t *out = next.data();
        #pragma omp parallel for collapse(2) schedule(static)
        for (int64_t k2 = 0; k2 < (int64_t)oZ; k2++) {
            for (int64_t j = 0; j < (int64_t)oY; j++) {
                const uint16_t *a = cur + ((2 * k2) * cY + 2 * j) * cX;
                const uint16_t *b = a + cX;
                const uint16_t *c = a + cY * cX;
                const uint16_t *e = c + cX;
                uint16_t *o = out + (k2 * oY + j) * oX;
                for (uint64_t i = 0; i < oX; i++) {
                    const uint64_t x = 2 * i;
                    const uint32_t s = (uint32_t)a[x] + a[x + 1] + b[x] + b[x + 1] +
                                       c[x] + c[x + 1] + e[x] + e[x + 1];
                    o[i] = (uint16_t)(s >> 3);
                }
            }
        }
        err = writeBlock(levels[k], out);
        prev.swap(next);
        cur = prev.data();
        cZ = oZ; cY = oY; cX = oX;
    }

    // 3. The Z-MIP.
    if (err.empty()) {
        std::vector<uint16_t> m(Y * X, 0);
        #pragma omp parallel for schedule(static)
        for (int64_t y = 0; y < (int64_t)Y; y++) {
            uint16_t *row = m.data() + y * X;
            for (uint64_t z = 0; z < Z; z++) {
                const uint16_t *src = vol + (z * Y + y) * X;
                for (uint64_t x = 0; x < X; x++) row[x] = std::max(row[x], src[x]);
            }
        }
        err = writeBlock(mip, m.data());
    }
    const auto t2 = Clock::now();
    if (view) startPrep(dir, fileSize, map, fileSize);
    if (!err.empty()) mexErrMsgIdAndTxt("opymLive:write", "%s", err.c_str());

    plhs[0] = mxCreateDoubleMatrix(1, 2, mxREAL);
    mxGetPr(plhs[0])[0] = std::chrono::duration<double>(t1 - t0).count();
    mxGetPr(plhs[0])[1] = std::chrono::duration<double>(t2 - t1).count();
}
