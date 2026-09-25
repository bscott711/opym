// opymWriteZarrBlock(store, data, leadingIndex)
//
// Writes `data` as the whole trailing 3-D block of an EXISTING N-D zarr
// array at the 1-based `leadingIndex` -- e.g. [t+1, c+1] for the live
// pipeline's (T, C, Z, Y, X) processed store, so a deconvolved + deskewed
// volume goes from MATLAB straight into the OME-Zarr napari reads.
//
// The .zarray is read, never written: array metadata belongs to whoever
// created the store (opym, in Python). size(data) must equal the array's
// trailing shape and its class the array's dtype -- nothing is converted.
// A MATLAB array of size [Z Y X] lands as zarr axes (..., Z, Y, X).
//
// Chunks are compressed with the array's own blosc/gzip settings by
// PetaKit5D's parallelWriteZarr (OpenMP, one chunk per task), each written
// under a temporary name and renamed into place, so a reader never sees a
// partial chunk. All-zero chunks are not written at all; zarr reads a
// missing chunk as fill_value, which the store's creator sets to 0.
#include <cstdint>
#include <cstring>
#include <set>
#include <string>
#include <vector>
#include "mex.h"
#include "../src/helperfunctions.h"
#include "../src/parallelwritezarr.h"
#include "../src/zarr.h"

void mexFunction(int nlhs, mxArray *plhs[], int nrhs, const mxArray *prhs[])
{
    if(nrhs != 3){
        mexErrMsgIdAndTxt("opymZarr:inputError",
                          "Usage: opymWriteZarrBlock(store, data, leadingIndex)");
    }
    if(!mxIsChar(prhs[0])) mexErrMsgIdAndTxt("opymZarr:inputError","store must be a string");
    if(!mxIsDouble(prhs[2]) || !mxGetNumberOfElements(prhs[2])){
        mexErrMsgIdAndTxt("opymZarr:inputError","leadingIndex must be a non-empty double vector");
    }

    std::string folderName(mxArrayToString(prhs[0]));
    folderName = expandTilde(folderName.c_str());
    zarr Zarr;
    try{
        Zarr = zarr(folderName);
    }
    catch(const std::string &e){
        mexErrMsgIdAndTxt("opymZarr:zarrayError","Cannot open %s (%s)",folderName.c_str(),e.c_str());
    }

    std::vector<uint64_t> leadingIndex;
    for(uint64_t k = 0; k < mxGetNumberOfElements(prhs[2]); k++){
        const double v = *(mxGetPr(prhs[2])+k);
        if(v < 1) mexErrMsgIdAndTxt("opymZarr:inputError","leadingIndex values are 1-based");
        leadingIndex.push_back((uint64_t)v - 1);
    }
    try{
        Zarr.set_leadingIndex(leadingIndex);
    }
    catch(const std::string &e){
        mexErrMsgIdAndTxt("opymZarr:leadingIndex","Cannot index %s that way (%s): it needs a C-order array with '/' separators, chunk size 1 on every leading axis, and indices in range.",folderName.c_str(),e.c_str());
    }
    if(Zarr.get_cname() == "raw"){
        mexErrMsgIdAndTxt("opymZarr:inputError","%s is uncompressed; opymWriteZarrBlock writes blosc or gzip arrays",folderName.c_str());
    }

    std::string dtype;
    uint64_t bits = 0;
    switch(mxGetClassID(prhs[1])){
        case mxUINT8_CLASS: dtype = "<u1"; bits = 8; break;
        case mxUINT16_CLASS: dtype = "<u2"; bits = 16; break;
        case mxSINGLE_CLASS: dtype = "<f4"; bits = 32; break;
        case mxDOUBLE_CLASS: dtype = "<f8"; bits = 64; break;
        default: mexErrMsgIdAndTxt("opymZarr:inputError","data must be uint8, uint16, single or double");
    }
    if(dtype != Zarr.get_dtype()){
        mexErrMsgIdAndTxt("opymZarr:inputError","data is %s but %s holds %s",dtype.c_str(),folderName.c_str(),Zarr.get_dtype().c_str());
    }
    const uint64_t nd = mxGetNumberOfDimensions(prhs[1]);
    if(nd > 3) mexErrMsgIdAndTxt("opymZarr:inputError","data must be 2-D or 3-D");
    const mwSize* dims = mxGetDimensions(prhs[1]);
    uint64_t d[3] = {1,1,1};
    for(uint64_t i = 0; i < nd; i++) d[i] = dims[i];
    for(uint64_t i = 0; i < 3; i++){
        if(d[i] != Zarr.get_shape(i)){
            mexErrMsgIdAndTxt("opymZarr:inputError",
                              "size(data) is [%llu %llu %llu] but the block is [%llu %llu %llu]",
                              (unsigned long long)d[0],(unsigned long long)d[1],(unsigned long long)d[2],
                              (unsigned long long)Zarr.get_shape(0),(unsigned long long)Zarr.get_shape(1),
                              (unsigned long long)Zarr.get_shape(2));
        }
    }

    const std::vector<uint64_t> start = {0,0,0};
    const std::vector<uint64_t> end = {Zarr.get_shape(0),Zarr.get_shape(1),Zarr.get_shape(2)};
    Zarr.set_chunkInfo(start, end);

    // "/"-separated chunk keys are nested directories; PetaKit creates them
    // only when it writes the .zarray itself, which this never does. A few
    // dozen unique parents, so serially (no mkdir races between threads).
    std::set<std::string> parents;
    for(uint64_t i = 0; i < Zarr.get_numChunks(); i++){
        const std::string path = Zarr.get_fileName()+"/"+Zarr.get_chunkNames(i);
        parents.insert(path.substr(0, path.find_last_of("/")));
    }
    for(const std::string &p : parents) mkdirRecursive(p.c_str());

    const uint8_t err = parallelWriteZarr(Zarr, mxGetData(prhs[1]), start, end, end,
                                          bits, true, false, true);
    if(err) mexErrMsgIdAndTxt("opymZarr:writeError","%s",Zarr.get_errString().c_str());
}
