function stats = run_live_zarr(p, numCPUs)
% RUN_LIVE_ZARR  One streamed (t, c) volume: raw OME-Zarr in, processed
% OME-Zarr out, everything in between in memory.
%
% The one-format live path (jobType 'live_zarr', opym.stream.live):
%
%   raw store (T, Z, Y, X)   --parallelReadZarr 'leadingIndex' [t+1],
%       'orientForDecon': exactly opym.utils.orient_zyx_for_decon_tiff + TIFF
%       paging, done by the reader-->  (ny, nx, nz)
%   --RLdecon, in memory ('rawdata' in, nothing written)-->  uint16 (ny, nx, nz)
%   --deskewRotateFrame3D, the call patches/XR_deskewRotateFrame.m makes-->  DSR (y, x, z)
%   --opymWriteLiveOutputs--> the view buffer, then the processed store:
%       3 pyramid levels of (T, C, Z, Y, X) + the Z-MIP at [t+1, c+1].
%
% Same functions with the same arguments as the TIFF live path
% (run_live_frames.m -> XR_RLdeconFrame3D -> RLdecon, then
% XR_deskewRotateFrame -> deskewRotateFrame3D), minus every intermediate
% file. tests/test_live_zarr_equivalence.py (GPU) checks that level 0 and
% the MIP are bit-identical to that path's DSR and MIP TIFFs.
%
% Only the configuration the live lane uses is supported (no objective or
% stage scan, no flat-field/background correction, no resampling, 16-bit
% output); anything else is an error rather than a silently different result.
%
% Ticket parameters (p):
%   raw_store       the channel's raw array (<...>.ome.zarr/p0)
%   t, c            0-based timepoint, and channel index in the processed store
%   mask_store      raw array of channel 0: the edge-erosion mask is built
%                   from its t = 0, as the wrappers' erodeByFTP does
%   levels          processed arrays, full resolution first (<store>/0/0, ...)
%   mip             processed Z-MIP array (<store>/1/0)
%   view_npy        optional: where to put the full-resolution volume for the
%                   live viewer first, as an uncompressed C-order (Z, Y, X)
%                   uint16 .npy on the RAM disk (numpy memory-maps it: no
%                   decode). It lands before the compressed store is written,
%                   so the view never waits on encoding.
%   psf_path, decon_dir (the session's work dir), plus a 'live' ticket's
%                   decon and DSR fields
%   psf_cache_dir   optional: where the generated PSF and OMW back projector
%                   live (<dir>/psfgen), shared by every acquisition with the
%                   same PSF, settings and shape (default decon_dir)
%   warmup          optional: true for the warm-up a server runs when a
%                   session starts (see opym.stream.live_zarr.WARMUP_NAME) --
%                   a decon + DSR of a constant volume of raw_shape_zyx,
%                   discarded, and the first view buffer readied in view_dir.
%                   Nothing is read or written besides the PSF cache.

t = double(p.t);
c = double(p.c);
deconDir = char(p.decon_dir);
psf = char(p.psf_path);
levels = cellstr(p.levels);

xy     = getp(p, 'xy_pixel_size', 0.136);
dz     = getp(p, 'z_step_um', 0.3);
ang    = getp(p, 'sheet_angle_deg', 60.0);
method = lower(char(getp(p, 'rl_method', 'omw')));
if strcmp(method, 'omw'), defIter = 2; else, defIter = 25; end
iter   = getp(p, 'decon_iter', defIter);
bg     = getp(p, 'background', []);
if isempty(bg), bg = 100; end            % XR_RLdeconFrame3D's default
erode  = getp(p, 'edge_erosion', 0);
gpu    = getp(p, 'gpu_decon', false);
alpha  = getp(p, 'wiener_alpha', 0.005);
otfCT  = getp(p, 'otf_cum_thresh', 0.9);
hann   = getp(p, 'hann_win_bounds', [0.8, 1.0]);
damp   = getp(p, 'damp_factor', 1);
dzPSF  = 0.1;                            % the wrapper's default; the 'deskew' branch passes none
interp = char(getp(p, 'interp_method', 'linear'));
reverse = getp(p, 'reverse', false);
if getp(p, 'objective_scan', false) || getp(p, 'z_stage_scan', false)
    error('run_live_zarr:unsupported', 'Objective/stage-scan data is not supported by the live zarr path.');
end

if ~exist(deconDir, 'dir'), mkdir(deconDir); end
stats = struct('frames', 1, 'read_s', 0, 'decon_s', 0, 'dsr_s', 0, 'view_s', 0, 'write_s', 0, ...
    'buffer_at', NaN, 'store_at', NaN);
cacheRoot = char(getp(p, 'psf_cache_dir', deconDir));

% --- once per PSF cache: generated PSF, exactly as XR_decon_data_wrapper ---
% (same code as run_live_frames.m; RLdecon loads it from <its output dir>/psfgen)
psfgenDir = fullfile(cacheRoot, 'psfgen');
if ~exist(psfgenDir, 'dir'), mkdir(psfgenDir); end
[~, psfFsn] = fileparts(psf);
psfgenFn = sprintf('%s/%s_%s.tif', psfgenDir, psfFsn, method);
if ~exist(psfgenFn, 'file')
    psfIm = single(readtiff(psf));
    psfIm = psf_gen_new(psfIm, dzPSF, dz, 1.5, 'masked');
    if ~strcmp(method, 'omw')
        py = find(squeeze(sum(psfIm, [2, 3])));
        px = find(squeeze(sum(psfIm, [1, 3])));
        pz = find(squeeze(sum(psfIm, [1, 2])));
        cropSz = [min(py(1) - 1, size(psfIm, 1) - py(end)), min(px(1) - 1, size(psfIm, 2) - px(end)), ...
            min(pz(1) - 1, size(psfIm, 3) - pz(end))] - 1;
        cropSz = max(0, cropSz);
        bbox = [cropSz + 1, size(psfIm, 1 : 3) - cropSz];
        psfIm = psfIm(bbox(1) : bbox(4), bbox(2) : bbox(5), bbox(3) : bbox(6));
    end
    tmpFn = sprintf('%s/%s_%s.tif', psfgenDir, psfFsn, get_uuid());
    writetiff(psfIm, tmpFn);
    publishOnce(tmpFn, psfgenFn, false);
end

% RLdecon exactly as XR_RLdeconFrame3D calls it (its defaults for what
% run_live_frames.m doesn't pass: Reverse true, fixIter false, mipAxis
% [0 0 1]), with the volume in memory and nothing written: save3Dstack all
% false returns the result, and the output name only places the PSF cache.
% The erosion mask is applied by this job rather than by RLdecon: given a
% mask file it runs the core with EdgeErosion 0, then reads the file and
% multiplies by it on every call -- EdgeErosion 0 with no file is that same
% core call, and zeroing the mask's voxels is that same product.
rlArgs = {'save16bit', true, 'SkewAngle', ang, 'Deskew', false, 'Rotate', false, 'DSRCombined', true, ...
    'Reverse', true, 'Background', bg, 'DeconIter', iter, 'RLMethod', method, 'skewed', true, ...
    'wienerAlpha', alpha, 'OTFCumThresh', otfCT, 'hannWinBounds', hann, 'saveZarr', false, ...
    'blockSize', [256, 256, 256], 'fixIter', false, 'dampFactor', damp, 'scaleFactor', 1, ...
    'deconOffset', 0, 'EdgeErosion', 0, 'ErodeMaskfile', '', 'errThresh', [], ...
    'saveStep', 5, 'useGPU', gpu, 'psfGen', true, 'debug', false, ...
    'save3Dstack', [false, false, false], 'mipAxis', [0, 0, 1]};
% The DSRCombined call patches/XR_deskewRotateFrame.m makes for this
% configuration (no stage scan, so skewAngle_1 = skewAngle).
dsrArgs = {'reverse', reverse, 'crop', true, 'objectiveScan', false, ...
    'resampleFactor', [], 'interpMethod', interp, 'xStepThresh', 2.0, 'save16bit', true};

% --- warm-up: the same decon and DSR on a constant volume of the session's
% shape, discarded. Compiles the code paths, builds the back projector into
% the cache, plans the FFTs and leaves the OTF on the GPU
% (decon_lucy_omw_function keys it on size and back projector, so the first
% real ticket reuses it), then readies the first view buffer. ---
if getp(p, 'warmup', false)
    t0 = tic;
    rs = double(p.raw_shape_zyx(:)');
    dummy = repmat(uint16(bg + 1), rs([3, 2, 1]));   % (X, Y, Z), as the read returns
    d = uint16(RLdecon('', fullfile(cacheRoot, 'warmup.tif'), psf, xy, dz, dzPSF, ...
        'rawdata', dummy, rlArgs{:}));
    clear dummy;
    d = deskewRotateFrame3D(d, ang, dz, xy, dsrArgs{:});
    if isfield(p, 'view_dir') && ~isempty(p.view_dir)
        opymWriteLiveOutputs('prepare', char(p.view_dir), double(size(d, 1 : 3)));
    end
    stats.decon_s = toc(t0);
    return;
end

% --- once per session and server: the first-time-point erosion mask
% (erodeByFTP), kept in memory as the voxels it zeroes ---
zeroIdx = [];
if erode > 0
    zeroIdx = ftpMaskZeros(char(p.mask_store), erode, deconDir);
end

% --- read, straight into the orientation the decon takes ---
t0 = tic;
raw = parallelReadZarr(char(p.raw_store), 'leadingIndex', t + 1, 'orientForDecon', true);
stats.read_s = toc(t0);

% --- decon ---
t0 = tic;
fsname = sprintf('live_T%04d_C%d', t, c);
deconvolved = RLdecon('', fullfile(cacheRoot, [fsname '.tif']), psf, xy, dz, dzPSF, ...
    'rawdata', raw, rlArgs{:});
clear raw;
% What the TIFF path writes and XR_deskewRotateFrame reads back (RLdecon's
% save branch does this cast; the early return above skips it).
deconvolved = uint16(deconvolved);
deconvolved(zeroIdx) = 0;
stats.decon_s = toc(t0);

% --- deskew/rotate ---
t0 = tic;
dsr = deskewRotateFrame3D(deconvolved, ang, dz, xy, dsrArgs{:});
clear deconvolved;
dsr = uint16(dsr);                        % the wrapper's writetiff(uint16(dsr))
stats.dsr_s = toc(t0);

% --- every output from one transpose (opymWriteLiveOutputs): the live
% viewer's buffer first, published before anything is encoded, then every
% pyramid level and the Z-MIP (the wrapper's MIP: max over z) into the
% processed store at [t+1, c+1] ---
viewNpy = '';
if isfield(p, 'view_npy') && ~isempty(p.view_npy)
    viewNpy = char(p.view_npy);
end
% (Absolute times too, for opym-live-trace: the view buffer is what the
% viewer shows, so its landing time is where the viewer's hops start.)
tCall = posixtime(datetime('now', 'TimeZone', 'UTC'));
tt = opymWriteLiveOutputs(dsr, viewNpy, levels, char(p.mip), [t + 1, c + 1]);
stats.view_s = tt(1);
stats.write_s = tt(2);
stats.buffer_at = tCall + tt(1);
stats.store_at = tCall + tt(1) + tt(2);
end


function zeroIdx = ftpMaskZeros(maskStore, erode, sessionDir)
% Linear indices (into the decon's (X, Y, Z) array) of the voxels the
% first-time-point erosion mask zeroes: decon_mask_edge_erosion of channel
% 0's first time point, as the wrappers' erodeByFTP builds it. Computed on
% this server's first ticket of a session and kept for the rest of it
% (keyed on the session's own work dir too: a later session can reuse a
% raw store's path).
persistent key idx
k = sprintf('%s|%d|%s', maskStore, erode, sessionDir);
if ~isequal(key, k)
    first = parallelReadZarr(maskStore, 'leadingIndex', 1, 'orientForDecon', true);
    idx = uint32(find(~decon_mask_edge_erosion(first > 0, erode)));
    key = k;
end
zeroIdx = idx;
end


function publishOnce(tmpPath, finalPath, isDir)
% Rename a freshly built per-session artifact into place; whoever renames
% second finds it already there, which is success (see run_live_frames.m).
if isDir
    present = @() exist(finalPath, 'dir') == 7;
else
    present = @() exist(finalPath, 'file') == 2;
end
if present()
    discard(tmpPath, isDir);
    return;
end
% One rename(2): a file lands atomically, replacing an identical one (never
% a moment without it, as movefile's delete-then-move could leave); a
% directory can't replace one that's already there, which is fine too.
if ~java.io.File(tmpPath).renameTo(java.io.File(finalPath))
    if ~present()
        error('run_live_zarr:publish', 'Cannot move %s to %s', tmpPath, finalPath);
    end
    discard(tmpPath, isDir);
end
end


function discard(path, isDir)
if isDir
    if exist(path, 'dir'), rmdir(path, 's'); end
elseif exist(path, 'file')
    delete(path);
end
end


function v = getp(s, name, default)
if isfield(s, name) && ~isempty(s.(name))
    v = s.(name);
else
    v = default;
end
end
