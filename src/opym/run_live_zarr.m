function stats = run_live_zarr(p, numCPUs)
% RUN_LIVE_ZARR  One streamed (t, c) volume: raw OME-Zarr in, processed
% OME-Zarr out, everything in between in memory.
%
% The one-format live path (jobType 'live_zarr', opym.stream.live):
%
%   raw store (T, Z, Y, X)   --parallelReadZarr 'leadingIndex' [t+1]-->  (Z, Y, X)
%   --orient: exactly opym.utils.orient_zyx_for_decon_tiff + TIFF paging--> (ny, nx, nz)
%   --RLdecon, in memory ('rawdata' in, nothing written)-->  uint16 (ny, nx, nz)
%   --deskewRotateFrame3D, the call patches/XR_deskewRotateFrame.m makes-->  DSR (y, x, z)
%   --> processed store: 3 pyramid levels of (T, C, Z, Y, X) + the Z-MIP,
%       each through opymWriteZarrBlock at [t+1, c+1].
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
%   psf_path, decon_dir (per-session cache: generated PSF, OMW back projector,
%                   erosion mask), plus a 'live' ticket's decon and DSR fields.

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
stats = struct('frames', 1, 'read_s', 0, 'decon_s', 0, 'dsr_s', 0, 'view_s', 0, 'write_s', 0);

% --- once per session: generated PSF, exactly as XR_decon_data_wrapper ---
% (same code and cache location as run_live_frames.m; RLdecon loads it)
psfgenDir = fullfile(deconDir, 'psfgen');
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

% --- once per session: the first-time-point erosion mask (erodeByFTP) ---
maskFn = '';
if erode > 0
    maskDir = fullfile(deconDir, 'Masks');
    if ~exist(maskDir, 'dir'), mkdir(maskDir); end
    maskFn = sprintf('%s/first_timepoint_C0_eroded.zarr', maskDir);
    if ~exist(maskFn, 'dir')
        first = orientForDecon(parallelReadZarr(char(p.mask_store), 'leadingIndex', 1));
        im_bw_erode = decon_mask_edge_erosion(first > 0, erode);
        tmpMask = sprintf('%s/first_timepoint_C0_eroded_%s.zarr', maskDir, get_uuid());
        writezarr(uint8(im_bw_erode), tmpMask, 'blockSize', [256, 256, 256]);
        publishOnce(tmpMask, maskFn, true);
    end
end

% --- read ---
t0 = tic;
raw = orientForDecon(parallelReadZarr(char(p.raw_store), 'leadingIndex', t + 1));
stats.read_s = toc(t0);

% --- decon: RLdecon exactly as XR_RLdeconFrame3D calls it (its defaults for
% what run_live_frames.m doesn't pass: Reverse true, fixIter false, mipAxis
% [0 0 1]), with the volume in memory and nothing written. The output name
% only places RLdecon's PSF cache (deconDir/psfgen); save3Dstack all false
% returns the uint16 result instead of writing it. ---
t0 = tic;
fsname = sprintf('live_T%04d_C%d', t, c);
deconvolved = RLdecon('', fullfile(deconDir, [fsname '.tif']), psf, xy, dz, dzPSF, 'rawdata', raw, ...
    'save16bit', true, 'SkewAngle', ang, 'Deskew', false, 'Rotate', false, 'DSRCombined', true, ...
    'Reverse', true, 'Background', bg, 'DeconIter', iter, 'RLMethod', method, 'skewed', true, ...
    'wienerAlpha', alpha, 'OTFCumThresh', otfCT, 'hannWinBounds', hann, 'saveZarr', false, ...
    'blockSize', [256, 256, 256], 'fixIter', false, 'dampFactor', damp, 'scaleFactor', 1, ...
    'deconOffset', 0, 'EdgeErosion', erode, 'ErodeMaskfile', maskFn, 'errThresh', [], ...
    'saveStep', 5, 'useGPU', gpu, 'psfGen', true, 'debug', false, ...
    'save3Dstack', [false, false, false], 'mipAxis', [0, 0, 1]);
clear raw;
% What the TIFF path writes and XR_deskewRotateFrame reads back (RLdecon's
% save branch does this cast; the early return above skips it).
deconvolved = uint16(deconvolved);
stats.decon_s = toc(t0);

% --- deskew/rotate: the DSRCombined call patches/XR_deskewRotateFrame.m
% makes for this configuration (no stage scan, so skewAngle_1 = skewAngle). ---
t0 = tic;
dsr = deskewRotateFrame3D(deconvolved, ang, dz, xy, ...
    'reverse', reverse, 'crop', true, 'objectiveScan', false, ...
    'resampleFactor', [], 'interpMethod', interp, ...
    'xStepThresh', 2.0, 'save16bit', true);
clear deconvolved;
dsr = uint16(dsr);                        % the wrapper's writetiff(uint16(dsr))
mip = uint16(max(dsr, [], 3));            % the wrapper's MIP: max over z, uint16
stats.dsr_s = toc(t0);

% --- the live viewer's copy first: nothing between it and the screen ---
if isfield(p, 'view_npy') && ~isempty(p.view_npy)
    t0 = tic;
    writeNpyZYX(char(p.view_npy), dsr);
    stats.view_s = toc(t0);
end

% --- write: (y, x, z) -> (z, y, x), the axis order the TIFF pages had ---
t0 = tic;
vol = permute(dsr, [3, 1, 2]);
clear dsr;
leading = [t + 1, c + 1];
opymWriteZarrBlock(levels{1}, vol, leading);
for k = 2 : numel(levels)
    vol = downsample2(vol);
    opymWriteZarrBlock(levels{k}, vol, leading);
end
opymWriteZarrBlock(char(p.mip), reshape(mip, [1, size(mip)]), leading);
stats.write_s = toc(t0);
end


function writeNpyZYX(path, dsr)
% dsr (y, x, z) as deskewRotateFrame3D returns it -> a C-order (Z, Y, X)
% uint16 .npy (format 1.0). Written under a temporary name and renamed, so
% a reader never maps a partial buffer.
v = permute(dsr, [2, 1, 3]);   % (x, y, z) column-major == (z, y, x) C order
sz = size(dsr, 1 : 3);         % [Y X Z]
hdr = sprintf('{''descr'': ''<u2'', ''fortran_order'': False, ''shape'': (%d, %d, %d), }', ...
    sz(3), sz(1), sz(2));
pad = mod(-(10 + numel(hdr) + 1), 64);   % the whole header a multiple of 64
hdr = [hdr, repmat(' ', 1, pad), newline];
tmp = [path, '.tmp'];
fid = fopen(tmp, 'w');
if fid < 0
    error('run_live_zarr:viewBuffer', 'Cannot write %s', tmp);
end
fwrite(fid, [uint8(147), uint8('NUMPY'), uint8(1), uint8(0)], 'uint8');
fwrite(fid, numel(hdr), 'uint16', 0, 'l');
fwrite(fid, hdr, 'char');
fwrite(fid, v, 'uint16', 0, 'l');
fclose(fid);
movefile(tmp, path);
end


function a = orientForDecon(zyx)
% (Z, Y, X) as read from the raw store -> the (ny, nx, nz) array MATLAB's
% readtiff returns for a staged TIFF written by
% opym.utils.orient_zyx_for_decon_tiff: np.rot90(v, 1, axes=(1, 2)), whose
% pages readtiff stacks along dim 3. Element for element:
% a(i, j, z) = v(z, j, X + 1 - i). An exact index permutation.
a = flip(permute(zyx, [3, 2, 1]), 1);
end


function out = downsample2(v)
% 2x block mean on each axis, dropping an odd trailing plane/row/column,
% floored back to uint16 -- bit-identical to opym.ome_zarr_writer.downsample2
% (8-voxel sum in uint32, floor-divided by 8). On the GPU when there is one.
sz = size(v, 1 : 3);
sz = sz - mod(sz, 2);
if gpuDeviceCount > 0
    v = gpuArray(v);
end
v = uint32(v(1 : sz(1), 1 : sz(2), 1 : sz(3)));
s = v(1:2:end, 1:2:end, 1:2:end) + v(2:2:end, 1:2:end, 1:2:end) ...
  + v(1:2:end, 2:2:end, 1:2:end) + v(2:2:end, 2:2:end, 1:2:end) ...
  + v(1:2:end, 1:2:end, 2:2:end) + v(2:2:end, 1:2:end, 2:2:end) ...
  + v(1:2:end, 2:2:end, 2:2:end) + v(2:2:end, 2:2:end, 2:2:end);
out = gather(uint16(bitshift(s, -3)));
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
try
    movefile(tmpPath, finalPath);
catch ME
    if ~present()
        rethrow(ME);
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
