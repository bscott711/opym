function stats = run_live_frames(p, numCPUs)
% RUN_LIVE_FRAMES  Decon -> deskew/rotate a list of frames, one at a time.
%
% The live lane's job (jobType 'live', see run_petakit_server.m). It calls
% the SAME per-frame functions the batch wrappers call for each file, with
% the SAME arguments, and skips only the wrapper machinery around them
% (directory scan and filename parsing, parameters.mat/json, the job
% framework, parfor dispatch):
%
%   XR_decon_data_wrapper         -> XR_RLdeconFrame3D   (one call per frame)
%   XR_deskew_rotate_data_wrapper -> XR_deskewRotateFrame (one call per frame)
%
% Arguments below mirror the func_str each wrapper builds per file
% (XR_decon_data_wrapper.m ~l.427, XR_deskew_rotate_data_wrapper.m ~l.347),
% with the values the wrappers derive from what run_petakit_server.m's
% 'deskew' branch passes them. Output must match the batch path pixel for
% pixel; tests/test_live_equivalence (GPU) checks it.
%
% Per-session work happens once, not per frame or per ticket, because every
% ticket of a session shares one decon_dir:
%   - the generated PSF (<decon_dir>/psfgen/<psf>_<method>.tif) is built
%     here exactly as XR_decon_data_wrapper builds it, and RLdecon reuses it;
%   - RLdecon caches the OMW back projector beside it;
%   - the edge-erosion mask comes from the session's first C0 frame, as the
%     wrapper's erodeByFTP does, and is reused for every frame.
%
% Ticket parameters (p): frames (cell of staged TIFF paths), channel_patterns,
% psf_paths (one per pattern), decon_dir, erode_mask_source, plus the decon
% and DSR parameters of a 'deskew' ticket. keep_decon (default false) keeps
% the intermediate decon TIFF; the live lane only needs the DSR result.

frames = cellstr(p.frames);
chans  = cellstr(p.channel_patterns);
psfs   = cellstr(p.psf_paths);
if numel(psfs) ~= numel(chans)
    error('run_live_frames:psfChannelMismatch', ...
        'psf_paths has %d entries but channel_patterns has %d.', numel(psfs), numel(chans));
end
deconDir = char(p.decon_dir);

xy     = getp(p, 'xy_pixel_size', 0.136);
dz     = getp(p, 'z_step_um', 0.3);
ang    = getp(p, 'sheet_angle_deg', 60.0);
method = lower(char(getp(p, 'rl_method', 'omw')));
if strcmp(method, 'omw'), defIter = 2; else, defIter = 25; end
iter   = getp(p, 'decon_iter', defIter);
bg     = getp(p, 'background', []);
if isempty(bg), bg = 100; end            % XR_decon_data_wrapper's default
erode  = getp(p, 'edge_erosion', 0);
gpu    = getp(p, 'gpu_decon', false);
alpha  = getp(p, 'wiener_alpha', 0.005);
otfCT  = getp(p, 'otf_cum_thresh', 0.9);
hann   = getp(p, 'hann_win_bounds', [0.8, 1.0]);
damp   = getp(p, 'damp_factor', 1);
dzPSF  = 0.1;                            % the wrapper's default; the 'deskew' branch passes none
dsDir  = char(getp(p, 'ds_dir_name', 'DS'));
dsrDir = char(getp(p, 'dsr_dir_name', 'DSR'));
interp = char(getp(p, 'interp_method', 'linear'));   % opym.decon_config.DSR_INTERP_METHOD
reverse = getp(p, 'reverse', false);
objScan = getp(p, 'objective_scan', false);
zStage  = getp(p, 'z_stage_scan', false);
keepDecon = getp(p, 'keep_decon', false);

if ~exist(deconDir, 'dir'), mkdir(deconDir); end

% --- once per session: generated PSFs, exactly as XR_decon_data_wrapper ---
psfgenDir = fullfile(deconDir, 'psfgen');
if ~exist(psfgenDir, 'dir'), mkdir(psfgenDir); end
for k = 1:numel(psfs)
    [~, psfFsn] = fileparts(psfs{k});
    psfgenFn = sprintf('%s/%s_%s.tif', psfgenDir, psfFsn, method);
    if exist(psfgenFn, 'file'), continue; end
    psf = single(readtiff(psfs{k}));
    psf = psf_gen_new(psf, dzPSF, dz, 1.5, 'masked');
    if ~strcmp(method, 'omw')
        py = find(squeeze(sum(psf, [2, 3])));
        px = find(squeeze(sum(psf, [1, 3])));
        pz = find(squeeze(sum(psf, [1, 2])));
        cropSz = [min(py(1) - 1, size(psf, 1) - py(end)), min(px(1) - 1, size(psf, 2) - px(end)), ...
            min(pz(1) - 1, size(psf, 3) - pz(end))] - 1;
        cropSz = max(0, cropSz);
        bbox = [cropSz + 1, size(psf, 1 : 3) - cropSz];
        psf = psf(bbox(1) : bbox(4), bbox(2) : bbox(5), bbox(3) : bbox(6));
    end
    tmpFn = sprintf('%s/%s_%s.tif', psfgenDir, psfFsn, get_uuid());
    writetiff(psf, tmpFn);
    publishOnce(tmpFn, psfgenFn, false);
end

% --- once per session: the first-time-point erosion mask (erodeByFTP) ---
maskFn = '';
if erode > 0
    src = char(p.erode_mask_source);
    [~, srcFsn] = fileparts(src);
    maskDir = fullfile(deconDir, 'Masks');
    if ~exist(maskDir, 'dir'), mkdir(maskDir); end
    maskFn = sprintf('%s/%s_eroded.zarr', maskDir, srcFsn);
    if ~exist(maskFn, 'dir')
        im_bw_erode = decon_mask_edge_erosion(readtiff(src) > 0, erode);
        tmpMask = sprintf('%s/%s_eroded_%s.zarr', maskDir, srcFsn, get_uuid());
        writezarr(uint8(im_bw_erode), tmpMask, 'blockSize', [256, 256, 256]);
        publishOnce(tmpMask, maskFn, true);
    end
end

% --- per frame ---
nF = numel(frames);
stats = struct('frames', nF, 'decon_s', 0, 'dsr_s', 0);
for f = 1:nF
    frame = frames{f};
    [~, fsname] = fileparts(frame);
    k = find(~cellfun(@isempty, regexpi(frame, chans)), 1);   % the wrappers' psfMapping
    if isempty(k)
        error('run_live_frames:noChannel', 'No channel pattern matches %s', frame);
    end

    t0 = tic;
    XR_RLdeconFrame3D(frame, xy, dz, deconDir, 'PSFfile', psfs{k}, ...
        'dzPSF', dzPSF, 'background', bg, 'skewAngle', ang, 'flipZstack', false, ...
        'edgeErosion', erode, 'ErodeMaskfile', maskFn, 'SaveMaskfile', false, 'Rotate', false, ...
        'deconIter', iter, 'RLMethod', method, 'wienerAlpha', alpha, 'OTFCumThresh', otfCT, ...
        'hannWinBounds', hann, 'skewed', true, 'debug', false, 'saveStep', 5, 'psfGen', true, ...
        'saveZarr', false, 'parseCluster', false, 'parseParfor', true, 'GPUJob', gpu, ...
        'save16bit', true, 'largeFile', false, 'largeMethod', 'inmemory', ...
        'batchSize', [1024, 1024, 1024], 'blockSize', [256, 256, 256], 'dampFactor', damp, ...
        'scaleFactor', 1, 'deconOffset', 0, 'maskFullpaths', {}, 'uuid', get_uuid(), ...
        'cpusPerTask', numCPUs, 'mccMode', false, 'configFile', '', 'GPUConfigFile', '');
    stats.decon_s = stats.decon_s + toc(t0);

    deconFrame = sprintf('%s/%s.tif', deconDir, fsname);
    if ~exist(deconFrame, 'file')
        error('run_live_frames:noDecon', 'Decon wrote no output for %s', frame);
    end

    t0 = tic;
    XR_deskewRotateFrame({deconFrame}, xy, dz, 'DSDirName', dsDir, 'DSRDirName', dsrDir, ...
        'skewAngle', ang, 'objectiveScan', objScan, 'zStageScan', zStage, 'reverse', reverse, ...
        'inputAxisOrder', 'yxz', 'outputAxisOrder', 'yxz', 'FFCorrection', false, ...
        'BKRemoval', false, 'lowerLimit', 0.4, 'constOffset', [], 'FFImage', '', ...
        'BackgroundImage', '', 'rotate', true, 'resampleFactor', [], 'DSRCombined', true, ...
        'inputBbox', [], 'flipZstack', false, 'save16bit', true, 'save3DStack', true, ...
        'saveZarr', false, 'interpMethod', interp);
    stats.dsr_s = stats.dsr_s + toc(t0);

    if ~keepDecon
        delete(deconFrame);
    end
end
end

function publishOnce(tmpPath, finalPath, isDir)
% Rename a freshly built per-session artifact (generated PSF, erosion mask)
% into place. Both GPU servers can build the same one at once, on a
% session's first tickets. Whoever renames second finds it already there,
% which is success: identical content. Before this, a check-then-rename
% left a window where the second rename failed with "already exists" and
% failed its live ticket (2026-09-24, Cell_006).
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
