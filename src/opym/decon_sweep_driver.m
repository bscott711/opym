function decon_sweep_driver(variantsJsonPath)
%DECON_SWEEP_DRIVER Out-of-band low-SNR decon parameter sweep.
%
% Runs a list of decon/DSR variants against ONE already-staged timepoint,
% outside the /dev/shm/petakit_jobs queue so the production backfill and
% its two live servers are untouched. Mirrors, verbatim in argument list,
% the decon + deskew/rotate calls run_petakit_server.m makes for a
% 'deskew' job that also runs decon (see that file, the branch guarded by
% `otherwise` in its jobType switch, decon call ~line 617 and the
% XR_deskew_rotate_data_wrapper call ~line 682) -- this driver is not a
% new code path, it is the same two PetaKit5D calls with a JSON variant
% list standing in for one ticket's `parameters`.
%
% variantsJsonPath : path to a JSON file with fields:
%   stage_dir, psf_path, channel_patterns, xy_pixel_size, z_step_um,
%   dz_psf, skew_angle_deg, edge_erosion, variants[]
% Each variants[] entry: name, decon (bool), rl_method, iters, alpha,
%   otf_thresh, hann ([lo,hi] or null), background, damp.
% Optional per-variant overrides (decon v2 trials; absent = the top-level value):
%   psf_paths   cell, one PSF per channel pattern (else psf_path is broadcast)
%   stage_dir   a different staged input (e.g. a denoised copy of the TIFFs)
%   edge_erosion  integer
%   interp_method 'linear' | 'cubic' (top-level interp_method, default 'cubic')
%   pad_z       integer P: mirror-pad the scan axis (dim 3) by P planes before
%               decon and crop back to the original planes before DSR, so the
%               axial faces are not wrapped by the FFT and the DSR input shape
%               equals production's. edge_erosion then applies to the padded
%               faces, which are cropped away.
%
% Output, per variant, under stage_dir/../:
%   Decon_<name>/                 (skewed decon output, psfgen/ QC figs)
%   Decon_<name>/DSR/             (final lab-frame volume)
%   nodecon/DS, nodecon/DSR       (DSR run straight on the staged TIFFs)

addpath(fileparts(mfilename('fullpath')));

petakit_source_path = getenv('PETAKIT_ROOT');
if isempty(petakit_source_path)
    petakit_source_path = '/cm/shared/apps_local/petakit5d';
end
fprintf('[Sweep] PetaKit path: %s\n', petakit_source_path);
addpath(genpath(petakit_source_path));

envCPUs = getenv('PETAKIT_CPUS');
if ~isempty(envCPUs)
    numCPUs = str2double(envCPUs);
else
    numCPUs = 8;
end

cfg = jsondecode(fileread(variantsJsonPath));
stageDir = cfg.stage_dir;
sweepRoot = fileparts(stageDir);
chanPatterns = cellstr(cfg.channel_patterns);
xyPixelSize = cfg.xy_pixel_size;
dz = cfg.z_step_um;
dzPSF = cfg.dz_psf;
skewAngle = cfg.skew_angle_deg;
edgeErosion = cfg.edge_erosion;
psfPath = cfg.psf_path;
interpDefault = 'cubic';
if isfield(cfg, 'interp_method') && ~isempty(cfg.interp_method)
    interpDefault = cfg.interp_method;
end

variants = cfg.variants;
if isstruct(variants) && numel(variants) == 1
    variants = {variants};
elseif iscell(variants)
    % already a cell array of structs, fine
elseif isstruct(variants)
    variants = num2cell(variants);
end

nV = numel(variants);
fprintf('[Sweep] %d variants queued.\n', nV);

nChans = numel(chanPatterns);
% PetaKit5D's XR_decon_data_wrapper indexes psfFullpaths with a LOGICAL
% mask the width of channelPatterns (one-hot per matching channel), so
% supplying fewer entries than channels indexes out of bounds ("logical
% indices contain a true value outside of the array bounds") -- exactly
% the broadcast run_petakit_server.m already does for a multi-channel
% ticket with a single psf_path (see its 'Broadcasting single PSF to %d
% channels' branch).
psfPathsBroadcast = repmat({psfPath}, 1, nChans);

for vi = 1:nV
    v = variants{vi};
    name = v.name;
    fprintf('\n=== [%d/%d] variant: %s ===\n', vi, nV, name);
    tStart = tic;

    % XR_decon_data_wrapper's 'resultDirName' nests under its INPUT dataDir
    % (stageDir here), not under an arbitrary output root -- confirmed by
    % running it: 'Decon_prod' landed at stageDir/Decon_prod, not
    % sweepRoot/Decon_prod. Match that so step B looks in the right place.
    vStage = stageDir;
    if isfield(v, 'stage_dir') && ~isempty(v.stage_dir)
        vStage = v.stage_dir;
    end
    variantDir = fullfile(vStage, ['Decon_' name]);
    vErosion = edgeErosion;
    if isfield(v, 'edge_erosion') && ~isempty(v.edge_erosion)
        vErosion = v.edge_erosion;
    end
    vInterp = interpDefault;
    if isfield(v, 'interp_method') && ~isempty(v.interp_method)
        vInterp = v.interp_method;
    end
    vPsf = psfPathsBroadcast;
    if isfield(v, 'psf_paths') && ~isempty(v.psf_paths)
        vPsf = cellstr(v.psf_paths);
        if numel(vPsf) ~= nChans
            error('decon_sweep_driver:badPsfPaths', ...
                'variant %s: %d psf_paths for %d channels', name, numel(vPsf), nChans);
        end
    end
    padZ = 0;
    if isfield(v, 'pad_z') && ~isempty(v.pad_z)
        padZ = v.pad_z;
    end

    try

    if isfield(v, 'decon') && v.decon
        % ---- STEP A: skewed-space deconvolution ----
        val_method = normalizeRLMethodLocal(v.rl_method);
        val_iter = v.iters;
        val_bg = v.background;
        if isempty(val_bg)
            val_bg = [];
        end
        if isfield(v, 'alpha') && ~isempty(v.alpha)
            val_wAlpha = v.alpha;
        else
            val_wAlpha = 0.005;
        end
        if isfield(v, 'otf_thresh') && ~isempty(v.otf_thresh)
            val_otfCT = v.otf_thresh;
        else
            val_otfCT = 0.9;
        end
        if isfield(v, 'hann') && ~isempty(v.hann)
            val_hann = v.hann;
        else
            val_hann = [0.8, 1.0];
        end
        if isfield(v, 'damp') && ~isempty(v.damp)
            val_damp = v.damp;
        else
            val_damp = 1;
        end

        fprintf('[Sweep] Decon: method=%s iter=%d alpha=%g otfThresh=%g hann=[%g,%g] bg=%s damp=%g\n', ...
            val_method, val_iter, val_wAlpha, val_otfCT, val_hann(1), val_hann(2), mat2str(val_bg), val_damp);

        decon_in = vStage;
        if padZ > 0
            decon_in = fullfile(sweepRootFor(vStage), ['pad' num2str(padZ) '_' name]);
            padStageTiffs(vStage, decon_in, chanPatterns, padZ);
            variantDir = fullfile(decon_in, ['Decon_' name]);
        end

        XR_decon_data_wrapper( ...
            {decon_in}, ...
            'channelPatterns', chanPatterns, ...
            'psfFullpaths', vPsf, ...
            'deconIter', val_iter, ...
            'xyPixelSize', xyPixelSize, ...
            'dz', dz, ...
            'dzPSF', dzPSF, ...
            'skewAngle', skewAngle, ...
            'skewed', true, ...
            'background', val_bg, ...
            'edgeErosion', vErosion, ...
            'GPUJob', true, ...
            'RLMethod', val_method, ...
            'wienerAlpha', val_wAlpha, ...
            'OTFCumThresh', val_otfCT, ...
            'hannWinBounds', val_hann, ...
            'dampFactor', val_damp, ...
            'save16bit', true, ...
            'resultDirName', ['Decon_' name], ...
            'zarrFile', false, ...
            'parseCluster', false, ...
            'parseParfor', true, ...
            'masterCompute', true, ...
            'cpusPerTask', numCPUs, ...
            'overwrite', false ...
        );

        if padZ > 0
            cropDeconTiffs(variantDir, chanPatterns, padZ);
        end
        deskewInputDir = variantDir;
    else
        fprintf('[Sweep] No-decon arm: DSR runs straight on staged TIFFs.\n');
        deskewInputDir = vStage;
        variantDir = vStage;
    end

    % ---- STEP B: deskew/rotate (identical for every arm) ----
    fprintf('[Sweep] Deskew/Rotate -> %s/DSR\n', variantDir);
    XR_deskew_rotate_data_wrapper( ...
        {deskewInputDir}, ...
        'DSDirName', 'DS', ...
        'DSRDirName', 'DSR', ...
        'channelPatterns', chanPatterns, ...
        'deskew', true, ...
        'rotate', true, ...
        'xyPixelSize', xyPixelSize, ...
        'dz', dz, ...
        'skewAngle', skewAngle, ...
        'interpMethod', vInterp, ...
        'inputAxisOrder', 'yxz', ...
        'outputAxisOrder', 'yxz', ...
        'objectiveScan', false, ...
        'zStageScan', false, ...
        'reverse', true, ...
        'DSRCombined', true, ...
        'save16bit', true, ...
        'save3DStack', true, ...
        'saveMIP', false, ...
        'zarrFile', false, ...
        'parseCluster', false, ...
        'parseParfor', false, ...
        'masterCompute', true, ...
        'cpusPerTask', numCPUs, ...
        'overwrite', false ...
    );

    fprintf('[Sweep] variant %s done in %.1f s\n', name, toc(tStart));

    catch ME
        % One bad variant must not take the other 21 down with it --
        % mirrors run_petakit_server.m's own per-ticket try/catch.
        fprintf('[Sweep] !!! variant %s FAILED: %s\n', name, ME.message);
        fprintf('%s\n', getReport(ME));
    end

    % run_petakit_server.m always calls resetGpu after each job; do the
    % same here so one variant's leftover GPU memory can't OOM the next,
    % and so a failed variant still frees the GPU for the next one.
    try
        for g = 1:gpuDeviceCount
            reset(gpuDevice(g));
        end
    catch
    end
end

fprintf('\n[Sweep] All %d variants complete.\n', nV);
end

function method = normalizeRLMethodLocal(method)
% Mirror run_petakit_server.m's normalizeRLMethod: 'simple' -> 'simplified',
% and RLdecon.m's switch(RLMethod) has no `otherwise` branch (writes an
% EMPTY volume silently), so an unrecognized name errors here instead.
method = lower(strtrim(method));
if strcmp(method, 'simple')
    method = 'simplified';
end
valid = {'original', 'simplified', 'omw', 'cudagen'};
if ~ismember(method, valid)
    error('decon_sweep_driver:badRLMethod', ...
        'Unrecognized RLMethod "%s"; must be one of: %s', method, strjoin(valid, ', '));
end
end

function d = sweepRootFor(stageDir)
% Padded copies live beside the stage dir, never inside the dataset.
d = fileparts(stageDir);
end

function padStageTiffs(srcDir, dstDir, chanPatterns, P)
% Mirror-pad dim 3 (scan Z of the (ny,nx,nz) staging layout) by P planes.
if ~exist(dstDir, 'dir'), mkdir(dstDir); end
for ci = 1:numel(chanPatterns)
    files = dir(fullfile(srcDir, ['*' chanPatterns{ci} '*.tif']));
    for fi = 1:numel(files)
        out = fullfile(dstDir, files(fi).name);
        if exist(out, 'file'), continue; end
        vol = readtiff(fullfile(srcDir, files(fi).name));
        vol = padarray(vol, [0 0 P], 'symmetric');
        writetiff(vol, out);
    end
end
end

function cropDeconTiffs(deconDir, chanPatterns, P)
% Undo padStageTiffs on the decon output, in place, so DSR sees production's
% input shape. Idempotent via a sidecar marker (decon output is re-run safe).
for ci = 1:numel(chanPatterns)
    files = dir(fullfile(deconDir, ['*' chanPatterns{ci} '*.tif']));
    for fi = 1:numel(files)
        f = fullfile(deconDir, files(fi).name);
        marker = [f '.cropped'];
        if exist(marker, 'file'), continue; end
        vol = readtiff(f);
        vol = vol(:, :, P+1:end-P);
        writetiff(vol, f);
        fclose(fopen(marker, 'w'));
    end
end
end
