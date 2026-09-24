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
    variantDir = fullfile(stageDir, ['Decon_' name]);

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

        XR_decon_data_wrapper( ...
            {stageDir}, ...
            'channelPatterns', chanPatterns, ...
            'psfFullpaths', psfPathsBroadcast, ...
            'deconIter', val_iter, ...
            'xyPixelSize', xyPixelSize, ...
            'dz', dz, ...
            'dzPSF', dzPSF, ...
            'skewAngle', skewAngle, ...
            'skewed', true, ...
            'background', val_bg, ...
            'edgeErosion', edgeErosion, ...
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

        deskewInputDir = variantDir;
    else
        fprintf('[Sweep] No-decon arm: DSR runs straight on staged TIFFs.\n');
        deskewInputDir = stageDir;
        variantDir = stageDir;
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
        'interpMethod', 'cubic', ...
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
