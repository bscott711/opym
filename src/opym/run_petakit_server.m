%% run_petakit_server.m
% A persistent server that watches a directory for JSON job files.
% FEATURES:
%   - Intelligently dispatches BigTiff vs Standard jobs.
%   - Forces log flushing for real-time monitoring.
%   - Auto-shutdown timeout for releasing GPU resources.

% Lock this script's directory into the MATLAB path so we don't lose it if a job uses cd()
addpath(fileparts(mfilename('fullpath')));

% --- SYSTEM CONFIGURATION ------------------------------------------------
% 1. PetaKit Path
petakit_source_path = getenv('PETAKIT_ROOT');
if isempty(petakit_source_path)
    petakit_source_path = '/cm/shared/apps_local/petakit5d';
    logMsg('[Server] Warning: PETAKIT_ROOT not set. Using default: %s', petakit_source_path);
else
    logMsg('[Server] Using PetaKit path: %s', petakit_source_path);
end

% 2. Python Path
pythonPath = getenv('OPYM_PYTHON');
if isempty(pythonPath)
    [status, cmdOut] = system('which python');
    if status == 0
        pythonPath = strtrim(cmdOut);
        logMsg('[Server] ⚠️ OPYM_PYTHON not set. Falling back to system python: %s', pythonPath);
    else
        logMsg('[Server] ⚠️ Warning: OPYM_PYTHON not set and no python found in PATH.');
    end
else
    if ~exist(pythonPath, 'file')
        logMsg('[Server] ❌ CRITICAL: The path in OPYM_PYTHON does not exist:\n   %s', pythonPath);
    else
        logMsg('[Server] Using Python: %s', pythonPath);
    end
end

% Overridable so a test server can run against its own queue without
% touching production's (local_gpu_worker.py honors the same variable).
base_queue_dir = getenv('PETAKIT_JOBS_DIR');
if isempty(base_queue_dir)
    base_queue_dir = '/dev/shm/petakit_jobs';
end

% --- DYNAMIC CPU DETECTION -----------------------------------------------
envCPUs = getenv('PETAKIT_CPUS');
if ~isempty(envCPUs)
    numCPUs = str2double(envCPUs);
else
    numCPUs = 10; % Default to 10 workers to prevent GPU OOM
end

% Setup Directories
queue_dir = fullfile(base_queue_dir, 'queue');
done_dir  = fullfile(base_queue_dir, 'completed');
fail_dir  = fullfile(base_queue_dir, 'failed');

if ~exist(queue_dir, 'dir'), mkdir(queue_dir); end
% Priority lanes (see opym/lanes.py): live (streaming) tickets in queue_live
% are always claimed first; backfill tickets in queue only while no live
% acquisition holds a fresh LIVE_LEASE.json.
live_queue_dir = fullfile(base_queue_dir, 'queue_live');
if ~exist(live_queue_dir, 'dir'), mkdir(live_queue_dir); end
leasePath = fullfile(base_queue_dir, 'LIVE_LEASE.json');
if ~exist(done_dir, 'dir'),  mkdir(done_dir); end
if ~exist(fail_dir, 'dir'),  mkdir(fail_dir); end

% GPU pipeline-job concurrency lock, scoped per server (= per physical GPU,
% see local_gpu_worker.py's CUDA_VISIBLE_DEVICES assignment per PETAKIT_SERVER_ID).
envServerId = getenv('PETAKIT_SERVER_ID');
if isempty(envServerId), envServerId = 'default'; end
gpu_lock_dir = fullfile(base_queue_dir, 'gpu_locks', sprintf('server_%s', envServerId));
if ~exist(gpu_lock_dir, 'dir'), mkdir(gpu_lock_dir); end
% Clear stale locks from a previous crashed/killed server instance.
staleLocks = dir(fullfile(gpu_lock_dir, '*.lock'));
for si = 1:numel(staleLocks)
    delete(fullfile(gpu_lock_dir, staleLocks(si).name));
end

% Which ticket this server holds, for local_gpu_worker.py: a ticket claimed by
% a server that then dies (crash, OOM, kill) is put back in the queue instead
% of sitting as an orphaned .active_ claim forever. Written right after a
% claim, removed once the ticket resolves.
claims_dir = fullfile(base_queue_dir, 'claims');
if ~exist(claims_dir, 'dir'), mkdir(claims_dir); end
claimPath = fullfile(claims_dir, sprintf('S%s.json', envServerId));

% One JSON line per ticket (stage timings, frame count, outcome) in
% profiling/S<id>.jsonl -- the numbers the live lane's throughput budget is
% measured against. Never allowed to break a job: see writeProfile.
profiling_dir = fullfile(base_queue_dir, 'profiling');
if ~exist(profiling_dir, 'dir'), mkdir(profiling_dir); end

% --- INITIALIZATION ------------------------------------------------------
if ~exist('XR_deskew_rotate_data_wrapper', 'file')
    if exist(fullfile(petakit_source_path, 'setup.m'), 'file')
        run(fullfile(petakit_source_path, 'setup.m'));
    else
        warning('Could not find PetaKit setup.m. Decon/Deskew may fail.');
    end
end
addpath(fullfile(fileparts(mfilename('fullpath')), 'patches'));
% Shadow-patched parallelReadZarr: the shared PetaKit5D build
% (/cm/shared/apps_local/petakit5d, not writable by this account) can't
% parse a spec-legal `"compressor": null` (uncompressed) zarr v2 array --
% both its blosc and gzip metadata-parsing attempts throw on JSON null,
% and it mislabels the result "Metadata is incomplete. Check the .zarray
% file" (confirmed against zarr.cpp's compressor-parsing try/catch).
% Prepending our patched build here (source + patch notes in
% patches/cpp-zarr/) shadows the shared one via normal MATLAB path order,
% without touching the shared install other users share.
addpath(fullfile(fileparts(mfilename('fullpath')), 'patches', 'cpp-zarr', 'linux'));

% --- VERIFY MEX ---
verify_mex();

% --- GPU BINDING ---------------------------------------------------------
envGPU = getenv('PETAKIT_GPU_ID');
if ~isempty(envGPU)
    targetGpu = str2double(envGPU);
else
    targetGpu = 1; % Default to first GPU
end

try
    warning('off', 'parallel:gpu:device:DeviceLibsNeedsRecompiling');
    g = gpuDevice(targetGpu);
    logMsg('[Server] 🎮 Locked to GPU %d: %s (Available vRAM: %.1f GB)', g.Index, g.Name, g.AvailableMemory / 1e9);
catch e
    logMsg('[Server] ❌ Failed to lock GPU %d: %s', targetGpu, e.message);
end

% --- PARPOOL ---------------------------------------------------------------
% Started on the first job that needs it (ensurePool), not here: it takes
% ~24 s of a ~40 s boot, and live tickets never use it, so a server launched
% for an acquisition is ready that much sooner.
pool = [];

logMsg('[Server] Ready. Watching: %s', queue_dir);
% The live-session warm-up spec (opym.stream.live_zarr.WARMUP_NAME) and the
% last one this server acted on.
warmupPath = fullfile(base_queue_dir, 'live_warmup.json');
warmedStamp = 0;

% --- MAIN SERVER LOOP ----------------------------------------------------
envTimeout = getenv('PETAKIT_IDLE_TIMEOUT');
if ~isempty(envTimeout)
    idleTimeoutSec = str2double(envTimeout);
else
    idleTimeoutSec = 300; % Default to 5 minutes
end
idleTimer = 0;

while true
    [jobFiles, claim_dir, liveActive] = nextJobFiles(live_queue_dir, queue_dir, leasePath);

    if isempty(jobFiles)
        % A new live session: warm up for it while there's nothing to do.
        [warmedStamp, warmed] = maybeWarmup(warmupPath, warmedStamp, numCPUs, profiling_dir, envServerId);
        if warmed
            idleTimer = 0;
            continue;
        end
        % Short poll: a live ticket waiting here is time the live view is
        % behind (up to 2 s per timepoint with the old 2 s pause). dir() on
        % the tmpfs queue is cheap; poll hardest while an acquisition holds
        % the live lease.
        if liveActive, pollS = 0.02; else, pollS = 0.1; end
        pause(pollS);
        idleTimer = idleTimer + pollS;

        if idleTimeoutSec > 0 && idleTimer >= idleTimeoutSec
            logMsg('[Server] Idle timeout (%d s) reached. Shutting down to release GPUs.', idleTimeoutSec);
            break; % Exits the while loop, allowing Matlab to close
        end
        continue;
    end

    % Reset the timer the moment we find work
    idleTimer = 0;

    % Atomic file lock (prevents both GPUs from grabbing same job): the
    % oldest ticket this server manages to rename is its. One the other
    % server just took is skipped, not waited out -- a live timepoint's two
    % channels land together, and the old rand()*1.5 s backoff after losing
    % the first could hold the second back that long.
    [~, claimLane] = fileparts(claim_dir);
    currentFile = '';
    for k = 1:numel(jobFiles)
        srcPath = fullfile(claim_dir, jobFiles(k).name);
        activePath = fullfile(claim_dir, ['.active_' jobFiles(k).name]);
        if claimFile(srcPath, activePath)
            currentFile = jobFiles(k).name;
            break;
        end
    end
    if isempty(currentFile)
        continue;
    end

    logMsg('[Server] >>> Processing job: %s (%s)', currentFile, claimLane);
    writeClaim(claimPath, envServerId, currentFile, claimLane);
    tTicket = tic;
    prof = struct('ticket', currentFile, 'lane', claimLane, 'server_id', envServerId, ...
        'started_at', posixtime(datetime('now', 'TimeZone', 'UTC')), ...
        'job_type', '', 'data_dir', '', 'n_input_tifs', NaN, ...
        'decon_s', NaN, 'dsr_s', NaN, 'read_s', NaN, 'view_s', NaN, 'write_s', NaN, ...
        'total_s', NaN, 'status', '', 'error', '');
    % Defined before the try: the catch block reads it, and a ticket that fails
    % before its jobType is parsed (e.g. malformed JSON) would otherwise throw
    % an undefined-variable error from inside the catch and kill the server.
    jobType = '';

    try
        fid = fopen(activePath);
        raw = fread(fid, inf);
        fclose(fid);
        job = jsondecode(char(raw'));

        if isfield(job, 'parameters')
            p = job.parameters;
        else
            p = struct();
        end

        jobType = safelyGetParam(job, 'jobType', 'deskew');
        prof.job_type = jobType;
        prof.data_dir = safelyGetParam(job, 'dataDir', '');
        if ~strcmp(jobType, 'live_zarr')
            pool = ensurePool(pool, numCPUs, targetGpu);
        end

        % Echo the revision of the opym checkout that BUILT this ticket. This
        % MATLAB process loads run_petakit_server.m exactly once at startup,
        % so after a code change without `systemctl --user restart opym-serve`
        % a stale server silently interprets new tickets. Printing the
        % submitter's rev next to the server's own startup banner makes that
        % mismatch visible in the log instead of inferred hours later.
        logMsg('[Server] Ticket built by opym rev %s (jobType=%s)', ...
            safelyGetParam(job, 'submitterRev', 'unknown'), jobType);

        switch jobType
            case 'crop'
                % --- CROPPING JOB ---
                logMsg('         Type: OPM Cropping');

                % Check if BigTiff
                isBigTiff = endsWith(job.dataDir, '.ome.tif', 'IgnoreCase', true);

                if isBigTiff
                    logMsg('         -> Mode: BigTiff Split (Parallel)');
                    % Calls the new BigTiff cropper (Requires 'job' struct)
                    run_bigtiff_cropper(job);
                else
                    logMsg('         -> Mode: Standard PetaKit Crop');
                    % Calls the legacy cropper (Requires 'path' string)
                    run_petakit_cropper(srcPath);
                end

            case 'pipeline'
                % --- UNIFIED GPU PIPELINE JOB ---
                logMsg('         Type: Unified GPU Pipeline (RAM Disk)');
                val_shm    = safelyGetParam(p, 'shm_path', '');
                val_psfs   = safelyGetParam(p, 'psf_paths', {});
                val_xyPix  = safelyGetParam(p, 'xy_pixel_size', 0.136);
                val_zStep  = safelyGetParam(p, 'z_step_um', 0.3);
                val_angle  = safelyGetParam(p, 'sheet_angle_deg', 60.0);
                val_interp = safelyGetParam(p, 'interp_method', 'cubic');
                val_method = normalizeRLMethod(safelyGetParam(p, 'rl_method', 'simple'));
                if strcmp(val_method, 'omw')
                    default_iter = 2;
                else
                    default_iter = 25;
                end
                val_iter   = safelyGetParam(p, 'iterations', default_iter);
                val_zarr   = safelyGetParam(p, 'save_zarr', true);
                val_debug  = safelyGetParam(p, 'debug', false);
                val_dzPSF  = safelyGetParam(p, 'dz_psf', []);

                if isempty(val_shm)
                    error('No /dev/shm/ path provided for pipeline job.');
                end

                % outputFn corresponds to dataDir / baseName (which will be a TIF)
                % The ticket provides dataDir as the target directory (e.g. cell_1/Bot)
                % and baseName as the file name (e.g. cell_MMStack_Pos0_T0000_C0_bot.tif)
                outFn = fullfile(job.dataDir, job.baseName);

                % Ensure array is correct
                if iscell(val_psfs) && ~isempty(val_psfs)
                    psfFn = val_psfs{1};
                elseif isstring(val_psfs) || ischar(val_psfs)
                    psfFn = val_psfs;
                else
                    psfFn = '';
                end

                % dz_psf is required whenever a PSF (i.e. deconvolution) is
                % requested -- a silent wrong default here previously caused
                % the decon step to resample against the wrong z-geometry.
                if ~isempty(psfFn) && isempty(val_dzPSF)
                    error(['pipeline job requests deconvolution (psf_paths set) but is ', ...
                        'missing required "dz_psf" parameter (the PSF''s own z-step, in um).']);
                end

                % --- GPU CONCURRENCY LOCK ---
                % Only one 'pipeline' job may run on this server's GPU at a time
                % (deconvolution + DSR both allocate full-volume GPU buffers;
                % running >1 concurrently reliably OOMs the device). Each server
                % process owns exactly one physical GPU (see local_gpu_worker.py),
                % so this lock is scoped per-server via PETAKIT_SERVER_ID.
                maxGpuJobs = str2double(getenv('PETAKIT_MAX_GPU_PIPELINE_JOBS'));
                if isnan(maxGpuJobs)
                    maxGpuJobs = 1;
                end
                lockAcquired = false;
                while ~lockAcquired
                    existingLocks = dir(fullfile(gpu_lock_dir, '*.lock'));
                    if numel(existingLocks) < maxGpuJobs
                        lockName = fullfile(gpu_lock_dir, [currentFile '.lock']);
                        lfid = fopen(lockName, 'w');
                        if lfid > 0
                            fclose(lfid);
                            lockAcquired = true;
                        end
                    end
                    if ~lockAcquired
                        pause(1 + rand());
                    end
                end

                % Note on PSF/OTF cache warmth (persistent vars in
                % decon_lucy_function.m and run_gpu_pipeline.m are per-worker-
                % process): since the GPU lock above already fully serializes
                % pipeline-job dispatch to this pool (one in flight at a time),
                % parfeval's scheduler was empirically verified to always
                % reassign the job to the same, now-idle worker rather than
                % scattering across the pool -- 8/8 consecutive serialized
                % jobs landed on the same worker PID in an isolated probe
                % (see the performance plan's Phase 1.1). So both caches
                % already stay warm across same-PSF jobs without needing a
                % dedicated single-worker pool; no dispatch change was made.
                % OMW back-projector knobs (used only when rl_method='omw';
                % defaults reproduce PetaKit5D's stock behavior otherwise).
                val_wAlpha = safelyGetParam(p, 'wiener_alpha', 0.005);
                val_otfCT  = safelyGetParam(p, 'otf_cum_thresh', 0.9);
                val_hann   = safelyGetParam(p, 'hann_win_bounds', [0.8, 1.0]);

                f = parfeval(pool, @run_gpu_pipeline_async, 0, activePath, done_dir, fail_dir, val_shm, outFn, psfFn, gpu_lock_dir, currentFile, ...
                    'xyPixelSize', val_xyPix, ...
                    'z_step_um', val_zStep, ...
                    'DeconIter', val_iter, ...
                    'RLMethod', val_method, ...
                    'SkewAngle', val_angle, ...
                    'interpMethod', val_interp, ...
                    'saveZarr', val_zarr, ...
                    'debug', val_debug, ...
                    'wienerAlpha', val_wAlpha, ...
                    'OTFCumThresh', val_otfCT, ...
                    'hannWinBounds', val_hann, ...
                    'dzPSF', val_dzPSF);

            case 'pipeline_batch'
                % --- UNIFIED GPU PIPELINE BATCH JOB ---
                % Same per-frame decon->DSR->zarr-save logic as the 'pipeline'
                % case above, but N (shm_path, output_file) pairs that share
                % one PSF are processed inside a single parfeval'd call, so
                % the GPU concurrency lock below (and the PSF/OTF persistent
                % caches in run_gpu_pipeline.m / decon_lucy_function.m) are
                % only paid for once per batch instead of once per frame.
                % See the performance plan's Phase 2 for the rationale.
                logMsg('         Type: Unified GPU Pipeline Batch (RAM Disk)');
                val_items  = safelyGetParam(p, 'items', []);
                val_psfs   = safelyGetParam(p, 'psf_paths', {});
                val_xyPix  = safelyGetParam(p, 'xy_pixel_size', 0.136);
                val_zStep  = safelyGetParam(p, 'z_step_um', 0.3);
                val_angle  = safelyGetParam(p, 'sheet_angle_deg', 60.0);
                val_interp = safelyGetParam(p, 'interp_method', 'cubic');
                val_method = normalizeRLMethod(safelyGetParam(p, 'rl_method', 'simple'));
                if strcmp(val_method, 'omw')
                    default_iter = 2;
                else
                    default_iter = 25;
                end
                val_iter   = safelyGetParam(p, 'iterations', default_iter);
                val_zarr   = safelyGetParam(p, 'save_zarr', true);
                val_debug  = safelyGetParam(p, 'debug', false);
                val_dzPSF  = safelyGetParam(p, 'dz_psf', []);

                if isempty(val_items)
                    error('pipeline_batch job requires a non-empty "items" array.');
                end

                % Ensure array is correct
                if iscell(val_psfs) && ~isempty(val_psfs)
                    psfFn = val_psfs{1};
                elseif isstring(val_psfs) || ischar(val_psfs)
                    psfFn = val_psfs;
                else
                    psfFn = '';
                end

                if ~isempty(psfFn) && isempty(val_dzPSF)
                    error(['pipeline_batch job requests deconvolution (psf_paths set) but is ', ...
                        'missing required "dz_psf" parameter (the PSF''s own z-step, in um).']);
                end

                % --- GPU CONCURRENCY LOCK (one lock for the whole batch) ---
                % Identical acquire mechanism to the 'pipeline' case -- same
                % gpu_lock_dir, same *.lock-file counting against
                % PETAKIT_MAX_GPU_PIPELINE_JOBS -- just held for the whole
                % batch instead of re-acquired per frame.
                maxGpuJobs = str2double(getenv('PETAKIT_MAX_GPU_PIPELINE_JOBS'));
                if isnan(maxGpuJobs)
                    maxGpuJobs = 1;
                end
                lockAcquired = false;
                while ~lockAcquired
                    existingLocks = dir(fullfile(gpu_lock_dir, '*.lock'));
                    if numel(existingLocks) < maxGpuJobs
                        lockName = fullfile(gpu_lock_dir, [currentFile '.lock']);
                        lfid = fopen(lockName, 'w');
                        if lfid > 0
                            fclose(lfid);
                            lockAcquired = true;
                        end
                    end
                    if ~lockAcquired
                        pause(1 + rand());
                    end
                end

                % OMW back-projector knobs (see 'pipeline' case above).
                val_wAlpha = safelyGetParam(p, 'wiener_alpha', 0.005);
                val_otfCT  = safelyGetParam(p, 'otf_cum_thresh', 0.9);
                val_hann   = safelyGetParam(p, 'hann_win_bounds', [0.8, 1.0]);

                f = parfeval(pool, @run_gpu_pipeline_batch_async, 0, activePath, done_dir, fail_dir, val_items, psfFn, gpu_lock_dir, currentFile, ...
                    'xyPixelSize', val_xyPix, ...
                    'z_step_um', val_zStep, ...
                    'DeconIter', val_iter, ...
                    'RLMethod', val_method, ...
                    'SkewAngle', val_angle, ...
                    'interpMethod', val_interp, ...
                    'saveZarr', val_zarr, ...
                    'debug', val_debug, ...
                    'wienerAlpha', val_wAlpha, ...
                    'OTFCumThresh', val_otfCT, ...
                    'hannWinBounds', val_hann, ...
                    'dzPSF', val_dzPSF);

            case 'decon'
                % --- DECONVOLUTION JOB ---
                logMsg('         Type: Deconvolution');
                val_resDir = safelyGetParam(p, 'result_dir_name', 'decon');
                val_chans  = safelyGetParam(p, 'channel_patterns', {job.baseName});
                val_psfs   = safelyGetParam(p, 'psf_paths', {});
                % Default 'omw' matches petakit.py's submit_remote_decon_job;
                % this jobType reaches RLdecon.m's switch.
                val_method = normalizeRLMethod(safelyGetParam(p, 'rl_method', 'omw'));
                if strcmp(val_method, 'omw')
                    default_iter = 2;
                else
                    default_iter = 25;
                end
                val_iter   = safelyGetParam(p, 'iterations', default_iter);
                val_gpu    = safelyGetParam(p, 'gpu_job', true);
                val_skewed = safelyGetParam(p, 'skewed', true);
                val_16bit  = safelyGetParam(p, 'save_16bit', true);

                if isstring(val_chans), val_chans = cellstr(val_chans); end
                if isstring(val_psfs), val_psfs = cellstr(val_psfs); end
                if ischar(val_psfs), val_psfs = {val_psfs}; end
                if isempty(val_psfs)
                    error(['decon job has no "psf_paths". XR_decon_data_wrapper ', ...
                        'would fail indexing dc_psfFullpaths{psfMapping} several ', ...
                        'steps later, which reads as a PSF-file problem rather ', ...
                        'than a missing parameter.']);
                end

                % Ensure number of PSFs matches number of channels
                if numel(val_chans) > 1 && numel(val_psfs) == 1
                    logMsg('         [Decon] Broadcasting single PSF to %d channels.', numel(val_chans));
                    val_psfs = repmat(val_psfs, 1, numel(val_chans));
                end

                val_wAlpha = safelyGetParam(p, 'wiener_alpha', 0.005);
                val_otfCT  = safelyGetParam(p, 'otf_cum_thresh', 0.9);
                val_hann   = safelyGetParam(p, 'hann_win_bounds', [0.8, 1.0]);
                % Only has an effect when > 1 (decon_lucy_omw_function.m),
                % where it caps a decon value's departure from its own input
                % -- PetaKit5D's own remedy for isolated over-sharpened
                % voxel spikes. Default 1 matches PetaKit5D's own default
                % (off), so a ticket that omits it is unchanged.
                val_damp   = safelyGetParam(p, 'damp_factor', 1);

                % Geometry. XR_decon_data_wrapper defaults dz=0.5 and
                % dzPSF=0.1, and psf_gen_new FFT-decimates the PSF's dim 3 by
                % dz/dzPSF whenever that ratio is > 1. Deconvolving data that
                % has ALREADY been deskewed and rotated means data and PSF
                % share one isotropic lab grid, so both must be the lab voxel
                % size -- otherwise the PSF is silently shrunk 5x and the
                % result looks merely disappointing rather than wrong.
                val_xy     = safelyGetParam(p, 'xy_pixel_size', 0.108);
                val_dz     = safelyGetParam(p, 'z_step_um', []);
                val_dzPSF  = safelyGetParam(p, 'dz_psf', []);
                val_bg     = safelyGetParam(p, 'background', []);
                val_erode  = safelyGetParam(p, 'edge_erosion', 0);
                % Required, same as the pipeline branches above: this jobType
                % always deconvolves, so there is no case where guessing the
                % geometry is better than refusing to run.
                if isempty(val_dz) || isempty(val_dzPSF)
                    error(['decon job is missing required "z_step_um" and/or "dz_psf" ', ...
                        '(the data''s and the PSF''s z-steps, in um). psf_gen_new ', ...
                        'decimates the PSF by z_step_um/dz_psf, so a wrong default ', ...
                        'silently shrinks the PSF instead of failing.']);
                end
                logMsg(['         [Decon] method=%s iter=%d skewed=%d alpha=%g ' ...
                        'xy=%g dz=%g dzPSF=%g erode=%d'], val_method, val_iter, ...
                        val_skewed, val_wAlpha, val_xy, val_dz, val_dzPSF, val_erode);

                XR_decon_data_wrapper( ...
                    {job.dataDir}, ...
                    'resultDirName', val_resDir, ...
                    'channelPatterns', val_chans, ...
                    'psfFullpaths', val_psfs, ...
                    'deconIter', val_iter, ...
                    'GPUJob', val_gpu, ...
                    'skewed', val_skewed, ...
                    'RLMethod', val_method, ...
                    'wienerAlpha', val_wAlpha, ...
                    'OTFCumThresh', val_otfCT, ...
                    'hannWinBounds', val_hann, ...
                    'dampFactor', val_damp, ...
                    'xyPixelSize', val_xy, ...
                    'dz', val_dz, ...
                    'dzPSF', val_dzPSF, ...
                    'background', val_bg, ...
                    'edgeErosion', val_erode, ...
                    'save16bit', val_16bit, ...
                    'parseCluster', false, ...
                    'parseParfor', true, ...
                    'masterCompute', true, ...
                    'cpusPerTask', numCPUs ...
                );

            case 'live_zarr'
                % One streamed (t, c) volume, raw OME-Zarr in -> processed
                % OME-Zarr out, in memory (opym.stream.live's one-format
                % path). See run_live_zarr.m.
                logMsg('[Server] Live zarr: T=%d C=%d -> %s', p.t, p.c, char(p.levels(1)));
                liveStats = run_live_zarr(p, numCPUs);
                prof.n_input_tifs = liveStats.frames;
                prof.read_s = liveStats.read_s;
                prof.decon_s = liveStats.decon_s;
                prof.dsr_s = liveStats.dsr_s;
                prof.write_s = liveStats.write_s;
                prof.view_s = liveStats.view_s;

            case 'live'
                % Streamed timepoints (opym.stream.live): decon -> DSR per
                % frame through the same frame functions the wrappers below
                % call, minus the wrappers. See run_live_frames.m.
                logMsg('[Server] Live: %d frame(s) -> %s', numel(cellstr(p.frames)), char(p.decon_dir));
                liveStats = run_live_frames(p, numCPUs);
                prof.n_input_tifs = liveStats.frames;
                prof.decon_s = liveStats.decon_s;
                prof.dsr_s = liveStats.dsr_s;

            otherwise
                % --- DESKEW / DECONVOLUTION / ROTATION PIPELINE ---

                % 1. Extract Shared Parameters
                % These defaults are a backstop only -- every real ticket
                % sets all three. `z_step_um` used to default to 1.0 here
                % while the two branches above defaulted to 0.3, so the same
                % missing field meant different geometry depending on job
                % type; aligned to 0.3. Geometry is logged below because a
                % wrong-but-plausible step produces output that looks fine
                % and is silently the wrong size.
                val_xy        = safelyGetParam(p, 'xy_pixel_size', 0.136);
                val_dz        = safelyGetParam(p, 'z_step_um', 0.3);
                val_ang       = safelyGetParam(p, 'sheet_angle_deg', 60.0);
                if ~isfield(p, 'z_step_um') || isempty(p.z_step_um)
                    logMsg('[Server] WARNING: ticket set no z_step_um; falling back to %g um', val_dz);
                end
                val_chans     = safelyGetParam(p, 'channel_patterns', {job.baseName});
                if ischar(val_chans) || isstring(val_chans)
                    val_chans = {val_chans};
                end

                % Decon Params
                val_psfPath   = safelyGetParam(p, 'psf_path', '');
                val_runDecon  = safelyGetParam(p, 'run_decon', ~isempty(val_psfPath));
                % Deskew/Rotate Params
                val_deskew    = safelyGetParam(p, 'deskew', true);
                val_rotate    = safelyGetParam(p, 'rotate', true);
                val_interp    = safelyGetParam(p, 'interp_method', 'cubic');
                % Default 'omw' matches petakit.py's submit_remote_deskew_job.
                val_method    = normalizeRLMethod(safelyGetParam(p, 'rl_method', 'omw'));
                if strcmp(val_method, 'omw')
                    default_iter = 2;
                else
                    default_iter = 25;
                end
                val_iter      = safelyGetParam(p, 'decon_iter', default_iter);
                val_dsDir     = safelyGetParam(p, 'ds_dir_name', 'DS');
                val_dsrDir    = safelyGetParam(p, 'dsr_dir_name', 'DSR');
                val_saveMIP   = safelyGetParam(p, 'save_mip', false); % preserve prior default for existing callers
                % Newer pymmcore-based acquisitions write already-cropped,
                % per-channel data straight to zarr instead of the legacy
                % Micro-Manager OME-TIFF format. Both
                % XR_decon_data_wrapper and XR_deskew_rotate_data_wrapper
                % accept a 'zarrFile' flag that changes how they discover
                % input files by channelPatterns (.zarr instead of .tif);
                % XR_deskewRotateFrame itself already reads either
                % (readtiff/readzarr, dispatched on file extension).
                % Default false preserves every existing caller's behavior.
                val_zarrInput = safelyGetParam(p, 'zarr_input', false);

                % ✅ Axis Order Parameters
                val_inputAxis  = safelyGetParam(p, 'input_axis_order', 'yxz');
                val_outputAxis = safelyGetParam(p, 'output_axis_order', 'yxz');

                % 2. Execution logic
                current_input_dir = job.dataDir;
                prof.n_input_tifs = numel(dir(fullfile(job.dataDir, '*.tif')));

                if val_runDecon && ~isempty(val_psfPath)
                    % --- STEP A: Deconvolution (Skewed) ---
                    logMsg('         Type: Deconvolution (Skewed Mode)');
                    deconDirName = 'Decon'; % Consistent output name for pipeline

                    % PetaKit5D indexes psfFullpaths by channel, in the same
                    % order as channelPatterns. A ticket may supply either
                    % `psf_paths` (per-channel, preferred) or the single
                    % `psf_path` shorthand, which we broadcast.
                    val_psfList = safelyGetParam(p, 'psf_paths', {});
                    if ischar(val_psfList) || isstring(val_psfList)
                        val_psfList = {char(val_psfList)};
                    elseif iscell(val_psfList)
                        val_psfList = reshape(cellfun(@char, val_psfList, ...
                            'UniformOutput', false), 1, []);
                    end
                    if ~isempty(val_psfList)
                        if numel(val_psfList) ~= numel(val_chans)
                            error('run_petakit_server:psfChannelMismatch', ...
                                ['psf_paths has %d entries but channel_patterns ' ...
                                 'has %d; PetaKit5D indexes psfFullpaths by ' ...
                                 'channel.'], numel(val_psfList), numel(val_chans));
                        end
                        val_psfs = val_psfList;
                        logMsg('         [Decon] Using %d per-channel PSFs.', numel(val_psfs));
                    else
                        val_psfs = {val_psfPath};
                        if numel(val_chans) > 1
                            logMsg('         [Decon] Broadcasting single PSF to %d channels.', numel(val_chans));
                            val_psfs = repmat(val_psfs, 1, numel(val_chans));
                        end
                    end
                    for pi = 1:numel(val_psfs)
                        if ~exist(val_psfs{pi}, 'file')
                            error('run_petakit_server:psfNotFound', ...
                                'PSF not found: %s', val_psfs{pi});
                        end
                    end

                    val_gpuDecon = safelyGetParam(p, 'gpu_decon', false);
                    % Camera offset subtracted before decon. Left empty,
                    % XR_decon_data_wrapper resolves its own default of 100,
                    % which matches this microscope's measured dark level; for
                    % 'omw'/'simplified' it is subtracted inside the decon
                    % kernel rather than up front.
                    val_bg       = safelyGetParam(p, 'background', []);
                    % RLdecon edge-tapers each z-PLANE laterally but never
                    % along z, so a short scan rings at its z faces. Eroding
                    % the result's boundary is PetaKit5D's own remedy.
                    val_erode    = safelyGetParam(p, 'edge_erosion', 0);

                    val_wAlpha = safelyGetParam(p, 'wiener_alpha', 0.005);
                    val_otfCT  = safelyGetParam(p, 'otf_cum_thresh', 0.9);
                    val_hann   = safelyGetParam(p, 'hann_win_bounds', [0.8, 1.0]);
                    % Only has an effect when > 1 (decon_lucy_omw_function.m),
                    % where it caps a decon value's departure from its own
                    % input -- PetaKit5D's own remedy for isolated
                    % over-sharpened voxel spikes. Default 1 matches
                    % PetaKit5D's own default (off).
                    val_damp   = safelyGetParam(p, 'damp_factor', 1);

                    logMsg(['[Server] Decon: method=%s, iters=%d, wienerAlpha=%g, ' ...
                        'OTFCumThresh=%g, dampFactor=%g, edgeErosion=%d, skewed=1, psf=%s'], val_method, val_iter, ...
                        val_wAlpha, val_otfCT, val_damp, val_erode, val_psfs{1});

                    tDecon = tic;
                    XR_decon_data_wrapper( ...
                        {current_input_dir}, ...
                        'channelPatterns', val_chans, ...
                        'psfFullpaths', val_psfs, ...
                        'deconIter', val_iter, ...
                        'xyPixelSize', val_xy, ...
                        'dz', val_dz, ...
                        'skewAngle', val_ang, ...
                        'skewed', true, ...
                        'background', val_bg, ...
                        'edgeErosion', val_erode, ...
                        'GPUJob', val_gpuDecon, ...
                        'RLMethod', val_method, ...
                        'wienerAlpha', val_wAlpha, ...
                        'OTFCumThresh', val_otfCT, ...
                        'hannWinBounds', val_hann, ...
                        'dampFactor', val_damp, ...
                        'save16bit', true, ...
                        'resultDirName', deconDirName, ...
                        'zarrFile', val_zarrInput, ...
                        'parseCluster', false, ...
                        'parseParfor', true, ...
                        'masterCompute', true, ...
                        'cpusPerTask', numCPUs ...
                    );

                    prof.decon_s = toc(tDecon);
                    % Update input for the next step to point to the deconvolved results
                    current_input_dir = fullfile(job.dataDir, deconDirName);
                end

                if val_deskew || val_rotate
                    % --- STEP B: Deskew / Rotate ---
                    logMsg('         Type: Deskew/Rotate');

                    % Scan Geometry Parameters (from JSON ticket)
                    val_objScan   = safelyGetParam(p, 'objective_scan', false);
                    val_zStage    = safelyGetParam(p, 'z_stage_scan', false);
                    val_reverse   = safelyGetParam(p, 'reverse', false);

                    % zarr_input describes job.dataDir (the ORIGINAL raw
                    % input), not necessarily current_input_dir. If decon
                    % ran first (same gating condition as the decon branch
                    % above: val_runDecon && ~isempty(val_psfPath)), this
                    % stage reads FROM decon's own output directory instead,
                    % which XR_decon_data_wrapper writes as TIFF by default
                    % (its 'saveZarr' isn't threaded from our ticket, so
                    % it's always false here) -- so only pass zarrFile
                    % through when decon did NOT actually run and
                    % current_input_dir is still the original raw zarr dir.
                    val_deconRan = val_runDecon && ~isempty(val_psfPath);
                    val_deskewZarrInput = val_zarrInput && ~val_deconRan;

                    % Echo the geometry actually in force. A zarr mirror is
                    % (z,y,x) on disk and needs inputAxisOrder 'zxy' to
                    % reach PetaKit5D as (y,x,z) in this microscope's
                    % convention (the tilted axis and the coverslip axis are
                    % named the opposite way round from what PetaKit5D
                    % means by them -- see the comment in petakit.py's
                    % submit_remote_deskew_job); 'yxz' on a zarr input means
                    % the scan planes are being sheared as image rows.
                    logMsg('[Server] DSR geometry: xyPixelSize=%g um, dz=%g um, skewAngle=%g deg, inputAxisOrder=%s, zarrFile=%d', ...
                        val_xy, val_dz, val_ang, val_inputAxis, val_deskewZarrInput);
                    if val_deskewZarrInput && strcmpi(val_inputAxis, 'yxz')
                        logMsg('[Server] WARNING: zarr input with inputAxisOrder=yxz -- DSR output will be the wrong size.');
                    end

                    tDsr = tic;
                    XR_deskew_rotate_data_wrapper( ...
                        {current_input_dir}, ...
                        'DSDirName', val_dsDir, ...
                        'DSRDirName', val_dsrDir, ...
                        'channelPatterns', val_chans, ...
                        'deskew', val_deskew, ...
                        'rotate', val_rotate, ...
                        'xyPixelSize', val_xy, ...
                        'dz', val_dz, ...
                        'skewAngle', val_ang, ...
                        'interpMethod', val_interp, ...
                        'inputAxisOrder', val_inputAxis, ...
                        'outputAxisOrder', val_outputAxis, ...
                        'objectiveScan', val_objScan, ...
                        'zStageScan', val_zStage, ...
                        'reverse', val_reverse, ...
                        'DSRCombined', true, ...
                        'save16bit', true, ...
                        'save3DStack', true, ...
                        'saveMIP', val_saveMIP, ...
                        'zarrFile', val_deskewZarrInput, ...
                        'parseCluster', false, ...
                        'parseParfor', false, ...
                        'masterCompute', true, ...
                        'cpusPerTask', numCPUs ...
                    );
                    prof.dsr_s = toc(tDsr);
                end
        end % End switch jobType
        prof.status = 'done';

        if ~ismember(jobType, {'pipeline', 'pipeline_batch'})
            movefile(activePath, fullfile(done_dir, currentFile));
            logMsg('[Server] <<< Finished: %s', currentFile);

            % --- FREE GPU MEMORY --- except between live tickets: a reset
            % costs ~0.3 s of this server's time and throws away what the
            % next one reuses (the decon's cached OTFs, cuFFT plans, the
            % memory pool). The next backfill job's reset frees them.
            if ~strcmp(jobType, 'live_zarr')
                try
                    for g = 1:gpuDeviceCount
                        reset(gpuDevice(g));
                    end
                catch
                end
            end
        else
            logMsg('[Server] <<< Dispatched %s to background worker.', currentFile);
        end

    catch ME
        logMsg('[Server] !!! ERROR on %s: %s', currentFile, ME.message);
        prof.status = 'failed';
        prof.error = ME.message;
        if ~ismember(jobType, {'pipeline', 'pipeline_batch'})
            % Never let a missing claim take the server down with it.
            if exist(activePath, 'file')
                movefile(activePath, fullfile(fail_dir, currentFile));
            else
                logMsg('[Server] !!! Claim %s is gone; nothing to move to failed/.', activePath);
            end
            errLog = fullfile(fail_dir, [currentFile '.log']);
            fid = fopen(errLog, 'w');
            fprintf(fid, '%s\n', getReport(ME));
            fclose(fid);

            % --- FREE GPU MEMORY ---
            try
                for g = 1:gpuDeviceCount
                    reset(gpuDevice(g));
                end
            catch
            end
        end
    end
    clearClaim(claimPath);
    prof.total_s = toc(tTicket);
    writeProfile(profiling_dir, envServerId, prof);
end

function ok = claimFile(src, dst)
    % Claim a ticket with one rename(2) (through Java): atomic, and a server
    % that loses the race finds its source gone and touches nothing. Not
    % MATLAB's movefile -- with two servers racing for one ticket, the
    % loser's movefile left the winner without its claim file in 7 of 150
    % contested claims (2026-09-25; 0 of 150 with rename). That crashed a
    % server on the very next line, fopen of its own claim.
    ok = java.io.File(src).renameTo(java.io.File(dst));
end

function pool = ensurePool(pool, numCPUs, targetGpu)
    % The parallel pool every non-live job type may use (parfeval'd pipeline
    % jobs, parseParfor wrappers), started on first need.
    if ~isempty(pool) && isvalid(pool)
        return;
    end
    pool = gcp('nocreate');
    if isempty(pool) || pool.NumWorkers ~= numCPUs
        try
            delete(pool);
            pc = parcluster('local');
            envServer = getenv('PETAKIT_SERVER_ID');
            if isempty(envServer), envServer = num2str(targetGpu); end
            pc.JobStorageLocation = fullfile(getenv('HOME'), '.matlab', 'local_cluster_jobs', sprintf('server_%s', envServer));
            if ~exist(pc.JobStorageLocation, 'dir')
                mkdir(pc.JobStorageLocation);
            end
            logMsg('[Server] Starting parallel pool (%d workers) for non-live work...', numCPUs);
            pool = parpool(pc, numCPUs, 'IdleTimeout', Inf);
        catch
            logMsg('[Server] Warning: Could not start parpool. Continuing...');
        end
    end
    % Prevent the pool from shutting down after 30 minutes of inactivity
    pool = gcp('nocreate');
    if ~isempty(pool)
        pool.IdleTimeout = Inf;
    end
end

function [stamp, warmed] = maybeWarmup(path, stamp, numCPUs, profiling_dir, serverId)
    % Once per new warm-up spec (its mtime), if it is fresh: a live session
    % started in the last 10 minutes. Never allowed to break the server.
    warmed = false;
    d = dir(path);
    if isempty(d) || d(1).datenum == stamp
        return;
    end
    stamp = d(1).datenum;
    if (now - stamp) * 86400 > 600
        return;
    end
    warmed = true;
    t0 = tic;
    try
        spec = jsondecode(fileread(path));
        logMsg('[Server] Warming up for live session %s ...', spec.session_id);
        s = run_live_zarr(spec.parameters, numCPUs);
        logMsg('[Server] Warm for session %s (%.1f s).', spec.session_id, s.decon_s);
        writeProfile(profiling_dir, serverId, struct('ticket', ['warmup:' spec.session_id], ...
            'job_type', 'live_warmup', 'total_s', toc(t0), 'status', 'done', 'error', ''));
    catch ME
        logMsg('[Server] Warm-up failed (the first timepoint will just be slower): %s', ME.message);
    end
end

function [jobFiles, fromDir, liveActive] = nextJobFiles(liveDir, backfillDir, leasePath)
    % Claimable tickets, oldest name first, from the live queue if it has
    % any, else from the backfill queue unless a live lease is fresh.
    fromDir = liveDir;
    jobFiles = claimableIn(liveDir);
    liveActive = ~isempty(jobFiles) || liveLeaseActive(leasePath);
    if liveActive
        return;
    end
    fromDir = backfillDir;
    jobFiles = claimableIn(backfillDir);
end

function files = claimableIn(dirPath)
    % Dot-files are claims (.active_*) or in-progress requeues (.requeue_*).
    files = dir(fullfile(dirPath, '*.json'));
    if isempty(files), return; end
    files = files(~startsWith({files.name}, '.'));
    if isempty(files), return; end
    [~, order] = sort({files.name});
    files = files(order);
end

function tf = liveLeaseActive(leasePath)
    % Fresh = rewritten within 60 s (opym.lanes.LEASE_MAX_AGE_S), so a
    % crashed receiver can never park the backfill for longer than that.
    d = dir(leasePath);
    tf = ~isempty(d) && (now - d(1).datenum) * 86400 <= 60;
end

function writeClaim(claimPath, serverId, ticketName, lane)
    % See claims_dir above. Written via a temp file so the supervisor never
    % reads a half-written record. `queue` names the lane directory the
    % ticket was claimed from, so it can be requeued into the same one.
    rec = struct('server_id', serverId, 'ticket', ticketName, 'queue', lane, ...
        'pid', feature('getpid'));
    tmpPath = [claimPath '.tmp'];
    fid = fopen(tmpPath, 'w');
    fprintf(fid, '%s', jsonencode(rec));
    fclose(fid);
    movefile(tmpPath, claimPath, 'f');
end

function writeProfile(profiling_dir, serverId, prof)
    % Best effort: a profiling write must never fail a ticket or the server.
    try
        fid = fopen(fullfile(profiling_dir, sprintf('S%s.jsonl', serverId)), 'a');
        fprintf(fid, '%s\n', jsonencode(prof));
        fclose(fid);
    catch
    end
end

function clearClaim(claimPath)
    if exist(claimPath, 'file')
        delete(claimPath);
    end
end

function val = safelyGetParam(structure, fieldName, defaultValue)
    if isfield(structure, fieldName)
        val = structure.(fieldName);
        if isempty(val)
            val = defaultValue;
        end
    else
        val = defaultValue;
    end
end

% --- HELPER: Map an RL method name onto one PetaKit5D actually implements ---
% PetaKit5D's RLdecon.m dispatches RLMethod through two `switch` statements
% that have NO `otherwise` branch, over {original, simplified, omw, cudagen}.
% `deconvolved` is initialized to [] at the top of that function, so an
% unrecognized name means nothing ever assigns it and an EMPTY volume is
% written -- silently, after background subtraction has already run.
%
% Our historical default, 'simple', is exactly such a name. It never bit
% because the pipeline/pipeline_batch jobTypes go through run_gpu_pipeline.m,
% which dispatches on strcmpi(RLMethod,'omw') itself and never reaches that
% switch. The 'decon' and 'deskew'-with-decon jobTypes DO reach it.
function method = normalizeRLMethod(method)
    valid = {'original', 'simplified', 'omw', 'cudagen'};
    method = lower(char(method));
    if strcmp(method, 'simple')
        method = 'simplified';
    end
    if ~ismember(method, valid)
        error('run_petakit_server:badRLMethod', ...
            ['Unknown rl_method ''%s''. PetaKit5D recognizes {original, ' ...
             'simplified, omw, cudagen} and silently writes an EMPTY volume ' ...
             'for anything else.'], method);
    end
end

% --- HELPER: Forced Flushing Log ---
function logMsg(fmt, varargin)
    % Prints to stdout (1). Under `matlab -batch` (how local_gpu_worker.py
    % launches this) each line reaches the log as it is printed; the
    % pause(0.05) this used to add cost a live ticket ~0.15 s before its job
    % started, for nothing (checked 2026-09-25: a line printed right before a
    % 6 s busy loop was in the log 6 s before the next one).
    fprintf(1, [fmt '\n'], varargin{:});
end
