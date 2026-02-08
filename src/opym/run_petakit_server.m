%% run_petakit_server.m
% A persistent server that watches a directory for JSON job files.
% FEATURES:
%   - Intelligently dispatches BigTiff vs Standard jobs.
%   - Forces log flushing for real-time monitoring.

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

base_queue_dir = fullfile(getenv('HOME'), 'petakit_jobs');

% --- DYNAMIC CPU DETECTION -----------------------------------------------
envCPUs = getenv('SLURM_CPUS_PER_TASK');
if ~isempty(envCPUs)
    numCPUs = str2double(envCPUs);
else
    numCPUs = 24;
end

% Setup Directories
queue_dir = fullfile(base_queue_dir, 'queue');
done_dir  = fullfile(base_queue_dir, 'completed');
fail_dir  = fullfile(base_queue_dir, 'failed');

if ~exist(queue_dir, 'dir'), mkdir(queue_dir); end
if ~exist(done_dir, 'dir'),  mkdir(done_dir); end
if ~exist(fail_dir, 'dir'),  mkdir(fail_dir); end

% --- INITIALIZATION ------------------------------------------------------
if ~exist('XR_deskew_rotate_data_wrapper', 'file')
    if exist(fullfile(petakit_source_path, 'setup.m'), 'file')
        run(fullfile(petakit_source_path, 'setup.m'));
    else
        warning('Could not find PetaKit setup.m. Decon/Deskew may fail.');
    end
end

pool = gcp('nocreate');
if isempty(pool) || pool.NumWorkers ~= numCPUs
    try
        delete(pool);
        parpool('local', numCPUs);
    catch
        logMsg('[Server] Warning: Could not start parpool. Continuing...');
    end
end

logMsg('[Server] Ready. Watching: %s', queue_dir);

% --- MAIN SERVER LOOP ----------------------------------------------------
while true
    jobFiles = dir(fullfile(queue_dir, '*.json'));

    if isempty(jobFiles)
        pause(2);
        continue;
    end

    currentFile = jobFiles(1).name;
    srcPath = fullfile(queue_dir, currentFile);

    logMsg('[Server] >>> Processing job: %s', currentFile);

    try
        fid = fopen(srcPath);
        raw = fread(fid, inf);
        fclose(fid);
        job = jsondecode(char(raw'));

        if isfield(job, 'parameters')
            p = job.parameters;
        else
            p = struct();
        end

        jobType = safelyGetParam(job, 'jobType', 'deskew');

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

            case 'decon'
                % --- DECONVOLUTION JOB ---
                logMsg('         Type: Deconvolution');
                val_resDir = safelyGetParam(p, 'result_dir_name', 'decon');
                val_chans  = safelyGetParam(p, 'channel_patterns', {});
                val_psfs   = safelyGetParam(p, 'psf_paths', {});
                val_iter   = safelyGetParam(p, 'iterations', 10);
                val_gpu    = safelyGetParam(p, 'gpu_job', true);
                val_skewed = safelyGetParam(p, 'skewed', true);
                val_method = safelyGetParam(p, 'rl_method', 'simplified');
                val_16bit  = safelyGetParam(p, 'save_16bit', true);

                if isstring(val_chans), val_chans = cellstr(val_chans); end
                if isstring(val_psfs), val_psfs = cellstr(val_psfs); end

                XR_decon_data_wrapper( ...
                    {job.dataDir}, ...
                    'resultDirName', val_resDir, ...
                    'channelPatterns', val_chans, ...
                    'psfFullpaths', val_psfs, ...
                    'deconIter', val_iter, ...
                    'GPUJob', val_gpu, ...
                    'skewed', val_skewed, ...
                    'RLMethod', val_method, ...
                    'save16bit', val_16bit, ...
                    'parseCluster', false, ...
                    'parseParfor', true, ...
                    'masterCompute', true, ...
                    'cpusPerTask', numCPUs ...
                );

            otherwise
                % --- DESKEW/ROTATE JOB ---
                logMsg('         Type: Deskew/Rotate');
                val_xyPixelSize = safelyGetParam(p, 'xy_pixel_size', 0.136);
                val_dz          = safelyGetParam(p, 'z_step_um', 1.0);
                val_skewAngle   = safelyGetParam(p, 'sheet_angle_deg', 31.8);
                val_deskew      = safelyGetParam(p, 'deskew', true);
                val_rotate      = safelyGetParam(p, 'rotate', true);
                val_interp      = safelyGetParam(p, 'interp_method', 'cubic');
                val_dsDir       = safelyGetParam(p, 'ds_dir_name', 'DS');
                val_dsrDir      = safelyGetParam(p, 'dsr_dir_name', 'DSR');

                XR_deskew_rotate_data_wrapper( ...
                    {job.dataDir}, ...
                    'DSDirName', val_dsDir, ...
                    'DSRDirName', val_dsrDir, ...
                    'channelPatterns', {job.baseName}, ...
                    'deskew', val_deskew, ...
                    'rotate', val_rotate, ...
                    'xyPixelSize', val_xyPixelSize, ...
                    'dz', val_dz, ...
                    'skewAngle', val_skewAngle, ...
                    'interpMethod', val_interp, ...
                    'save16bit', true, ...
                    'save3DStack', true, ...
                    'saveMIP', true, ...
                    'parseCluster', false, ...
                    'parseParfor', true, ...
                    'masterCompute', true, ...
                    'cpusPerTask', numCPUs ...
                );
        end

        movefile(srcPath, fullfile(done_dir, currentFile));
        logMsg('[Server] <<< Finished: %s', currentFile);

    catch ME
        logMsg('[Server] !!! ERROR on %s: %s', currentFile, ME.message);
        movefile(srcPath, fullfile(fail_dir, currentFile));
        errLog = fullfile(fail_dir, [currentFile '.log']);
        fid = fopen(errLog, 'w');
        fprintf(fid, '%s\n', getReport(ME));
        fclose(fid);
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

% --- HELPER: Forced Flushing Log ---
function logMsg(fmt, varargin)
    % Prints to stdout (1) and pauses briefly to force buffer flush
    fprintf(1, [fmt '\n'], varargin{:});
    pause(0.05);
end
