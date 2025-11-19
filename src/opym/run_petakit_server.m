%% run_petakit_server.m
% A persistent server that watches a directory for JSON job files.
% Usage: matlab -nodisplay -r "run_petakit_server"

% --- SYSTEM CONFIGURATION ------------------------------------------------
petakit_source_path = '/cm/shared/apps_local/petakit5d';
base_queue_dir = fullfile(getenv('HOME'), 'petakit_jobs');

% --- DYNAMIC CPU DETECTION -----------------------------------------------
envCPUs = getenv('SLURM_CPUS_PER_TASK');
if ~isempty(envCPUs)
    numCPUs = str2double(envCPUs);
    fprintf('[Server] Auto-detected %d CPUs from Slurm environment.\n', numCPUs);
else
    numCPUs = 24;
    fprintf('[Server] No Slurm CPU count found. Using fallback: %d\n', numCPUs);
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
        error('Could not find PetaKit setup.m at: %s', petakit_source_path);
    end
end

pool = gcp('nocreate');
if isempty(pool) || pool.NumWorkers ~= numCPUs
    delete(pool);
    c = parcluster('local');
    if c.NumWorkers < numCPUs
        c.NumWorkers = numCPUs;
        saveProfile(c);
    end
    fprintf('[Server] Starting persistent parpool with %d workers...\n', numCPUs);
    parpool('local', numCPUs);
else
    fprintf('[Server] Using existing parpool with %d workers.\n', pool.NumWorkers);
end

fprintf('[Server] Ready. Watching: %s\n', queue_dir);

% --- MAIN SERVER LOOP ----------------------------------------------------
while true
    jobFiles = dir(fullfile(queue_dir, '*.json'));

    if isempty(jobFiles)
        pause(2);
        continue;
    end

    currentFile = jobFiles(1).name;
    srcPath = fullfile(queue_dir, currentFile);

    fprintf('[Server] >>> Processing job: %s\n', currentFile);

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

        % --- EXTRACT PARAMETERS ---
        val_xyPixelSize = safelyGetParam(p, 'xy_pixel_size', 0.136);
        val_dz          = safelyGetParam(p, 'z_step_um', 1.0);
        val_skewAngle   = safelyGetParam(p, 'sheet_angle_deg', 31.8);
        val_deskew      = safelyGetParam(p, 'deskew', true);
        val_rotate      = safelyGetParam(p, 'rotate', true);

        % FIX: Get dynamic directory names (default to DS/DSR if missing)
        val_dsDir       = safelyGetParam(p, 'ds_dir_name', 'DS');
        val_dsrDir      = safelyGetParam(p, 'dsr_dir_name', 'DSR');
        % --------------------------

        fprintf('         Data: %s\n', job.dataDir);
        fprintf('         Out:  %s\n', val_dsrDir);
        fprintf('         Params: xy=%.3f, dz=%.3f, angle=%.2f\n', ...
                val_xyPixelSize, val_dz, val_skewAngle);

        XR_deskew_rotate_data_wrapper( ...
            {job.dataDir}, ...
            'DSDirName', val_dsDir, ...   % <-- Use dynamic name
            'DSRDirName', val_dsrDir, ... % <-- Use dynamic name
            'channelPatterns', {job.baseName}, ...
            'deskew', val_deskew, ...
            'rotate', val_rotate, ...
            'xyPixelSize', val_xyPixelSize, ...
            'dz', val_dz, ...
            'skewAngle', val_skewAngle, ...
            'objectiveScan', false, ...
            'reverse', false, ...
            'save16bit', true, ...
            'save3DStack', true, ...
            'saveMIP', true, ...
            'interpMethod', 'linear', ...
            'FFCorrection', false, ...
            'BKRemoval', false, ...
            'parseCluster', false, ...
            'parseParfor', true, ...
            'masterCompute', true, ...
            'largeFile', false, ...
            'cpusPerTask', numCPUs ...
        );

        movefile(srcPath, fullfile(done_dir, currentFile));
        fprintf('[Server] <<< Finished: %s\n', currentFile);

    catch ME
        fprintf('[Server] !!! ERROR on %s: %s\n', currentFile, ME.message);
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
