%% run_petakit_server.m
% A persistent server that watches a directory for JSON job files.
% Usage: matlab -nodisplay -r "run_petakit_server"

% --- SYSTEM CONFIGURATION ------------------------------------------------
petakit_source_path = '/cm/shared/apps_local/petakit5d';

% DYNAMIC PATH: Use the user's home directory + petakit_jobs
base_queue_dir = fullfile(getenv('HOME'), 'petakit_jobs');

% --- DYNAMIC CPU DETECTION -----------------------------------------------
% Check if running under Slurm and get the CPU allocation
envCPUs = getenv('SLURM_CPUS_PER_TASK');
if ~isempty(envCPUs)
    numCPUs = str2double(envCPUs);
    fprintf('[Server] Auto-detected %d CPUs from Slurm environment.\n', numCPUs);
else
    % Fallback for local testing
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
% 1. Load PetaKit
if ~exist('XR_deskew_rotate_data_wrapper', 'file')
    if exist(fullfile(petakit_source_path, 'setup.m'), 'file')
        run(fullfile(petakit_source_path, 'setup.m'));
    else
        error('Could not find PetaKit setup.m at: %s', petakit_source_path);
    end
end

% 2. Configure & Start Persistent Parallel Pool
pool = gcp('nocreate');
if isempty(pool) || pool.NumWorkers ~= numCPUs
    delete(pool);

    % Safety: Ensure the local profile allows this many workers
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
    % 1. Look for JSON files (FIFO order roughly)
    jobFiles = dir(fullfile(queue_dir, '*.json'));

    if isempty(jobFiles)
        pause(2); % Sleep if idle
        continue;
    end

    % 2. Pick the first job
    currentFile = jobFiles(1).name;
    srcPath = fullfile(queue_dir, currentFile);

    fprintf('[Server] >>> Processing job: %s\n', currentFile);

    try
        % 3. Parse JSON Payload
        fid = fopen(srcPath);
        raw = fread(fid, inf);
        fclose(fid);
        job = jsondecode(char(raw'));

        % 4. Extract Parameters with Defaults
        % Access the 'parameters' struct from Python kwargs
        if isfield(job, 'parameters')
            p = job.parameters;
        else
            p = struct();
        end

        % Helper: Get field or default
        getParam = @(s, f, d) iff(isfield(s, f), s.(f), d);

        % Map Python (snake_case) to MATLAB variables
        val_xyPixelSize = getParam(p, 'xy_pixel_size', 0.136);
        val_dz          = getParam(p, 'z_step_um', 1.0);
        val_skewAngle   = getParam(p, 'sheet_angle_deg', 31.8);
        val_deskew      = getParam(p, 'deskew', true);
        val_rotate      = getParam(p, 'rotate', true);

        fprintf('         Data: %s\n', job.dataDir);
        fprintf('         Params: xy=%.3f, dz=%.3f, angle=%.2f\n', ...
                val_xyPixelSize, val_dz, val_skewAngle);

        % 5. Execute Processing
        XR_deskew_rotate_data_wrapper( ...
            {job.dataDir}, ...
            'DSDirName', 'DS', ...
            'DSRDirName', 'DSR', ...
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
            'parseCluster', false, ... % Do NOT use Slurm
            'parseParfor', true, ...   % Use OUR persistent pool
            'masterCompute', true, ... % Run on this node
            'largeFile', false, ...    % Parallelize over file list
            'cpusPerTask', numCPUs ... % Pass the specific CPU count
        );

        % 6. Success: Move to Completed
        movefile(srcPath, fullfile(done_dir, currentFile));
        fprintf('[Server] <<< Finished: %s\n', currentFile);

    catch ME
        % 7. Failure: Move to Failed and Log Error
        fprintf('[Server] !!! ERROR on %s: %s\n', currentFile, ME.message);
        movefile(srcPath, fullfile(fail_dir, currentFile));

        errLog = fullfile(fail_dir, [currentFile '.log']);
        fid = fopen(errLog, 'w');
        fprintf(fid, '%s\n', getReport(ME));
        fclose(fid);
    end
end

% --- HELPER FUNCTION ---
function val = iff(condition, trueVal, falseVal)
    if condition
        val = trueVal;
    else
        val = falseVal;
    end
end
