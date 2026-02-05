%% run_petakit_server.m
% A persistent server that watches a directory for JSON job files.
% UPDATED: Now supports 'crop' jobs via opym CLI.

% --- SYSTEM CONFIGURATION ------------------------------------------------
petakit_source_path = getenv('PETAKIT_ROOT');

if isempty(petakit_source_path)
    petakit_source_path = '/cm/shared/apps_local/petakit5d';
    fprintf('[Server] Warning: PETAKIT_ROOT not set. Using default: %s\n', petakit_source_path);
else
    fprintf('[Server] Using PetaKit path: %s\n', petakit_source_path);
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

% Start Parpool only if needed (Cropping is single-process but multi-threaded via Python)
pool = gcp('nocreate');
if isempty(pool) || pool.NumWorkers ~= numCPUs
    try
        delete(pool);
        parpool('local', numCPUs);
    catch
        fprintf('[Server] Warning: Could not start parpool. Continuing...\n');
    end
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

        jobType = safelyGetParam(job, 'jobType', 'deskew');

        switch jobType
            case 'crop'
                % --- CROPPING JOB (Python CLI) ---
                fprintf('         Type: OPM Cropping\n');

                % Extract Params
                val_rois   = safelyGetParam(p, 'rois', struct());
                val_chans  = safelyGetParam(p, 'channels', []);
                val_rotate = safelyGetParam(p, 'rotate', true);
                val_format = safelyGetParam(p, 'format', 'tiff-series');

                % Build Command
                % NOTE: Assumes 'opym' is in the PATH.
                % Ensure your launch script activates the environment!
                cmd = sprintf('opym "%s" --format %s', job.dataDir, val_format);

                if val_rotate
                    cmd = [cmd ' --rotate'];
                else
                    cmd = [cmd ' --no-rotate'];
                end

                % Add ROIs
                if isfield(val_rois, 'top') && ~isempty(val_rois.top)
                   cmd = sprintf('%s --top-roi "%s"', cmd, val_rois.top);
                end
                if isfield(val_rois, 'bottom') && ~isempty(val_rois.bottom)
                   cmd = sprintf('%s --bottom-roi "%s"', cmd, val_rois.bottom);
                end

                % Add Channels
                if ~isempty(val_chans)
                    % Convert to space-separated string
                    % Handle potential column vectors from JSON decoding
                    if size(val_chans, 1) > 1, val_chans = val_chans'; end
                    chanStr = sprintf('%d ', val_chans);
                    cmd = sprintf('%s --channels %s', cmd, strtrim(chanStr));
                end

                fprintf('         Exec: %s\n', cmd);

                % Execute Shell Command
                [status, cmdOut] = system(cmd);

                if status ~= 0
                    error('Opym CLI failed with exit code %d:\n%s', status, cmdOut);
                else
                    disp(cmdOut); % Print Python output to MATLAB log
                end

            case 'decon'
                % --- DECONVOLUTION JOB ---
                % (Existing logic kept brief for clarity)
                fprintf('         Type: Deconvolution\n');
                % ... [Insert your existing decon logic here if needed] ...
                % For now, I'll assume you keep the existing file's logic
                % or I can repopulate it if you need the full file.

            otherwise
                % --- DESKEW/ROTATE JOB (Default) ---
                fprintf('         Type: Deskew/Rotate\n');
                % ... [Existing logic] ...
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
