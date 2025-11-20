%% run_petakit_server.m
% A persistent server that watches a directory for JSON job files.

% --- SYSTEM CONFIGURATION ------------------------------------------------
petakit_source_path = '/cm/shared/apps_local/petakit5d';
base_queue_dir = fullfile(getenv('HOME'), 'petakit_jobs');

envCPUs = getenv('SLURM_CPUS_PER_TASK');
if ~isempty(envCPUs)
    numCPUs = str2double(envCPUs);
    fprintf('[Server] Auto-detected %d CPUs from Slurm environment.\n', numCPUs);
else
    numCPUs = 24;
    fprintf('[Server] No Slurm CPU count found. Using fallback: %d\n', numCPUs);
end

queue_dir = fullfile(base_queue_dir, 'queue');
done_dir  = fullfile(base_queue_dir, 'completed');
fail_dir  = fullfile(base_queue_dir, 'failed');

if ~exist(queue_dir, 'dir'), mkdir(queue_dir); end
if ~exist(done_dir, 'dir'),  mkdir(done_dir); end
if ~exist(fail_dir, 'dir'),  mkdir(fail_dir); end

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
                % --- CROPPING JOB ---
                fprintf('         Type: Crop/Rotate\n');
                outputDir  = safelyGetParam(p, 'output_dir', '');
                baseName   = job.baseName;
                channels   = safelyGetParam(p, 'channels_to_output', []);
                rotate90   = safelyGetParam(p, 'rotate_90', false);
                fileMap    = safelyGetParam(p, 'file_map', []); % NEW: File Map

                dims = p.dims;
                T = dims.T; Z = dims.Z; C = dims.C; Y = dims.Y; X = dims.X;

                topRoiRaw = safelyGetParam(p, 'top_roi', []);
                botRoiRaw = safelyGetParam(p, 'bottom_roi', []);

                topData = []; botData = [];

                if ~isempty(topRoiRaw)
                    if iscell(topRoiRaw), yS = topRoiRaw{1}; xS = topRoiRaw{2};
                    else, yS = topRoiRaw(1,:); xS = topRoiRaw(2,:); end
                    yS = double(yS); xS = double(xS);
                    topData.ys = yS(1)+1; topData.ye = yS(2);
                    topData.xs = xS(1)+1; topData.xe = xS(2);
                end

                if ~isempty(botRoiRaw)
                    if iscell(botRoiRaw), yS = botRoiRaw{1}; xS = botRoiRaw{2};
                    else, yS = botRoiRaw(1,:); xS = botRoiRaw(2,:); end
                    yS = double(yS); xS = double(xS);
                    botData.ys = yS(1)+1; botData.ye = yS(2);
                    botData.xs = xS(1)+1; botData.xe = xS(2);
                end

                if ~exist(outputDir, 'dir'), mkdir(outputDir); end

                if iscell(channels), chanList = cell2mat(channels);
                else, chanList = double(channels); end

                fprintf('         Parfor over %d Timepoints (Multi-file aware)...\n', T);

                runCropParfor(fileMap, outputDir, baseName, chanList, rotate90, T, Z, C, topData, botData);

            case 'decon'
                fprintf('         Type: Deconvolution\n');
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
                fprintf('         Type: Deskew/Rotate\n');
                val_xyPixelSize = safelyGetParam(p, 'xy_pixel_size', 0.136);
                val_dz          = safelyGetParam(p, 'z_step_um', 1.0);
                val_skewAngle   = safelyGetParam(p, 'sheet_angle_deg', 31.8);
                val_deskew      = safelyGetParam(p, 'deskew', true);
                val_rotate      = safelyGetParam(p, 'rotate', true);
                val_objScan     = safelyGetParam(p, 'objective_scan', false);
                val_reverseZ    = safelyGetParam(p, 'reverse_z', false);
                val_dsDir       = safelyGetParam(p, 'ds_dir_name', 'DS');
                val_dsrDir      = safelyGetParam(p, 'dsr_dir_name', 'DSR');
                val_interp      = safelyGetParam(p, 'interp_method', 'cubic');

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
                    'objectiveScan', val_objScan, ...
                    'reverse', val_reverseZ, ...
                    'interpMethod', val_interp, ...
                    'save16bit', true, ...
                    'save3DStack', true, ...
                    'saveMIP', true, ...
                    'FFCorrection', false, ...
                    'BKRemoval', false, ...
                    'parseCluster', false, ...
                    'parseParfor', true, ...
                    'masterCompute', true, ...
                    'largeFile', false, ...
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

% --- HELPERS ---
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

% NEW: Helper to find which file contains the global index
function [fPath, localIdx] = getFileForIndex(globalIdx, fMap)
    % globalIdx is 0-based from MATLAB calculation (t*Z*C + ...)
    % We convert to 1-based for output at the very end if needed,
    % but imread 2nd arg is the "Index", which is 1-based.

    % Iterate through map to find the range
    % fMap is a struct array with fields: path, start, count, end

    for i = 1:length(fMap)
        fStart = fMap(i).start; % 0-based
        fEnd   = fMap(i).end;   % exclusive

        if globalIdx >= fStart && globalIdx < fEnd
            fPath = fMap(i).path;
            % Calculate 1-based index for that specific file
            % Offset = globalIdx - fStart
            % MATLAB imread index = Offset + 1
            localIdx = (globalIdx - fStart) + 1;
            return;
        end
    end

    error('Global index %d out of bounds for mapped files.', globalIdx);
end

function runCropParfor(fileMap, outputDir, baseName, chanList, rotate90, T, Z, C, topData, botData)
    % Convert struct array from JSON to MATLAB struct array if needed
    % Usually jsondecode makes a struct array automatically.

    parfor t = 0:(T-1)
        % Calculate Dimensions
        if ~isempty(topData)
            h = topData.ye - topData.ys + 1;
            w = topData.xe - topData.xs + 1;
        else
            h = botData.ye - botData.ys + 1;
            w = botData.xe - botData.xs + 1;
        end

        if rotate90
            stackSize = [Z, w, h];
        else
            stackSize = [Z, h, w];
        end

        stackMap = containers.Map('KeyType','double','ValueType','any');
        for k = 1:length(chanList)
            stackMap(chanList(k)) = zeros(stackSize, 'uint16');
        end

        for z = 0:(Z-1)
            % Global 0-based indices (interleaved)
            gIdx0 = t*Z*C + z*C + 0;
            gIdx1 = t*Z*C + z*C + 1;

            % Resolve File and Local Index
            [fPath0, lIdx0] = getFileForIndex(gIdx0, fileMap);
            [fPath1, lIdx1] = getFileForIndex(gIdx1, fileMap);

            img0 = imread(fPath0, lIdx0);
            img1 = imread(fPath1, lIdx1);

            if isKey(stackMap, 0) && ~isempty(botData)
                crop = img0(botData.ys:botData.ye, botData.xs:botData.xe);
                if rotate90, crop = rot90(crop); end
                temp = stackMap(0); temp(z+1,:,:) = crop; stackMap(0) = temp;
            end
            if isKey(stackMap, 1) && ~isempty(topData)
                crop = img0(topData.ys:topData.ye, topData.xs:topData.xe);
                if rotate90, crop = rot90(crop); end
                temp = stackMap(1); temp(z+1,:,:) = crop; stackMap(1) = temp;
            end
            if isKey(stackMap, 2) && ~isempty(topData)
                crop = img1(topData.ys:topData.ye, topData.xs:topData.xe);
                if rotate90, crop = rot90(crop); end
                temp = stackMap(2); temp(z+1,:,:) = crop; stackMap(2) = temp;
            end
            if isKey(stackMap, 3) && ~isempty(botData)
                crop = img1(botData.ys:botData.ye, botData.xs:botData.xe);
                if rotate90, crop = rot90(crop); end
                temp = stackMap(3); temp(z+1,:,:) = crop; stackMap(3) = temp;
            end
        end

        keys = stackMap.keys;
        for k = 1:length(keys)
            cIdx = keys{k};
            outName = sprintf('%s_C%d_T%03d.tif', baseName, cIdx, t);
            outPath = fullfile(outputDir, outName);

            dataToWrite = stackMap(cIdx);

            tObj = Tiff(outPath, 'w');
            tagstruct = struct();
            tagstruct.ImageLength = size(dataToWrite, 2);
            tagstruct.ImageWidth = size(dataToWrite, 3);
            tagstruct.Photometric = Tiff.Photometric.MinIsBlack;
            tagstruct.BitsPerSample = 16;
            tagstruct.SamplesPerPixel = 1;
            tagstruct.PlanarConfiguration = Tiff.PlanarConfiguration.Chunky;
            tagstruct.Software = 'MATLAB';

            for zSlice = 1:size(dataToWrite, 1)
                tObj.setTag(tagstruct);
                tObj.write(squeeze(dataToWrite(zSlice,:,:)));
                tObj.writeDirectory();
            end
            tObj.close();
        end
    end
end
