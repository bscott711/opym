%% run_petakit_server.m
% A persistent server that watches a directory for JSON job files.

% --- SYSTEM CONFIGURATION ------------------------------------------------
% Try to get path from environment (passed by launch script)
petakit_source_path = getenv('PETAKIT_ROOT');
if isempty(petakit_source_path)
    petakit_source_path = '/cm/shared/apps_local/petakit5d';
end

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
                fprintf('         Type: Crop/Rotate\n');
                outputDir  = safelyGetParam(p, 'output_dir', '');
                baseName   = job.baseName;
                channels   = safelyGetParam(p, 'channels_to_output', []);
                rotate90   = safelyGetParam(p, 'rotate_90', false);
                fileMap    = safelyGetParam(p, 'file_map', []);

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

                fprintf('         Parfor over %d Timepoints (Multi-file Optimized)...\n', T);

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

function [fPath, localIdx] = getFileForIndex(globalIdx, fMap)
    % Lookup which file contains the global 0-based index
    for i = 1:length(fMap)
        fStart = fMap(i).start;
        fEnd   = fMap(i).end;

        if globalIdx >= fStart && globalIdx < fEnd
            fPath = fMap(i).path;
            localIdx = (globalIdx - fStart) + 1; % 1-based for MATLAB
            return;
        end
    end
    error('Global index %d out of bounds.', globalIdx);
end

function runCropParfor(fileMap, outputDir, baseName, chanList, rotate90, T, Z, C, topData, botData)
    % Parallel loop over timepoints
    parfor t = 0:(T-1)

        % 1. Setup Output Buffers
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

        % 2. Efficient Read Loop (Minimizing File Opens)
        currentFile = '';
        tIn = [];

        try
            for z = 0:(Z-1)
                % We need to read C0 and C1 for this Z-slice
                img0 = []; img1 = [];

                for c = 0:1 % Assuming 2 cameras (0 and 1)
                    gIdx = t*Z*C + z*C + c;

                    [fPath, lIdx] = getFileForIndex(gIdx, fileMap);

                    % If file changed, open new Tiff object
                    if ~strcmp(fPath, currentFile)
                        if ~isempty(tIn), tIn.close(); end
                        tIn = Tiff(fPath, 'r');
                        currentFile = fPath;
                    end

                    tIn.setDirectory(lIdx);
                    img = tIn.read();

                    if c == 0, img0 = img; else, img1 = img; end
                end

                % 3. Crop Logic (In Memory)
                % Map: 0=BotC0, 1=TopC0, 2=TopC1, 3=BotC1
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
        catch ME
            if ~isempty(tIn), tIn.close(); end
            rethrow(ME);
        end

        if ~isempty(tIn), tIn.close(); end

        % 4. Write Output (Efficiently)
        keys = stackMap.keys;
        for k = 1:length(keys)
            cIdx = keys{k};
            outName = sprintf('%s_C%d_T%03d.tif', baseName, cIdx, t);
            outPath = fullfile(outputDir, outName);

            dataToWrite = stackMap(cIdx);

            tObj = Tiff(outPath, 'w');

            % Re-create struct locally to avoid parfor classification error
            ts = struct();
            ts.ImageLength = size(dataToWrite, 2);
            ts.ImageWidth = size(dataToWrite, 3);
            ts.Photometric = Tiff.Photometric.MinIsBlack;
            ts.BitsPerSample = 16;
            ts.SamplesPerPixel = 1;
            ts.PlanarConfiguration = Tiff.PlanarConfiguration.Chunky;
            ts.Software = 'MATLAB';

            for zSlice = 1:size(dataToWrite, 1)
                tObj.setTag(ts);
                tObj.write(squeeze(dataToWrite(zSlice,:,:)));
                tObj.writeDirectory();
            end
            tObj.close();
        end
    end
end
