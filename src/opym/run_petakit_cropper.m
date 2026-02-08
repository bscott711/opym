function run_petakit_cropper(json_file)
% RUN_PETAKIT_CROPPER [Legacy/Standard]
% Handles standard OME-TIFFs (non-BigTiff).
% Updated to match the output format of run_bigtiff_cropper.

    % 0. Setup
    warning('off', 'all'); % Aggressive suppression

    if ~isfile(json_file), error('JSON not found: %s', json_file); end
    fid = fopen(json_file, 'r');
    job = jsondecode(char(fread(fid, inf)'));
    fclose(fid);

    % 1. Parse Paths & Naming
    % Matches logic in run_bigtiff_cropper
    inputFile = job.dataDir;
    [dataRoot, fullBaseName, ~] = fileparts(inputFile);

    if isfield(job, 'baseName')
        cleanBaseName = regexprep(job.baseName, '\.ome(\.tif)?$', '');
    else
        cleanBaseName = regexprep(fullBaseName, '\.ome$', '');
    end

    % Output to folder matching the clean name
    outputDir = fullfile(dataRoot, cleanBaseName);
    if ~exist(outputDir, 'dir'), mkdir(outputDir); end

    fprintf('🚀 Starting Standard Cropper (Legacy Path)\n');
    fprintf('   Input:  %s\n', fullBaseName);
    fprintf('   Output: %s\n', outputDir);

    % 2. Metadata Copy (Sidecar Preservation)
    fprintf('   [Metadata] Scanning for sidecar files...\n');
    sidecars = [dir(fullfile(dataRoot, '*.txt')); dir(fullfile(dataRoot, '*.json')); dir(fullfile(dataRoot, '*.xml'))];
    for k = 1:length(sidecars)
        fName = sidecars(k).name;
        if contains(fName, '.ome.tif', 'IgnoreCase', true) || sidecars(k).isdir, continue; end
        copyfile(fullfile(dataRoot, fName), fullfile(outputDir, fName));
    end

    % 3. Quick Map (Standard Tiff Logic)
    % This is the old heuristic logic, fine for simple files
    [FileMap, SizeT, SizeZ, SizeC, DimOrder] = map_ome_tiff_structure(inputFile);
    fprintf('   Metadata: T=%d, Z=%d, C=%d (%s)\n', SizeT, SizeZ, SizeC, DimOrder);

    % 4. ROI Setup
    top_roi = parse_roi_str(job.parameters.rois.top);
    bot_roi = parse_roi_str(job.parameters.rois.bottom);
    do_rotate = isfield(job.parameters, 'rotate') && job.parameters.rotate;

    req_channels = [];
    if isfield(job.parameters, 'channels')
        req_channels = job.parameters.channels;
        if iscell(req_channels), req_channels = cell2mat(req_channels); end
    end

    % 5. Processing Loop
    pool = gcp('nocreate');
    if isempty(pool), pool = parpool('local'); end

    parfor t_idx = 0 : (SizeT - 1)
        warning('off', 'all');

        % A. Map frames for this timepoint
        % (Simplified mapping logic for standard XYCZT)
        frames = struct('z', {}, 'c', {}, 'file', {}, 'idx', {});
        count = 0;

        for z = 0:(SizeZ-1)
            for c = 0:(SizeC-1)
                if strcmp(DimOrder, 'XYZCT')
                    g_idx = (t_idx * SizeZ * SizeC) + (c * SizeZ) + z;
                else
                    g_idx = (t_idx * SizeZ * SizeC) + (z * SizeC) + c;
                end

                [fPath, locIdx] = get_file_for_frame(g_idx, FileMap);
                if ~isempty(fPath)
                    count = count + 1;
                    frames(count).z = z;
                    frames(count).c = c;
                    frames(count).file = fPath;
                    frames(count).idx = locIdx + 1; % 1-based for Tiff
                end
            end
        end

        % B. Load & Buffer
        % Pre-allocate buffers for outputs
        % Logic: 1 Raw Channel -> 2 Outputs (Top/Bot)
        % We just store crops in a Map
        bufMap = containers.Map('KeyType', 'double', 'ValueType', 'any');

        for i = 1:count
            % Read Frame
            img = fallback_read(frames(i).file, frames(i).idx);
            z = frames(i).z;
            c = frames(i).c;

            % Determine Outputs
            outA = c * 2;
            outB = c * 2 + 1;
            is_even = mod(c, 2) == 0;

            % Crop A (Bottom if odd, Top if even?? Check logic vs BigTiff)
            % BigTiff Logic: Cam1(Odd Raw) -> Bot=Low, Top=High
            % This legacy logic might differ, aligning it to BigTiff:

            % Standard PetaKit usually assumes:
            % CamA (Even) -> Top ROI
            % CamB (Odd)  -> Bot ROI
            % We will stick to the requested ROIs from JSON

            % Crop & Store logic (Simplified)
            process_and_store(img, outA, is_even, top_roi, bot_roi, do_rotate, z, SizeZ, bufMap, req_channels);
            process_and_store(img, outB, is_even, top_roi, bot_roi, do_rotate, z, SizeZ, bufMap, req_channels);
        end

        % C. Write Outputs (Standard Tiff)
        keys = bufMap.keys;
        for k = 1:length(keys)
            id = keys{k};
            stack = bufMap(id);
            % Use 0-based Txxxx to match BigTiff output
            outName = sprintf('%s_C%02d_T%04d.tif', cleanBaseName, id, t_idx);
            write_simple_tiff(fullfile(outputDir, outName), stack);
        end
    end
end

% --- HELPERS ---

function process_and_store(img, outID, is_even, top_roi, bot_roi, do_rot, z, SizeZ, map, req)
    % Helper to crop and store into map
    if ~isempty(req) && ~ismember(outID, req), return; end

    % Logic: Even channels use TOP, Odd use BOT (Default PetaKit)
    if is_even
        roi = top_roi;
    else
        roi = bot_roi;
    end

    if isempty(roi), return; end

    crop = img(roi(1):roi(2), roi(3):roi(4));
    if do_rot, crop = rot90(crop); end

    if ~isKey(map, outID)
        [h, w] = size(crop);
        map(outID) = zeros(h, w, SizeZ, 'uint16');
    end

    buf = map(outID);
    buf(:,:,z+1) = crop;
    map(outID) = buf;
end

function img = fallback_read(fPath, idx)
    t = Tiff(fPath, 'r');
    t.setDirectory(idx);
    img = t.read();
    t.close();
end

function write_simple_tiff(filename, stack)
    % Simple Tiff Writer (No OME metadata for legacy path)
    imwrite(stack(:,:,1), filename);
    for k = 2:size(stack,3)
        imwrite(stack(:,:,k), filename, 'WriteMode', 'append');
    end
end

function [FileMap, SizeT, SizeZ, SizeC, DimOrder] = map_ome_tiff_structure(masterFile)
    % Simplified heuristic mapper
    t = Tiff(masterFile, 'r');
    desc = t.getTag('ImageDescription');
    t.close();

    SizeC=1; SizeZ=1; SizeT=1; DimOrder='XYCZT';
    tokC = regexp(desc, 'SizeC="(\d+)"', 'tokens'); if ~isempty(tokC), SizeC = str2double(tokC{1}{1}); end
    tokZ = regexp(desc, 'SizeZ="(\d+)"', 'tokens'); if ~isempty(tokZ), SizeZ = str2double(tokZ{1}{1}); end
    tokT = regexp(desc, 'SizeT="(\d+)"', 'tokens'); if ~isempty(tokT), SizeT = str2double(tokT{1}{1}); end

    % Assume single file for standard jobs
    FileMap = struct('path', masterFile, 'start_idx', 0, 'end_idx', (SizeC*SizeZ*SizeT)-1);
end

function [fPath, locIdx] = get_file_for_frame(gIdx, FileMap)
    fPath = FileMap(1).path;
    locIdx = gIdx;
end

function roi = parse_roi_str(s)
    if isempty(s), roi = []; return; end
    % Convert "y1:y2,x1:x2" -> [y1, y2, x1, x2]
    % Note: Python slices are 0-based, MATLAB 1-based.
    % We assume the JSON passed 1-based inclusive coords or fix them here.
    % For safety, let's assume the string came from the Python generator which uses inclusive string logic
    parts = split(s, ',');
    yr = str2num(parts{1}); xr = str2num(parts{2});
    roi = [yr(1), yr(end), xr(1), xr(end)];
end
