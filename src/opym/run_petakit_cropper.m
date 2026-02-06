function run_petakit_cropper(json_file)
% RUN_PETAKIT_CROPPER High-performance parallel cropping for PetaKit5D
%   Refactored to handle Micro-Manager "Split" OME-TIFFs (multiple 4GB files).
%   Output: 3D Stacks [Base]_C[c]_T[t].tif

    % 0. Suppress annoying TIFF warnings globally
    warning('off', 'MATLAB:imagesci:Tiff:libraryWarning');
    warning('off', 'MATLAB:tifflib:TIFFReadDirectory:libraryWarning');

    % 1. Parse Job JSON
    if ~isfile(json_file), error('JSON not found: %s', json_file); end

    fid = fopen(json_file, 'r');
    str = char(fread(fid, inf)');
    fclose(fid);
    job = jsondecode(str);

    inputFile = job.dataDir;

    % --- FIX 1: Directory Naming (Strip .ome extension if present) ---
    [dataRoot, fullBaseName, ~] = fileparts(inputFile);
    cleanBaseName = regexprep(fullBaseName, '\.ome$', '');
    outputDir = fullfile(dataRoot, [cleanBaseName, '_cropped']);

    if ~exist(outputDir, 'dir'), mkdir(outputDir); end

    fprintf('🚀 Starting Parallel Crop (Split-File Support)\n');
    fprintf('   Input Base: %s\n', fullBaseName);
    fprintf('   Output:     %s\n', outputDir);

    % 2. Parse Parameters
    top_roi = parse_roi_str(job.parameters.rois.top);
    bot_roi = parse_roi_str(job.parameters.rois.bottom);
    do_rotate = isfield(job.parameters, 'rotate') && job.parameters.rotate;

    req_channels = [];
    if isfield(job.parameters, 'channels')
        req_channels = job.parameters.channels;
    end

    % --- NEW: Map Multi-Part Files ---
    % Micro-Manager splits files into Pos0.ome.tif, Pos0_1.ome.tif, etc.
    % We must map the global frame index to the specific file and local index.
    FileMap = map_multipage_tiff_files(inputFile);

    % Total physical frames available across all files
    total_physical_frames = FileMap(end).end_idx + 1;

    % 3. Read Metadata (SizeZ, SizeC, SizeT) from Master File
    % The first file usually contains the OME-XML for the whole dataset
    t = Tiff(inputFile, 'r');
    SizeC = 1; SizeZ = 1; SizeT = 1;

    try
        desc = t.getTag('ImageDescription');
        tokC = regexp(desc, 'SizeC="(\d+)"', 'tokens');
        tokZ = regexp(desc, 'SizeZ="(\d+)"', 'tokens');
        tokT = regexp(desc, 'SizeT="(\d+)"', 'tokens');

        if ~isempty(tokC), SizeC = str2double(tokC{1}{1}); end
        if ~isempty(tokZ), SizeZ = str2double(tokZ{1}{1}); end
        if ~isempty(tokT), SizeT = str2double(tokT{1}{1}); end
    catch
        warning('Failed to parse OME-XML. Calculations may be wrong.');
    end
    t.close();

    fprintf('📊 Metadata: T=%d, Z=%d, C=%d (Expected: %d frames)\n', SizeT, SizeZ, SizeC, SizeT*SizeZ*SizeC);
    fprintf('📊 Physical: %d frames found across %d files.\n', total_physical_frames, numel(FileMap));

    if total_physical_frames < (SizeT * SizeZ * SizeC)
        fprintf('⚠️ WARNING: Dataset seems truncated. Missing %d frames.\n', (SizeT*SizeZ*SizeC) - total_physical_frames);
    end

    % Write Processing Log
    logStruct = struct();
    logStruct.original_file = inputFile;
    logStruct.timestamp = char(datetime('now'));
    logStruct.dimensions = struct('SizeT', SizeT, 'SizeZ', SizeZ, 'SizeC', SizeC);
    logStruct.actual_frames = total_physical_frames;
    logStruct.rois = job.parameters.rois;
    logStruct.rotate = do_rotate;
    logStruct.channel_mapping = 'C0,3=Bot; C1,2=Top';

    jsonTxt = jsonencode(logStruct, 'PrettyPrint', true);
    logPath = fullfile(outputDir, [cleanBaseName, '_processing_log.json']);
    fid = fopen(logPath, 'w');
    fprintf(fid, '%s', jsonTxt);
    fclose(fid);
    fprintf('📝 Log written: %s\n', logPath);

    % 4. Build Job List (Flattened T * C)
    jobs = [];
    for t_idx = 0 : (SizeT - 1)
        for c_idx = 0 : (SizeC - 1)
            if ~isempty(req_channels) && ~ismember(c_idx, req_channels)
                continue;
            end
            jobs = [jobs; t_idx, c_idx]; %#ok<AGROW>
        end
    end

    num_jobs = size(jobs, 1);
    fprintf('🔥 Processing %d 3D Stacks on %d workers...\n', num_jobs, gcp('nocreate').NumWorkers);

    % 5. Execute Parallel 3D Stacks
    parfor i = 1:num_jobs
        warning('off', 'MATLAB:imagesci:Tiff:libraryWarning');
        warning('off', 'MATLAB:tifflib:TIFFReadDirectory:libraryWarning');

        t0 = jobs(i, 1);
        c0 = jobs(i, 2);

        % Channel Logic
        is_top = ismember(mod(c0, 4), [1, 2]);
        is_bot = ismember(mod(c0, 4), [0, 3]);

        target_roi = [];
        suffix = '';

        if is_top && ~isempty(top_roi)
            target_roi = top_roi;
            suffix = 'top';
        elseif is_bot && ~isempty(bot_roi)
            target_roi = bot_roi;
            suffix = 'bot';
        else
            continue;
        end

        % Prepare 3D Stack Buffer
        h = target_roi(2) - target_roi(1) + 1;
        w = target_roi(4) - target_roi(3) + 1;
        if do_rotate
            stack = zeros(w, h, SizeZ, 'uint16');
        else
            stack = zeros(h, w, SizeZ, 'uint16');
        end

        try
            % We keep track of the currently open Tiff to minimize fopen/fclose
            current_file_path = '';
            lt = [];

            for z0 = 0 : (SizeZ - 1)
                % 1. Global Index (0-based)
                global_idx = (t0 * SizeZ * SizeC) + (z0 * SizeC) + c0;

                % 2. Resolve File and Local Index
                [fPath, locIdx] = get_file_for_frame(global_idx, FileMap);

                if isempty(fPath)
                    continue; % Frame missing (truncated)
                end

                % 3. Switch file if needed
                if ~strcmp(fPath, current_file_path)
                    if ~isempty(lt), lt.close(); end
                    lt = Tiff(fPath, 'r');
                    current_file_path = fPath;
                end

                % 4. Read (locIdx is 0-based from mapping, Tiff needs 1-based)
                lt.setDirectory(locIdx + 1);
                img = lt.read();

                % 5. Crop
                crop = img(target_roi(1):target_roi(2), target_roi(3):target_roi(4));
                if do_rotate
                    crop = rot90(crop);
                end

                stack(:, :, z0 + 1) = crop;
            end

            if ~isempty(lt), lt.close(); end

            % Write Output
            outName = sprintf('%s_%s_C%d_T%03d.tif', cleanBaseName, suffix, c0, t0);
            outPath = fullfile(outputDir, outName);
            write_3d_tiff(outPath, stack);

        catch ME
            fprintf('❌ Error T=%d C=%d: %s\n', t0, c0, ME.message);
            if ~isempty(lt), close(lt); end
        end
    end

    fprintf('✅ Parallel Crop Complete.\n');
end

% --- HELPER FUNCTIONS ---

function FileMap = map_multipage_tiff_files(masterFile)
    % Scans directory for split files (Pos0.ome.tif, Pos0_1.ome.tif...)
    % Returns a struct array mapping global indices to files.

    [fDir, fName, fExt] = fileparts(masterFile);

    % 1. Identify Base Pattern
    % Expected: "Name.ome.tif" or "Name_MMStack_Pos0.ome.tif"
    % Split files: "Name_1.ome.tif", "Name_2.ome.tif"

    % Clean the ".ome" part for regex matching
    basePattern = regexprep(fName, '\.ome$', '');

    d = dir(fullfile(fDir, ['*.tif'])); % Broad search first

    fileList = {};
    indices = [];

    % Regex to find siblings: baseName_(\d+).ome.tif
    % Note: The master file usually has NO number, or we treat it as 0.

    % Escape special regex chars in filename
    safeBase = regexptranslate('escape', basePattern);

    % Pattern 1: The Master File
    patMaster = ['^' safeBase '\.ome\.tif$'];
    % Pattern 2: Sibling Files
    patSibling = ['^' safeBase '_(\d+)\.ome\.tif$'];

    for k = 1:length(d)
        thisName = d(k).name;

        % Check if master
        if ~isempty(regexp(thisName, patMaster, 'once'))
            fileList{end+1} = fullfile(d(k).folder, thisName);
            indices(end+1) = 0; % 0 sort order
            continue;
        end

        % Check if sibling
        tok = regexp(thisName, patSibling, 'tokens');
        if ~isempty(tok)
            idx = str2double(tok{1}{1});
            fileList{end+1} = fullfile(d(k).folder, thisName);
            indices(end+1) = idx;
        end
    end

    % Sort by index
    [~, sortIdx] = sort(indices);
    sortedFiles = fileList(sortIdx);

    % Build Map (Expensive but done once)
    FileMap = struct('path', {}, 'start_idx', {}, 'end_idx', {});

    current_global_start = 0;

    fprintf('   Mapping files...\n');
    for k = 1:length(sortedFiles)
        fPath = sortedFiles{k};

        % Count frames
        t = Tiff(fPath, 'r');
        num_frames = 0;
        while true
            num_frames = num_frames + 1;
            if t.lastDirectory(), break; end
            t.nextDirectory();
        end
        t.close();

        FileMap(k).path = fPath;
        FileMap(k).start_idx = current_global_start;
        FileMap(k).end_idx = current_global_start + num_frames - 1;

        [~, n, e] = fileparts(fPath);
        fprintf('     [%d] %s%s -> Frames %d to %d\n', k, n, e, FileMap(k).start_idx, FileMap(k).end_idx);

        current_global_start = current_global_start + num_frames;
    end
end

function [fPath, locIdx] = get_file_for_frame(globalIdx, FileMap)
    % Binary search or simple scan to find which file contains the frame
    fPath = '';
    locIdx = 0;

    % Optimization: It's likely the same file as before or the next one.
    % Simple loop is fast enough for <50 files.
    for k = 1:length(FileMap)
        if globalIdx >= FileMap(k).start_idx && globalIdx <= FileMap(k).end_idx
            fPath = FileMap(k).path;
            locIdx = globalIdx - FileMap(k).start_idx;
            return;
        end
    end
end

function write_3d_tiff(filename, stack)
    imwrite(stack(:,:,1), filename);
    num_z = size(stack, 3);
    if num_z > 1
        for k = 2:num_z
            imwrite(stack(:,:,k), filename, 'WriteMode', 'append');
        end
    end
end

function roi = parse_roi_str(s)
    if isempty(s) || strcmp(s, 'null'), roi = []; return; end
    clean = replace(s, {':', ','}, ' ');
    vals = sscanf(clean, '%d %d %d %d');
    if numel(vals) == 4
        roi = [vals(1)+1, vals(2), vals(3)+1, vals(4)];
    else
        roi = [];
    end
end
