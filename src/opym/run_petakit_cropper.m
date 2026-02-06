function run_petakit_cropper(json_file)
% RUN_PETAKIT_CROPPER High-performance parallel cropping for PetaKit5D
%   Refactored to handle Micro-Manager "Split" OME-TIFFs (multiple 4GB files).
%   Output: 3D Stacks [Base]_C[c]_T[t].tif matching notebook channel logic.

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

    % --- Map Multi-Part Files ---
    FileMap = map_multipage_tiff_files(inputFile);

    if isempty(FileMap)
        error('❌ CRITICAL: No matching TIFF files found for %s', inputFile);
    end

    % Total physical frames available across all files
    total_physical_frames = FileMap(end).end_idx + 1;

    % 3. Read Metadata (SizeZ, SizeC, SizeT)
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

    fprintf('📊 Metadata: T=%d, Z=%d, C=%d (Total Expected: %d)\n', SizeT, SizeZ, SizeC, SizeT*SizeZ*SizeC);

    if total_physical_frames < (SizeT * SizeZ * SizeC)
        fprintf('⚠️ WARNING: Dataset truncated. Missing %d frames.\n', (SizeT*SizeZ*SizeC) - total_physical_frames);
    end

    % Write Processing Log
    logStruct = struct();
    logStruct.original_file = inputFile;
    logStruct.timestamp = char(datetime('now'));
    logStruct.dimensions = struct('SizeT', SizeT, 'SizeZ', SizeZ, 'SizeC', SizeC);
    logStruct.actual_frames = total_physical_frames;
    logStruct.rois = job.parameters.rois;
    logStruct.rotate = do_rotate;
    logStruct.channel_mapping = 'Even Input(0,2): C(2n)=Bot, C(2n+1)=Top; Odd Input(1,3): C(2n)=Top, C(2n+1)=Bot';

    fid = fopen(fullfile(outputDir, [cleanBaseName, '_processing_log.json']), 'w');
    fprintf(fid, '%s', jsonencode(logStruct, 'PrettyPrint', true));
    fclose(fid);

    % 4. Build Job List (Flattened T * InputC)
    jobs = [];
    for t_idx = 0 : (SizeT - 1)
        for c_idx = 0 : (SizeC - 1)
            % Check if this input channel is needed for any requested output
            % Logic: Input C maps to Outputs (2*C) and (2*C+1)
            out1 = 2 * c_idx;
            out2 = 2 * c_idx + 1;

            if ~isempty(req_channels)
                % If neither derived output is requested, skip this input
                if ~ismember(out1, req_channels) && ~ismember(out2, req_channels)
                    continue;
                end
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
        c0 = jobs(i, 2); % Input channel index

        % --- Channel Mapping Logic ---
        % Even Inputs (0, 2): Bot -> Output 2*c0, Top -> Output 2*c0+1
        % Odd Inputs (1, 3):  Top -> Output 2*c0, Bot -> Output 2*c0+1
        is_even_input = (mod(c0, 2) == 0);

        % Structure to hold active outputs for this pass
        outputs_to_process = struct('id', {}, 'roi', {});

        idA = 2 * c0;
        idB = 2 * c0 + 1;

        if is_even_input
            % A=Bot, B=Top
            outputs_to_process(end+1) = struct('id', idA, 'roi', bot_roi);
            outputs_to_process(end+1) = struct('id', idB, 'roi', top_roi);
        else
            % A=Top, B=Bot
            outputs_to_process(end+1) = struct('id', idA, 'roi', top_roi);
            outputs_to_process(end+1) = struct('id', idB, 'roi', bot_roi);
        end

        % Filter: Remove outputs not requested or with invalid ROIs
        active_outputs = [];
        for k = 1:length(outputs_to_process)
            op = outputs_to_process(k);
            if (isempty(req_channels) || ismember(op.id, req_channels)) && ~isempty(op.roi)
                % Pre-allocate stack in struct
                h = op.roi(2) - op.roi(1) + 1;
                w = op.roi(4) - op.roi(3) + 1;
                if do_rotate
                    op.stack = zeros(w, h, SizeZ, 'uint16');
                else
                    op.stack = zeros(h, w, SizeZ, 'uint16');
                end
                active_outputs = [active_outputs, op]; %#ok<AGROW>
            end
        end

        if isempty(active_outputs)
            continue;
        end

        try
            % We keep track of the currently open Tiff to minimize fopen/fclose
            current_file_path = '';
            lt = [];

            for z0 = 0 : (SizeZ - 1)
                % 1. Global Index
                global_idx = (t0 * SizeZ * SizeC) + (z0 * SizeC) + c0;

                % 2. Resolve File
                [fPath, locIdx] = get_file_for_frame(global_idx, FileMap);

                if isempty(fPath)
                    continue; % Frame missing
                end

                % 3. Switch file
                if ~strcmp(fPath, current_file_path)
                    if ~isempty(lt), lt.close(); end
                    lt = Tiff(fPath, 'r');
                    current_file_path = fPath;
                end

                % 4. Read
                try
                    lt.setDirectory(locIdx + 1);
                    img = lt.read();

                    % 5. Crop for all active outputs
                    for k = 1:length(active_outputs)
                        roi = active_outputs(k).roi;
                        crop = img(roi(1):roi(2), roi(3):roi(4));
                        if do_rotate, crop = rot90(crop); end
                        active_outputs(k).stack(:,:,z0+1) = crop;
                    end
                catch
                    continue; % Silent fail for bad frame
                end
            end

            if ~isempty(lt), lt.close(); end

            % Write Outputs
            for k = 1:length(active_outputs)
                % Format: CleanBaseName_C#_T###.tif (No _bot/_top)
                outName = sprintf('%s_C%d_T%03d.tif', cleanBaseName, active_outputs(k).id, t0);
                outPath = fullfile(outputDir, outName);
                write_3d_tiff(outPath, active_outputs(k).stack);
            end

        catch ME
            fprintf('❌ Error T=%d C=%d: %s\n', t0, c0, ME.message);
            if ~isempty(lt), close(lt); end
        end
    end

    fprintf('✅ Parallel Crop Complete.\n');
end

% --- HELPER FUNCTIONS ---

function FileMap = map_multipage_tiff_files(masterFile)
    warning('off', 'MATLAB:imagesci:Tiff:libraryWarning');
    warning('off', 'MATLAB:tifflib:TIFFReadDirectory:libraryWarning');

    [fDir, fName, ~] = fileparts(masterFile);
    basePattern = regexprep(fName, '\.ome$', '');

    d = dir(fullfile(fDir, ['*.tif']));

    fileList = {};
    indices = [];

    safeBase = regexptranslate('escape', basePattern);
    patMaster = ['^' safeBase '\.ome\.tif$'];
    patSibling = ['^' safeBase '_(\d+)\.ome\.tif$'];

    for k = 1:length(d)
        thisName = d(k).name;
        if ~isempty(regexp(thisName, patMaster, 'once'))
            fileList{end+1} = fullfile(d(k).folder, thisName);
            indices(end+1) = 0;
            continue;
        end
        tok = regexp(thisName, patSibling, 'tokens');
        if ~isempty(tok)
            idx = str2double(tok{1}{1});
            fileList{end+1} = fullfile(d(k).folder, thisName);
            indices(end+1) = idx;
        end
    end

    [~, sortIdx] = sort(indices);
    sortedFiles = fileList(sortIdx);

    FileMap = struct('path', {}, 'start_idx', {}, 'end_idx', {});
    current_global_start = 0;

    fprintf('   Mapping files...\n');
    for k = 1:length(sortedFiles)
        fPath = sortedFiles{k};
        t = Tiff(fPath, 'r');
        num_frames = 0;
        while true
            num_frames = num_frames + 1;
            if t.lastDirectory(), break; end
            try t.nextDirectory(); catch, break; end
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
    fPath = ''; locIdx = 0;
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
