function run_petakit_cropper(json_file)
% RUN_PETAKIT_CROPPER High-performance parallel cropping (Native MATLAB)
%   Optimized for speed using 'tiffread' from PetaKit5D if available.
%   Falls back to optimized sequential Tiff class usage if not.

    % 0. Cleanup and Setup
    warning('off', 'all');

    if ~isfile(json_file), error('JSON not found: %s', json_file); end
    fid = fopen(json_file, 'r');
    job = jsondecode(char(fread(fid, inf)'));
    fclose(fid);

    inputFile = job.dataDir;
    [dataRoot, fullBaseName, ~] = fileparts(inputFile);
    cleanBaseName = regexprep(fullBaseName, '\.ome$', '');
    outputDir = fullfile(dataRoot, [cleanBaseName, '_cropped']);

    if ~exist(outputDir, 'dir'), mkdir(outputDir); end

    fprintf('🚀 Starting Crop (PetaKit5D Fast Reader)\n');
    fprintf('   Input:  %s\n', fullBaseName);
    fprintf('   Output: %s\n', outputDir);

    % Check for fast reader
    has_fast_reader = ~isempty(which('tiffread'));
    if has_fast_reader
        fprintf('✅ Fast reader found: "tiffread"\n');
    else
        fprintf('⚠️ "tiffread" not found. Falling back to native Tiff class.\n');
    end

    % 1. Map Files & Parse Metadata
    % Optimization: Check if metadata was passed in the JSON to save time
    use_cached_map = false;

    % Check if 'metadata' exists and has 'FileMap'
    if isfield(job, 'metadata') && isfield(job.metadata, 'FileMap') && ~isempty(job.metadata.FileMap)
        try
            fprintf('📥 Found metadata in JSON ticket. Validating...\n');

            % Extract Dimensions
            SizeT = job.metadata.SizeT;
            SizeZ = job.metadata.SizeZ;
            SizeC = job.metadata.SizeC;
            DimOrder = job.metadata.DimOrder;

            % Extract FileMap
            FileMap = job.metadata.FileMap;

            % Basic validation that it is a struct array with correct fields
            % Note: JSON decoding might result in camelCase or different structures depending on source
            if isstruct(FileMap) && isfield(FileMap, 'path') && isfield(FileMap, 'start_idx') && isfield(FileMap, 'end_idx')
                use_cached_map = true;
            else
                fprintf('⚠️ FileMap in JSON is missing required fields (path, start_idx, end_idx). Re-scanning.\n');
            end
        catch ME
            fprintf('⚠️ Error reading cached metadata: %s. Re-scanning.\n', ME.message);
            use_cached_map = false;
        end
    end

    if use_cached_map
        fprintf('✅ Using cached FileMap. Skipping structure scan.\n');
    else
        fprintf('🔍 Scanning file structure (this may take a moment)...\n');
        [FileMap, SizeT, SizeZ, SizeC, DimOrder] = map_ome_tiff_structure(inputFile);
    end

    fprintf('📊 Metadata: T=%d, Z=%d, C=%d\n', SizeT, SizeZ, SizeC);

    % 2. Setup Parameters
    top_roi = parse_roi_str(job.parameters.rois.top);
    bot_roi = parse_roi_str(job.parameters.rois.bottom);
    do_rotate = isfield(job.parameters, 'rotate') && job.parameters.rotate;

    req_channels = [];
    if isfield(job.parameters, 'channels')
        req_channels = job.parameters.channels;
        if iscell(req_channels), req_channels = cell2mat(req_channels); end
    end

    % Write Log
    logStruct = struct();
    logStruct.original_file = inputFile;
    logStruct.timestamp = char(datetime('now'));
    logStruct.metadata = struct('SizeT', SizeT, 'SizeZ', SizeZ, 'SizeC', SizeC);
    logStruct.parameters = job.parameters;
    fid = fopen(fullfile(outputDir, [cleanBaseName, '_processing_log.json']), 'w');
    fprintf(fid, '%s', jsonencode(logStruct, 'PrettyPrint', true));
    fclose(fid);

    % 3. Parallel Processing Loop
    pool = gcp('nocreate');
    if isempty(pool), pool = parpool('local'); end
    fprintf('🔥 Processing %d timepoints on %d workers...\n', SizeT, pool.NumWorkers);

    parfor t_idx = 0 : (SizeT - 1)
        % --- A. PLAN THE READS ---
        frames_to_read = struct('global_idx', {}, 'z', {}, 'c', {});
        count = 0;

        for z = 0:(SizeZ-1)
            for c = 0:(SizeC-1)
                if strcmp(DimOrder, 'XYCZT') || strcmp(DimOrder, 'Default')
                    g_idx = (t_idx * SizeZ * SizeC) + (z * SizeC) + c;
                elseif strcmp(DimOrder, 'XYZCT')
                    g_idx = (t_idx * SizeZ * SizeC) + (c * SizeZ) + z;
                else
                    g_idx = (t_idx * SizeZ * SizeC) + (z * SizeC) + c;
                end

                count = count + 1;
                frames_to_read(count).global_idx = g_idx;
                frames_to_read(count).z = z;
                frames_to_read(count).c = c;
            end
        end

        % --- B. SORT READS BY DISK LOCATION ---
        read_list = cell(count, 5);
        valid_read_count = 0;
        for i = 1:count
            g_idx = frames_to_read(i).global_idx;
            [fPath, locIdx] = get_file_for_frame(g_idx, FileMap);

            if ~isempty(fPath)
                valid_read_count = valid_read_count + 1;
                read_list{valid_read_count, 1} = fPath;
                read_list{valid_read_count, 2} = locIdx;
                read_list{valid_read_count, 3} = frames_to_read(i).z;
                read_list{valid_read_count, 4} = frames_to_read(i).c;
            end
        end

        if valid_read_count == 0, continue; end
        Tbl = cell2table(read_list(1:valid_read_count, :), 'VariableNames', {'Path', 'LocIdx', 'Z', 'C', 'Global'});
        Tbl = sortrows(Tbl, {'Path', 'LocIdx'});

        % --- C. PRE-ALLOCATE BUFFERS ---
        h_top=0; w_top=0; h_bot=0; w_bot=0;
        if ~isempty(top_roi), h_top=top_roi(2)-top_roi(1)+1; w_top=top_roi(4)-top_roi(3)+1; end
        if ~isempty(bot_roi), h_bot=bot_roi(2)-bot_roi(1)+1; w_bot=bot_roi(4)-bot_roi(3)+1; end

        if do_rotate
            final_h_top = w_top; final_w_top = h_top;
            final_h_bot = w_bot; final_w_bot = h_bot;
        else
            final_h_top = h_top; final_w_top = w_top;
            final_h_bot = h_bot; final_w_bot = w_bot;
        end

        output_buffers = containers.Map('KeyType', 'double', 'ValueType', 'any');

        % --- D. EXECUTE READS (FAST BATCH) ---
        uniqueFiles = unique(Tbl.Path);

        for uf = 1:length(uniqueFiles)
            currPath = uniqueFiles{uf};
            fileMask = strcmp(Tbl.Path, currPath);
            subTbl = Tbl(fileMask, :);

            % Indices to read (1-based for tiffread usually)
            indices_to_load = subTbl.LocIdx + 1;

            imgs = {};

            if has_fast_reader
                try
                    % PetaKit tiffread typically: tiffread(filename, indices)
                    % Returns struct array or cell array
                    raw_data = tiffread(currPath, indices_to_load);

                    if isstruct(raw_data)
                        % Assuming .data field
                        imgs = {raw_data.data};
                    elseif iscell(raw_data)
                        imgs = raw_data;
                    else
                        % Maybe it returns a 3D matrix?
                         if ndims(raw_data) == 3
                             % Unpack 3D matrix to cell
                             num_loaded = size(raw_data, 3);
                             imgs = cell(1, num_loaded);
                             for slice = 1:num_loaded
                                 imgs{slice} = raw_data(:,:,slice);
                             end
                         end
                    end
                catch
                    % Fallback to slow read for this file if fast reader errors
                    imgs = fallback_read(currPath, indices_to_load);
                end
            else
                imgs = fallback_read(currPath, indices_to_load);
            end

            % Process Loaded Images
            for k = 1:length(indices_to_load)
                if k > length(imgs), break; end
                img = imgs{k};

                % Metadata for this frame
                z_curr = subTbl.Z(k);
                c_input = subTbl.C(k);

                % Process Logic (Same as before)
                outA_id = 2 * c_input;
                outB_id = 2 * c_input + 1;
                is_even = mod(c_input, 2) == 0;

                % Output A
                if isempty(req_channels) || ismember(outA_id, req_channels)
                    if is_even, use_roi=bot_roi; H=final_h_bot; W=final_w_bot;
                    else, use_roi=top_roi; H=final_h_top; W=final_w_top; end

                    if ~isempty(use_roi)
                        if ~isKey(output_buffers, outA_id)
                            output_buffers(outA_id) = zeros(H, W, SizeZ, 'uint16');
                        end
                        crop = img(use_roi(1):use_roi(2), use_roi(3):use_roi(4));
                        if do_rotate, crop = rot90(crop); end
                        buf = output_buffers(outA_id);
                        buf(:,:,z_curr+1) = crop;
                        output_buffers(outA_id) = buf;
                    end
                end

                % Output B
                if isempty(req_channels) || ismember(outB_id, req_channels)
                    if is_even, use_roi=top_roi; H=final_h_top; W=final_w_top;
                    else, use_roi=bot_roi; H=final_h_bot; W=final_w_bot; end

                    if ~isempty(use_roi)
                         if ~isKey(output_buffers, outB_id)
                            output_buffers(outB_id) = zeros(H, W, SizeZ, 'uint16');
                        end
                        crop = img(use_roi(1):use_roi(2), use_roi(3):use_roi(4));
                        if do_rotate, crop = rot90(crop); end
                        buf = output_buffers(outB_id);
                        buf(:,:,z_curr+1) = crop;
                        output_buffers(outB_id) = buf;
                    end
                end
            end
        end

        % --- E. WRITE OUTPUTS ---
        keys = output_buffers.keys;
        for k = 1:length(keys)
            id = keys{k};
            stack = output_buffers(id);
            outName = sprintf('%s_C%d_T%03d.tif', cleanBaseName, id, t_idx);
            outPath = fullfile(outputDir, outName);
            write_3d_tiff(outPath, stack);
        end
    end
    fprintf('✅ Crop Complete.\n');
end

% --- HELPERS ---

function imgs = fallback_read(fPath, indices)
    % Optimized sequential reader using Tiff class
    imgs = cell(1, length(indices));
    try
        t = Tiff(fPath, 'r');
        for i = 1:length(indices)
            t.setDirectory(indices(i));
            imgs{i} = t.read();
        end
        t.close();
    catch
        if exist('t', 'var'), close(t); end
    end
end

function [FileMap, SizeT, SizeZ, SizeC, DimOrder] = map_ome_tiff_structure(masterFile)
    t = Tiff(masterFile, 'r');
    SizeC=1; SizeZ=1; SizeT=1; DimOrder='XYCZT';
    try
        desc = t.getTag('ImageDescription');
        tokC = regexp(desc, 'SizeC="(\d+)"', 'tokens');
        if ~isempty(tokC), SizeC = str2double(tokC{1}{1}); end
        tokZ = regexp(desc, 'SizeZ="(\d+)"', 'tokens');
        if ~isempty(tokZ), SizeZ = str2double(tokZ{1}{1}); end
        tokT = regexp(desc, 'SizeT="(\d+)"', 'tokens');
        if ~isempty(tokT), SizeT = str2double(tokT{1}{1}); end
        tokO = regexp(desc, 'DimensionOrder="([^"]+)"', 'tokens');
        if ~isempty(tokO), DimOrder = tokO{1}{1}; end
    catch
    end
    t.close();

    [fDir, fName, ~] = fileparts(masterFile);
    basePattern = regexprep(fName, '\.ome$', '');
    d = dir(fullfile(fDir, '*.tif'));
    files = struct('path', {}, 'start_idx', {}, 'end_idx', {}, 'count', {}, 'idx_suffix', {});
    valid_count = 0;
    safeBase = regexptranslate('escape', basePattern);

    for k = 1:length(d)
        fn = d(k).name;
        isMaster = ~isempty(regexp(fn, ['^' safeBase '\.ome\.tif$'], 'once'));
        tokSib = regexp(fn, ['^' safeBase '_(\d+)\.ome\.tif$'], 'tokens');

        idx_suffix = -1;
        if isMaster, idx_suffix = 0;
        elseif ~isempty(tokSib), idx_suffix = str2double(tokSib{1}{1}); end

        if idx_suffix >= 0
            valid_count = valid_count + 1;
            files(valid_count).path = fullfile(d(k).folder, fn);
            files(valid_count).idx_suffix = idx_suffix;
        end
    end

    [~, I] = sort([files.idx_suffix]);
    files = files(I);

    current_start = 0;
    for k = 1:length(files)
        fPath = files(k).path;
        count = 0;
        try
            % Check for tiffread to speed up counting?
            % Usually ImageDescription in each file works better.
            % But standard Tiff count is safest.
            t = Tiff(fPath, 'r');
            while true
                count = count + 1;
                if t.lastDirectory(), break; end
                t.nextDirectory();
            end
            t.close();
        catch
        end
        files(k).count = count;
        files(k).start_idx = current_start;
        files(k).end_idx = current_start + count - 1;
        current_start = current_start + count;
    end
    FileMap = files;
end

function [fPath, locIdx] = get_file_for_frame(gIdx, FileMap)
    fPath = ''; locIdx = 0;
    for k = 1:length(FileMap)
        if gIdx >= FileMap(k).start_idx && gIdx <= FileMap(k).end_idx
            fPath = FileMap(k).path;
            locIdx = gIdx - FileMap(k).start_idx;
            return;
        end
    end
end

function roi = parse_roi_str(s)
    if isempty(s) || strcmp(s, 'null'), roi = []; return; end
    clean = replace(s, {':', ','}, ' ');
    vals = sscanf(clean, '%d %d %d %d');
    if numel(vals) == 4, roi = [vals(1)+1, vals(2), vals(3)+1, vals(4)]; else, roi = []; end
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
