function run_petakit_cropper(json_file)
% RUN_PETAKIT_CROPPER High-performance parallel cropping (Native MATLAB)
%   Uses PetaKit5D 'readtiff' and 'writetiff' (C++ MEX) for maximum speed.
%   Features "Fast Scan" mapping to instantly handle split OME-TIFFs.

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

    fprintf('🚀 Starting PetaKit5D Cropper\n');
    fprintf('   Input:  %s\n', fullBaseName);
    fprintf('   Output: %s\n', outputDir);

    % --- 1. SETUP PETAKIT ENVIRONMENT ---
    petakit_root = getenv('PETAKIT_ROOT');
    if isempty(petakit_root), petakit_root = '/cm/shared/apps_local/petakit5d'; end

    setup_script = fullfile(petakit_root, 'setup.m');
    if exist(setup_script, 'file')
        fprintf('🔧 Initializing PetaKit5D from: %s\n', petakit_root);
        run(setup_script);
    else
        fprintf('⚠️ setup.m not found at %s. Attempting manual path add...\n', setup_script);
        addpath(genpath(petakit_root));
    end

    % --- 2. VERIFY OPTIMIZED I/O ---
    has_readtiff = ~isempty(which('readtiff'));
    has_writetiff = ~isempty(which('writetiff'));

    if has_readtiff
        fprintf('✅ Optimized Reader found: "%s"\n', which('readtiff'));
    else
        fprintf('⚠️ "readtiff" NOT found. Fallback to slow Tiff class will be used.\n');
    end

    if has_writetiff
        fprintf('✅ Optimized Writer found: "%s"\n', which('writetiff'));
    else
        fprintf('⚠️ "writetiff" NOT found. Fallback to slow imwrite will be used.\n');
    end

    % --- 3. METADATA & MAPPING (FAST SCAN) ---
    use_cached_map = false;
    if isfield(job, 'metadata') && isfield(job.metadata, 'FileMap') && ~isempty(job.metadata.FileMap)
        try
            SizeT = job.metadata.SizeT;
            SizeZ = job.metadata.SizeZ;
            SizeC = job.metadata.SizeC;
            DimOrder = job.metadata.DimOrder;
            FileMap = job.metadata.FileMap;
            if isstruct(FileMap) && isfield(FileMap, 'path') && isfield(FileMap, 'start_idx') && isfield(FileMap, 'end_idx')
                use_cached_map = true;
            end
        catch
        end
    end

    if use_cached_map
        fprintf('📥 Using cached file structure from JSON.\n');
    else
        fprintf('🔍 Instant-Scanning file structure...\n');
        [FileMap, SizeT, SizeZ, SizeC, DimOrder] = map_ome_tiff_structure(inputFile);
    end
    fprintf('📊 Metadata: T=%d, Z=%d, C=%d (%s)\n', SizeT, SizeZ, SizeC, DimOrder);

    % --- 4. PARAMETER SETUP ---
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

    % --- 5. PARALLEL PROCESSING LOOP ---
    pool = gcp('nocreate');
    if isempty(pool), pool = parpool('local'); end
    fprintf('🔥 Processing %d timepoints on %d workers...\n', SizeT, pool.NumWorkers);

    parfor t_idx = 0 : (SizeT - 1)
        % A. PLAN READS (Logic for sorting disk access)
        frames_to_read = struct('global_idx', {}, 'z', {}, 'c', {});
        count = 0;
        for z = 0:(SizeZ-1)
            for c = 0:(SizeC-1)
                % Calculate Index based on Order
                if strcmp(DimOrder, 'XYZCT')
                    g_idx = (t_idx * SizeZ * SizeC) + (c * SizeZ) + z;
                else % Default XYCZT
                    g_idx = (t_idx * SizeZ * SizeC) + (z * SizeC) + c;
                end
                count = count + 1;
                frames_to_read(count).global_idx = g_idx;
                frames_to_read(count).z = z;
                frames_to_read(count).c = c;
            end
        end

        % B. SORT READS (Sequential Disk Access)
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

        % C. PRE-ALLOCATE BUFFERS
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

        % D. EXECUTE READS (Using Optimized readtiff)
        uniqueFiles = unique(Tbl.Path);
        for uf = 1:length(uniqueFiles)
            currPath = uniqueFiles{uf};
            fileMask = strcmp(Tbl.Path, currPath);
            subTbl = Tbl(fileMask, :);

            % 1-based indices for PetaKit readtiff
            indices_to_load = subTbl.LocIdx + 1;

            imgs = {};
            if has_readtiff
                try
                    % PetaKit readtiff(filename, indices)
                    raw = readtiff(currPath, indices_to_load);

                    % Normalize output format
                    if isstruct(raw), imgs = {raw.data};
                    elseif iscell(raw), imgs = raw;
                    elseif ndims(raw) == 3
                        num_loaded = size(raw, 3);
                        imgs = cell(1, num_loaded);
                        for sl = 1:num_loaded, imgs{sl} = raw(:,:,sl); end
                    elseif ismatrix(raw) && length(indices_to_load) == 1
                        imgs = {raw};
                    end
                catch
                    imgs = fallback_read(currPath, indices_to_load);
                end
            else
                imgs = fallback_read(currPath, indices_to_load);
            end

            % Distribute to buffers
            for k = 1:length(indices_to_load)
                if k > length(imgs), break; end
                img = imgs{k};
                z_curr = subTbl.Z(k);
                c_input = subTbl.C(k);

                outA_id = 2 * c_input;
                outB_id = 2 * c_input + 1;
                is_even = mod(c_input, 2) == 0;

                % Output A
                if isempty(req_channels) || ismember(outA_id, req_channels)
                    if is_even, use_roi=bot_roi; H=final_h_bot; W=final_w_bot;
                    else, use_roi=top_roi; H=final_h_top; W=final_w_top; end

                    if ~isempty(use_roi)
                        if ~isKey(output_buffers, outA_id), output_buffers(outA_id) = zeros(H, W, SizeZ, 'uint16'); end
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
                        if ~isKey(output_buffers, outB_id), output_buffers(outB_id) = zeros(H, W, SizeZ, 'uint16'); end
                        crop = img(use_roi(1):use_roi(2), use_roi(3):use_roi(4));
                        if do_rotate, crop = rot90(crop); end
                        buf = output_buffers(outB_id);
                        buf(:,:,z_curr+1) = crop;
                        output_buffers(outB_id) = buf;
                    end
                end
            end
        end

        % E. WRITE OUTPUTS (Using Optimized writetiff)
        keys = output_buffers.keys;
        for k = 1:length(keys)
            id = keys{k};
            stack = output_buffers(id);
            outName = sprintf('%s_C%d_T%03d.tif', cleanBaseName, id, t_idx);
            outPath = fullfile(outputDir, outName);

            if has_writetiff
                try
                    % PetaKit writetiff(data, filename)
                    writetiff(stack, outPath);
                catch
                    write_3d_tiff_fallback(outPath, stack);
                end
            else
                write_3d_tiff_fallback(outPath, stack);
            end
        end
    end
    fprintf('✅ Crop Complete.\n');
end

% --- HELPERS ---

function imgs = fallback_read(fPath, indices)
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

function write_3d_tiff_fallback(filename, stack)
    imwrite(stack(:,:,1), filename);
    num_z = size(stack, 3);
    if num_z > 1
        for k = 2:num_z
            imwrite(stack(:,:,k), filename, 'WriteMode', 'append');
        end
    end
end

function [FileMap, SizeT, SizeZ, SizeC, DimOrder] = map_ome_tiff_structure(masterFile)
    % Reads metadata and uses FAST HEURISTIC (File Size) to map frames.

    warning('off', 'all');
    t = Tiff(masterFile, 'r');

    SizeC=1; SizeZ=1; SizeT=1; DimOrder='XYCZT';
    W=0; H=0; BPS=0; Compression=1;

    try
        desc = t.getTag('ImageDescription');
        tokC = regexp(desc, 'SizeC="(\d+)"', 'tokens'); if ~isempty(tokC), SizeC = str2double(tokC{1}{1}); end
        tokZ = regexp(desc, 'SizeZ="(\d+)"', 'tokens'); if ~isempty(tokZ), SizeZ = str2double(tokZ{1}{1}); end
        tokT = regexp(desc, 'SizeT="(\d+)"', 'tokens'); if ~isempty(tokT), SizeT = str2double(tokT{1}{1}); end
        tokO = regexp(desc, 'DimensionOrder="([^"]+)"', 'tokens'); if ~isempty(tokO), DimOrder = tokO{1}{1}; end

        W = double(t.getTag('ImageWidth'));
        H = double(t.getTag('ImageLength'));
        BPS = double(t.getTag('BitsPerSample'));
        try Compression = t.getTag('Compression'); catch, Compression=1; end
    catch
    end
    t.close();

    [fDir, fName, ~] = fileparts(masterFile);
    basePattern = regexprep(fName, '\.ome$', '');
    d = dir(fullfile(fDir, '*.tif'));
    files = struct('path', {}, 'start_idx', {}, 'end_idx', {}, 'count', {}, 'idx_suffix', {}, 'bytes', {});
    valid_count = 0;
    safeBase = regexptranslate('escape', basePattern);

    for k = 1:length(d)
        fn = d(k).name;
        isMaster = ~isempty(regexp(fn, ['^' safeBase '\.ome\.tif$'], 'once'));
        tokSib = regexp(fn, ['^' safeBase '_(\d+)\.ome\.tif$'], 'tokens');
        idx_suffix = -1;
        if isMaster, idx_suffix = 0; elseif ~isempty(tokSib), idx_suffix = str2double(tokSib{1}{1}); end
        if idx_suffix >= 0
            valid_count = valid_count + 1;
            files(valid_count).path = fullfile(d(k).folder, fn);
            files(valid_count).idx_suffix = idx_suffix;
            files(valid_count).bytes = d(k).bytes;
        end
    end
    [~, I] = sort([files.idx_suffix]);
    files = files(I);

    current_start = 0;
    bytes_per_frame = W * H * (BPS / 8);

    for k = 1:length(files)
        fPath = files(k).path;
        count = 0;

        % FAST HEURISTIC
        can_fast_scan = (Compression == 1) && (bytes_per_frame > 0);
        if can_fast_scan
             count = floor(files(k).bytes / bytes_per_frame);
             if count == 0 && files(k).bytes > 1024*1024
                 % Heuristic failed sanity check
                 count = 0;
             end
        end

        % FALLBACK
        if count == 0
             try
                 t = Tiff(fPath, 'r');
                 while true
                     count = count + 1;
                     if t.lastDirectory(), break; end
                     t.nextDirectory();
                 end
                 t.close();
             catch
             end
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
