function run_petakit_cropper(json_file)
% RUN_PETAKIT_CROPPER High-performance parallel cropping for PetaKit5D
%   Refactored to output 3D Stacks (Time/Channel) instead of 2D planes.
%   Matches standard naming: [Base]_C[c]_T[t].tif

    % 0. Suppress annoying TIFF warnings globally
    % Micro-Manager uses custom tags (50839, 51123) that cause spam logs.
    warning('off', 'MATLAB:imagesci:Tiff:libraryWarning');

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

    fprintf('🚀 Starting Parallel Crop (3D Stacks)\n');
    fprintf('   Input:  %s\n', fullBaseName);
    fprintf('   Output: %s\n', outputDir);

    % 2. Parse Parameters
    top_roi = parse_roi_str(job.parameters.rois.top);
    bot_roi = parse_roi_str(job.parameters.rois.bottom);
    do_rotate = isfield(job.parameters, 'rotate') && job.parameters.rotate;

    req_channels = [];
    if isfield(job.parameters, 'channels')
        req_channels = job.parameters.channels;
    end

    % 3. Read Metadata (SizeZ, SizeC, SizeT)
    if ~exist(inputFile, 'file'), error('Input file missing.'); end
    t = Tiff(inputFile, 'r');

    % Defaults
    SizeC = 1; SizeZ = 1; SizeT = 1;

    try
        desc = t.getTag('ImageDescription');
        % Parse OME-XML dimensions
        tokC = regexp(desc, 'SizeC="(\d+)"', 'tokens');
        tokZ = regexp(desc, 'SizeZ="(\d+)"', 'tokens');
        tokT = regexp(desc, 'SizeT="(\d+)"', 'tokens');

        if ~isempty(tokC), SizeC = str2double(tokC{1}{1}); end
        if ~isempty(tokZ), SizeZ = str2double(tokZ{1}{1}); end
        if ~isempty(tokT), SizeT = str2double(tokT{1}{1}); end
    catch
        % Warning already suppressed above, though we log failure here if crucial
        warning('Failed to parse OME-XML. Calculations may be wrong.');
    end
    t.close();

    fprintf('📊 Dimensions: T=%d, Z=%d, C=%d\n', SizeT, SizeZ, SizeC);

    % --- FIX 3: Write Processing Log ---
    logStruct = struct();
    logStruct.original_file = inputFile;
    logStruct.timestamp = char(datetime('now'));
    logStruct.dimensions = struct('SizeT', SizeT, 'SizeZ', SizeZ, 'SizeC', SizeC);
    logStruct.rois = job.parameters.rois;
    logStruct.rotate = do_rotate;
    logStruct.channel_mapping = 'C0,3=Bot; C1,2=Top'; % Static note for reference

    jsonTxt = jsonencode(logStruct, 'PrettyPrint', true);
    logPath = fullfile(outputDir, [cleanBaseName, '_processing_log.json']);
    fid = fopen(logPath, 'w');
    fprintf(fid, '%s', jsonTxt);
    fclose(fid);
    fprintf('📝 Log written: %s\n', logPath);

    % 4. Build Job List (Flattened T * C)
    % We process 3D stacks. Each worker grabs one (T, C) pair and reads all Z.
    % Job structure: [t_index, c_index] (0-based)

    jobs = [];
    for t_idx = 0 : (SizeT - 1)
        for c_idx = 0 : (SizeC - 1)
            % Filter requested channels
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
        % Ensure warnings are off on workers too (Critical for speed)
        warning('off', 'MATLAB:imagesci:Tiff:libraryWarning');

        t0 = jobs(i, 1);
        c0 = jobs(i, 2);

        % Channel Logic: 0,3=Bot; 1,2=Top
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
            continue; % Should not happen if inputs are valid
        end

        % --- Memory Pre-allocation for 3D Stack ---
        % Calculate crop size
        h = target_roi(2) - target_roi(1) + 1;
        w = target_roi(4) - target_roi(3) + 1;

        if do_rotate
            % Swap H/W if rotating
            stack = zeros(w, h, SizeZ, 'uint16');
        else
            stack = zeros(h, w, SizeZ, 'uint16');
        end

        % --- Read Z-Stack ---
        % Standard OME Order: T -> Z -> C (Interleaved) OR T -> C -> Z
        % We assume "XYCZT" (Interleaved Channels) which is common in MicroManager
        % Index = t * (SizeZ * SizeC) + z * SizeC + c

        try
            lt = Tiff(inputFile, 'r');

            for z0 = 0 : (SizeZ - 1)
                % Calculate Directory Index (1-based for MATLAB)
                dir_idx = (t0 * SizeZ * SizeC) + (z0 * SizeC) + c0 + 1;

                lt.setDirectory(dir_idx);
                img = lt.read();

                % Crop
                crop = img(target_roi(1):target_roi(2), target_roi(3):target_roi(4));

                if do_rotate
                    crop = rot90(crop);
                end

                % Insert into 3D buffer (z0+1 because MATLAB is 1-based)
                stack(:, :, z0 + 1) = crop;
            end
            lt.close();

            % --- Write 3D Output File ---
            % Naming: [Base]_C[c]_T[t].tif to match regex ^(.*?)_C\d_T\d{3}\.tif$
            % Note: We include '_bot' or '_top' in the base part to be descriptive but safe

            % Format: cleanBaseName_C#_T###.tif
            % We append suffix to base name so regex (.*?) eats it.
            outName = sprintf('%s_%s_C%d_T%03d.tif', cleanBaseName, suffix, c0, t0);
            outPath = fullfile(outputDir, outName);

            write_3d_tiff(outPath, stack);

        catch ME
            fprintf('❌ Error T=%d C=%d: %s\n', t0, c0, ME.message);
        end
    end

    fprintf('✅ Parallel Crop Complete.\n');
end

function write_3d_tiff(filename, stack)
    % Helper to write 3D volume
    % stack is HxWxZ

    % Write first frame (overwrite mode)
    imwrite(stack(:,:,1), filename);

    % Append rest
    num_z = size(stack, 3);
    if num_z > 1
        for k = 2:num_z
            imwrite(stack(:,:,k), filename, 'WriteMode', 'append');
        end
    end
end

function roi = parse_roi_str(s)
    if isempty(s) || strcmp(s, 'null')
        roi = []; return;
    end
    clean = replace(s, {':', ','}, ' ');
    vals = sscanf(clean, '%d %d %d %d');
    if numel(vals) == 4
        % Py [y1:y2, x1:x2] -> Mat [y1+1, y2, x1+1, x2]
        roi = [vals(1)+1, vals(2), vals(3)+1, vals(4)];
    else
        roi = [];
    end
end
