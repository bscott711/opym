function run_bigtiff_cropper(job)
% RUN_BIGTIFF_CROPPER Reads BigTiff via XML Map, crops, and saves split channels.
%
% UPDATES:
%   - Filenames now match the folder name (Fixes Deskew "File Not Found").
%   - Timepoints are 0-indexed (T0000) to match Python/Napari.
%   - Uses BigTiffFastLoader (XML Map version).

    % --- 1. Setup ---
    p = job.parameters;

    % Resolve Master File
    if isfile(job.dataDir)
        masterFile = job.dataDir;
        [workDir, ~] = fileparts(masterFile);
    else
        f = dir(fullfile(job.dataDir, '*.ome.tif'));
        idx = arrayfun(@(x) isempty(regexp(x.name, '_\d+\.ome\.tif$', 'once')), f);
        f = f(idx);
        if isempty(f), error('No master .ome.tif found in %s', job.dataDir); end
        masterFile = fullfile(job.dataDir, f(1).name);
        workDir = job.dataDir;
    end

    % Parse ROIs
    roiTop = parseROI(p.rois.top);
    roiBot = parseROI(p.rois.bottom);
    doRotate = isfield(p, 'rotate') && p.rotate;

    % Determine Output Folder & Filename Prefix
    % Logic: If job has baseName, use it. Otherwise derive from file.
    if isfield(job, 'baseName')
        % Remove extension if present in the baseName string
        cleanName = regexprep(job.baseName, '\.ome(\.tif)?$', '');
    else
        [~, cleanName] = fileparts(masterFile);
    end

    outDir = fullfile(workDir, cleanName);
    if ~exist(outDir, 'dir'), mkdir(outDir); end

    fprintf('   [Cropper] Source: %s\n', masterFile);
    fprintf('   [Cropper] Output: %s\n', outDir);
    fprintf('   [Cropper] Naming: %s_Cxx_Txxxx.tif\n', cleanName);

    % --- 2. Initialize XML Loader ---
    % Suppress annoying Tiff warnings on workers
    warning('off', 'MATLAB:imagesci:Tiff:libraryWarning');
    warning('off', 'MATLAB:imagesci:tifftagsread:expectedTagDataFormat');

    pool = gcp('nocreate');
    if isempty(pool), pool = parpool(48); end
    addAttachedFiles(pool, {'BigTiffFastLoader.m'});

    loader = BigTiffFastLoader(masterFile);

    % Dimensions
    T = loader.Dimensions.SizeT;
    Z = loader.Dimensions.SizeZ;
    RawC = loader.Dimensions.SizeC;

    num_raw_stacks = T * RawC;
    fprintf('   [Cropper] Processing %d Raw Stacks -> %d Output Channels\n', num_raw_stacks, RawC*2);

    % --- 3. Parallel Execution ---
    tic;
    parfor k = 1:num_raw_stacks
        % Suppress warnings inside workers
        warning('off', 'MATLAB:imagesci:Tiff:libraryWarning');

        % Map linear index 'k' to Raw(Channel, Time)
        % Note: t_in is 1-based (MATLAB), t_out will be 0-based (Python)
        [rc, t_in] = ind2sub([RawC, T], k);

        % A. Load Frame (Loader handles descrambling automatically)
        rawStack = zeros(loader.Geometry.H, loader.Geometry.W, Z, 'uint16');
        for z = 1:Z
            rawStack(:,:,z) = loader.getFrame(t_in, z, rc);
        end

        % B. Determine Output Channel IDs
        base_out_ch = (rc - 1) * 2;
        isCam1 = mod(rc, 2) ~= 0;

        if isCam1
            % Camera 1: Bottom is Low, Top is High
            ch_Bot = base_out_ch;     % e.g. 0
            ch_Top = base_out_ch + 1; % e.g. 1
        else
            % Camera 2: Top is Low, Bottom is High (Inverted)
            ch_Top = base_out_ch;     % e.g. 2
            ch_Bot = base_out_ch + 1; % e.g. 3
        end

        % --- CROP, ROTATE, WRITE (Dynamic Naming) ---

        % Bottom Crop
        stackBot = rawStack(roiBot.y, roiBot.x, :);
        if doRotate, stackBot = rot90(stackBot); end

        % Name Format: cleanName_C00_T0000.tif
        fNameBot = sprintf('%s_C%02d_T%04d.tif', cleanName, ch_Bot, t_in - 1);
        writetiff(stackBot, fullfile(outDir, fNameBot));

        % Top Crop
        stackTop = rawStack(roiTop.y, roiTop.x, :);
        if doRotate, stackTop = rot90(stackTop); end

        fNameTop = sprintf('%s_C%02d_T%04d.tif', cleanName, ch_Top, t_in - 1);
        writetiff(stackTop, fullfile(outDir, fNameTop));
    end
    t_end = toc;

    fprintf('   [Cropper] Complete in %.2f s. \n', t_end);
end

function roi = parseROI(str)
    parts = split(str, ',');
    yRange = str2num(parts{1}); %#ok<ST2NM>
    xRange = str2num(parts{2}); %#ok<ST2NM>
    roi.y = yRange; roi.x = xRange;
    roi.h = length(yRange); roi.w = length(xRange);
end
