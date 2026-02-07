function run_bigtiff_cropper(job)
% RUN_BIGTIFF_CROPPER Reads BigTiff, Crops Top/Bottom ROIs, and maps Channels.
%
% LOGIC UPDATE (User Correction):
%   - Raw Channel 1 (Cam 1): Bottom -> Ch0, Top -> Ch1
%   - Raw Channel 2 (Cam 2): Top -> Ch2, Bottom -> Ch3
%   - Raw Channel 3 (Cam 1): Bottom -> Ch4, Top -> Ch5
%   - Raw Channel 4 (Cam 2): Top -> Ch6, Bottom -> Ch7

    % --- 1. Setup & Parsing ---
    p = job.parameters;

    % Resolve Master File
    if isfile(job.dataDir)
        masterFile = job.dataDir;
        [workDir, ~] = fileparts(masterFile);
    else
        f = dir(fullfile(job.dataDir, '*.ome.tif'));
        % Filter out _1.ome.tif siblings
        idx = arrayfun(@(x) isempty(regexp(x.name, '_\d+\.ome\.tif$', 'once')), f);
        f = f(idx);
        if isempty(f), error('No master .ome.tif found in %s', job.dataDir); end
        masterFile = fullfile(job.dataDir, f(1).name);
        workDir = job.dataDir;
    end

    % Parse ROIs (Format: "y1:y2,x1:x2")
    roiTop = parseROI(p.rois.top);
    roiBot = parseROI(p.rois.bottom);

    doRotate = isfield(p, 'rotate') && p.rotate;

    % Output Directory
    if isfield(job, 'baseName')
        folderName = regexprep(job.baseName, '\.ome(\.tif)?$', '');
    else
        [~, folderName] = fileparts(masterFile);
    end
    outDir = fullfile(workDir, folderName);
    if ~exist(outDir, 'dir'), mkdir(outDir); end

    fprintf('   [Cropper] Source: %s\n', masterFile);
    fprintf('   [Cropper] Output: %s\n', outDir);

    % --- 2. Initialize Fast Loader ---
    pool = gcp('nocreate');
    if isempty(pool), pool = parpool(48); end
    addAttachedFiles(pool, {'BigTiffFastLoader.m'});

    loader = BigTiffFastLoader(masterFile);
    loader.ReaderHandle = @(f, idx) readtiff(f, idx); % Hook PetaKit reader

    % Dimensions
    T = loader.Dimensions.SizeT;
    Z = loader.Dimensions.SizeZ;
    RawC = loader.Dimensions.SizeC;

    % Total Processing Units
    num_raw_stacks = T * RawC;

    fprintf('   [Cropper] Processing %d Raw Stacks -> %d Output Channels\n', num_raw_stacks, RawC*2);

    % --- 3. Parallel Execution ---
    tic;
    parfor k = 1:num_raw_stacks
        % Map linear index 'k' to Raw(Channel, Time)
        [rc, t] = ind2sub([RawC, T], k);

        % A. Load Raw Stack (Full Frame)
        rawStack = zeros(loader.Geometry.H, loader.Geometry.W, Z, 'uint16');
        for z = 1:Z
            rawStack(:,:,z) = loader.getFrame(t, z, rc);
        end

        % B. Define Base Output Channel ID
        % (Raw 1->0, Raw 2->2, Raw 3->4, etc.)
        base_out_ch = (rc - 1) * 2;

        % C. Apply Logic Based on Camera (Odd vs Even Raw Channel)
        % isOdd (Cam 1): Bottom=Low(0), Top=High(1)
        % isEven (Cam 2): Top=Low(2), Bottom=High(3)

        isCam1 = mod(rc, 2) ~= 0;

        if isCam1
            % --- CAMERA 1 LOGIC ---
            ch_Bot = base_out_ch;     % e.g. 0
            ch_Top = base_out_ch + 1; % e.g. 1
        else
            % --- CAMERA 2 LOGIC (Inverted) ---
            ch_Top = base_out_ch;     % e.g. 2
            ch_Bot = base_out_ch + 1; % e.g. 3
        end

        % D. Crop and Save BOTTOM
        stackBot = rawStack(roiBot.y, roiBot.x, :);
        if doRotate, stackBot = rot90(stackBot); end

        fNameBot = sprintf('img_C%02d_T%04d.tif', ch_Bot, t);
        writetiff(stackBot, fullfile(outDir, fNameBot));

        % E. Crop and Save TOP
        stackTop = rawStack(roiTop.y, roiTop.x, :);
        if doRotate, stackTop = rot90(stackTop); end

        fNameTop = sprintf('img_C%02d_T%04d.tif', ch_Top, t);
        writetiff(stackTop, fullfile(outDir, fNameTop));

        % (Logging first timepoint only to verify logic)
        if t == 1
             fprintf('     Worker: Raw C%d -> Top=C%d / Bot=C%d\n', rc, ch_Top, ch_Bot);
        end
    end
    t_end = toc;

    fprintf('   [Cropper] Complete in %.2f s. \n', t_end);
end

% --- Helper: Parse "y1:y2,x1:x2" ---
function roi = parseROI(str)
    parts = split(str, ',');
    yRange = str2num(parts{1}); %#ok<ST2NM>
    xRange = str2num(parts{2}); %#ok<ST2NM>
    roi.y = yRange; roi.x = xRange;
    roi.h = length(yRange); roi.w = length(xRange);
end
