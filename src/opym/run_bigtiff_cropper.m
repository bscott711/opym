function run_bigtiff_cropper(job)
% RUN_BIGTIFF_CROPPER Reads BigTiff via XML Map, crops, and saves split channels.
%
% LOGIC:
%   - Loader gets perfectly ordered frames (descrambled via XML).
%   - We iterate Raw Channels 1-4.
%   - Cam 1 (Raw Ch 1 & 3): Bottom=Low, Top=High
%   - Cam 2 (Raw Ch 2 & 4): Top=Low, Bottom=High (Inverted)

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

    % Output Directory
    if isfield(job, 'baseName')
        folderName = regexprep(job.baseName, '\.ome(\.tif)?$', '');
    else
        [~, folderName] = fileparts(masterFile);
    end
    outDir = fullfile(workDir, folderName);
    if ~exist(outDir, 'dir'), mkdir(outDir); end

    fprintf('   [Cropper] Source: %s\n', masterFile);

    % --- 2. Initialize XML Loader ---
    % Suppress annoying Tiff warnings on workers
    warning('off', 'MATLAB:imagesci:Tiff:libraryWarning');
    warning('off', 'MATLAB:imagesci:tifftagsread:expectedTagDataFormat');

    pool = gcp('nocreate');
    if isempty(pool), pool = parpool(48); end
    addAttachedFiles(pool, {'BigTiffFastLoader.m'});

    loader = BigTiffFastLoader(masterFile);

    % FIX: Do NOT use petakit 'readtiff', it only accepts 1 arg.
    % We rely on the internal 'readStdTiff' set by the class constructor.
    % loader.ReaderHandle = @(f, idx) readtiff(f, idx); <--- REMOVED

    T = loader.Dimensions.SizeT;
    Z = loader.Dimensions.SizeZ;
    RawC = loader.Dimensions.SizeC;

    num_raw_stacks = T * RawC;
    fprintf('   [Cropper] Processing %d Raw Stacks -> %d Output Channels\n', num_raw_stacks, RawC*2);

    % --- 3. Parallel Execution ---
    tic;
    parfor k = 1:num_raw_stacks
        % Suppress warnings inside workers too
        warning('off', 'MATLAB:imagesci:Tiff:libraryWarning');

        % Map linear index 'k' to Raw(Channel, Time)
        [rc, t] = ind2sub([RawC, T], k);

        % A. Load Frame (Loader handles descrambling automatically)
        rawStack = zeros(loader.Geometry.H, loader.Geometry.W, Z, 'uint16');
        for z = 1:Z
            rawStack(:,:,z) = loader.getFrame(t, z, rc);
        end

        % B. Determine Output Channel IDs
        % Base ID for this Raw Channel (e.g., Raw 1 -> 0, Raw 2 -> 2...)
        base_out_ch = (rc - 1) * 2;

        % Logic: Odd Raw Channels are Camera 1, Even are Camera 2
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

        % C. Crop & Save
        % Bottom
        stackBot = rawStack(roiBot.y, roiBot.x, :);
        if doRotate, stackBot = rot90(stackBot); end
        fNameBot = sprintf('img_C%02d_T%04d.tif', ch_Bot, t);
        writetiff(stackBot, fullfile(outDir, fNameBot));

        % Top
        stackTop = rawStack(roiTop.y, roiTop.x, :);
        if doRotate, stackTop = rot90(stackTop); end
        fNameTop = sprintf('img_C%02d_T%04d.tif', ch_Top, t);
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
