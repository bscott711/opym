function run_bigtiff_cropper(job)
% RUN_BIGTIFF_CROPPER Reads BigTiff, crops, copies sidecars, and writes valid OME-TIFFs.
%
% FEATURES:
%   - Sidecar Preservation: Copies txt/json/xml metadata to output.
%   - OME Injection: Writes full OME-XML *only* in the first frame header.
%   - 0-Based Indexing: Output files are T0000 compatible.

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

    % Output Directory & Naming
    if isfield(job, 'baseName')
        cleanName = regexprep(job.baseName, '\.ome(\.tif)?$', '');
    else
        [~, cleanName] = fileparts(masterFile);
    end

    outDir = fullfile(workDir, cleanName);
    if ~exist(outDir, 'dir'), mkdir(outDir); end

    fprintf('   [Cropper] Source: %s\n', masterFile);
    fprintf('   [Cropper] Output: %s\n', outDir);

    % --- 2. METADATA RECOVERY (Sidecar Copy) ---
    fprintf('   [Metadata] Scanning for sidecar files (txt, json, xml)...\n');
    % Look for common metadata files in the source directory
    sidecars = [dir(fullfile(workDir, '*.txt')); ...
                dir(fullfile(workDir, '*.json')); ...
                dir(fullfile(workDir, '*.xml'))];

    countSide = 0;
    for k = 1:length(sidecars)
        fName = sidecars(k).name;
        % Skip the master OME-TIFF itself and directories
        if contains(fName, '.ome.tif', 'IgnoreCase', true) || sidecars(k).isdir, continue; end

        srcSide = fullfile(workDir, fName);
        dstSide = fullfile(outDir, fName);

        % Copy if it doesn't exist
        if ~exist(dstSide, 'file')
            copyfile(srcSide, dstSide);
            countSide = countSide + 1;
        end
    end
    fprintf('      -> Copied %d sidecar files.\n', countSide);

    % --- 3. Initialize Loader ---
    warning('off', 'MATLAB:imagesci:Tiff:libraryWarning');
    warning('off', 'MATLAB:imagesci:tifftagsread:expectedTagDataFormat');

    pool = gcp('nocreate');
    if isempty(pool), pool = parpool(48); end
    addAttachedFiles(pool, {'BigTiffFastLoader.m'});

    loader = BigTiffFastLoader(masterFile);

    % Extract Voxel Size (Requires updated BigTiffFastLoader)
    if isprop(loader, 'VoxelSize')
        vox = loader.VoxelSize;
    else
        % Fallback if class not updated
        vox = struct('x', 0.1, 'y', 0.1, 'z', 1.0, 'unit', 'µm');
    end
    fprintf('   [Metadata] Voxel Size: %.3f x %.3f x %.3f %s\n', vox.x, vox.y, vox.z, vox.unit);

    T = loader.Dimensions.SizeT;
    Z = loader.Dimensions.SizeZ;
    RawC = loader.Dimensions.SizeC;

    num_raw_stacks = T * RawC;
    fprintf('   [Cropper] Processing %d Raw Stacks -> %d Output Channels\n', num_raw_stacks, RawC*2);

    % --- 4. Parallel Execution ---
    tic;
    parfor k = 1:num_raw_stacks
        warning('off', 'MATLAB:imagesci:Tiff:libraryWarning');

        [rc, t_in] = ind2sub([RawC, T], k);

        % A. Load Frame
        rawStack = zeros(loader.Geometry.H, loader.Geometry.W, Z, 'uint16');
        for z = 1:Z
            rawStack(:,:,z) = loader.getFrame(t_in, z, rc);
        end

        % B. Determine Channels
        base_out_ch = (rc - 1) * 2;
        isCam1 = mod(rc, 2) ~= 0;

        if isCam1
            ch_Bot = base_out_ch;
            ch_Top = base_out_ch + 1;
        else
            ch_Top = base_out_ch;
            ch_Bot = base_out_ch + 1;
        end

        % C. Write BOTTOM
        stackBot = rawStack(roiBot.y, roiBot.x, :);
        if doRotate, stackBot = rot90(stackBot); end
        fNameBot = sprintf('%s_C%02d_T%04d.tif', cleanName, ch_Bot, t_in - 1);
        write_ome_tiff_stack(stackBot, fullfile(outDir, fNameBot), vox, ch_Bot, t_in-1);

        % D. Write TOP
        stackTop = rawStack(roiTop.y, roiTop.x, :);
        if doRotate, stackTop = rot90(stackTop); end
        fNameTop = sprintf('%s_C%02d_T%04d.tif', cleanName, ch_Top, t_in - 1);
        write_ome_tiff_stack(stackTop, fullfile(outDir, fNameTop), vox, ch_Top, t_in-1);
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

function write_ome_tiff_stack(stack, fpath, vox, c, t)
    % Writes a 3D Stack with OME-XML Metadata (First Frame Only)
    [H, W, Z] = size(stack);
    [~, fName, ext] = fileparts(fpath);
    fileName = [fName, ext];

    % 1. Construct Minimal OME-XML
    uuid = char(java.util.UUID.randomUUID());

    % Note: TiffData PlaneCount=Z tells reader this file holds the whole stack
    xmlStr = sprintf([...
        '<?xml version="1.0" encoding="UTF-8"?>' ...
        '<OME xmlns="http://www.openmicroscopy.org/Schemas/OME/2016-06" ' ...
        'xmlns:xsi="http://www.w3.org/2001/XMLSchema-instance" ' ...
        'xsi:schemaLocation="http://www.openmicroscopy.org/Schemas/OME/2016-06 http://www.openmicroscopy.org/Schemas/OME/2016-06/ome.xsd">' ...
        '<Image ID="Image:0" Name="%s">' ...
        '<Pixels ID="Pixels:0" DimensionOrder="XYCZT" Type="uint16" ' ...
        'SizeX="%d" SizeY="%d" SizeZ="%d" SizeC="1" SizeT="1" ' ...
        'PhysicalSizeX="%.3f" PhysicalSizeY="%.3f" PhysicalSizeZ="%.3f" PhysicalSizeXUnit="%s" PhysicalSizeYUnit="%s" PhysicalSizeZUnit="%s">' ...
        '<Channel ID="Channel:0:0" Name="Channel %d" />' ...
        '<TiffData FirstC="0" FirstT="0" FirstZ="0" IFD="0" PlaneCount="%d">' ...
        '<UUID FileName="%s">%s</UUID>' ...
        '</TiffData>' ...
        '</Pixels>' ...
        '</Image>' ...
        '</OME>'], ...
        fName, W, H, Z, vox.x, vox.y, vox.z, vox.unit, vox.unit, vox.unit, c, Z, fileName, uuid);

    % 2. Setup Tags
    tObj = Tiff(fpath, 'w');

    tagStruct.ImageLength = H;
    tagStruct.ImageWidth = W;
    tagStruct.Photometric = Tiff.Photometric.MinIsBlack;
    tagStruct.BitsPerSample = 16;
    tagStruct.SamplesPerPixel = 1;
    tagStruct.RowsPerStrip = H; % 1 strip per image is efficient for reading
    tagStruct.PlanarConfiguration = Tiff.PlanarConfiguration.Chunky;
    tagStruct.Software = 'Opym PetaKit Cropper';
    tagStruct.Compression = Tiff.Compression.None;

    % --- FRAME 1: Write WITH Metadata ---
    tagStruct.ImageDescription = xmlStr;
    tObj.setTag(tagStruct);
    tObj.write(stack(:,:,1));

    % --- FRAMES 2..Z: Write WITHOUT Metadata ---
    if Z > 1
        % Remove the heavy XML for subsequent frames
        tagStruct = rmfield(tagStruct, 'ImageDescription');

        for k = 2:Z
            tObj.writeDirectory(); % Create next IFD
            tObj.setTag(tagStruct);
            tObj.write(stack(:,:,k));
        end
    end

    tObj.close();
end
