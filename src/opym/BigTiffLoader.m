classdef BigTiffLoader < handle
    % BigTiffLoader Reads OME-TIFFs using the embedded XML Map.
    %
    % UPDATED: Aggressive warning suppression for Tiff constructor.

    properties
        MasterFile      % Path to the master .ome.tif
        FileMap         % Cell array of file paths
        FrameMap        % Struct containing FileIdx and IFD arrays
        Dimensions      % Struct with fields SizeT, SizeZ, SizeC
        Geometry        % Struct with W, H, BytesPerPixel
        VoxelSize       % Struct with x, y, z (in microns)
        ReaderHandle    % Function handle for the actual reading
    end

    methods
        function obj = BigTiffLoader(filePath)
            if ~isfile(filePath)
                error('BigTiffLoader:FileNotFound', 'File not found: %s', filePath);
            end
            obj.MasterFile = filePath;

            % 1. Parse Metadata & Build Map (Quietly)
            fprintf('📖 Parsing OME-XML Map (this may take 2-3 seconds)...\n');
            obj.parseXMLMap();

            % 2. Default Reader (Standard Tiff with Suppression)
            obj.ReaderHandle = @(f, idx) readStdTiff(f, idx);

            fprintf('✅ Loader initialized. Dimensions: T=%d, Z=%d, C=%d.\n', ...
                obj.Dimensions.SizeT, obj.Dimensions.SizeZ, obj.Dimensions.SizeC);
        end

        function img = getFrame(obj, t, z, c)
            % GETFRAME Retrieve slice using the XML Lookup Map.
            % Input: 1-based indices (Time, Z, Channel)

            if t > obj.Dimensions.SizeT || z > obj.Dimensions.SizeZ || c > obj.Dimensions.SizeC
                error('BigTiffLoader:OutOfBounds', ...
                      'Index out of bounds. Req: T%d Z%d C%d (Max: %d %d %d)', ...
                      t, z, c, obj.Dimensions.SizeT, obj.Dimensions.SizeZ, obj.Dimensions.SizeC);
            end

            % 1. Look up File Index and Local IFD
            fIdx = obj.FrameMap.FileIdx(c, z, t);
            ifd  = obj.FrameMap.IFD(c, z, t);

            if fIdx == 0
                error('Frame T%d Z%d C%d is defined in XML but not mapped to a file.', t, z, c);
            end

            targetFile = obj.FileMap{fIdx};

            % 2. Read Frame
            try
                % OME XML is 0-based. MATLAB is 1-based.
                idx = double(ifd) + 1;
                img = obj.ReaderHandle(targetFile, idx);
            catch ME
                % If we fail, try to print just the message, not the stack
                warning('Read failed on %s (IFD %d): %s', targetFile, ifd, ME.message);
                rethrow(ME);
            end
        end
    end

    methods (Access = private)
        function parseXMLMap(obj)
            % SUPPRESS ALL WARNINGS during metadata read (Nuclear Option)
            wState = warning('off', 'all');

            try
                % READ OME HEADER
                t = Tiff(obj.MasterFile, 'r');
                try
                    xmlStr = t.getTag('ImageDescription');
                catch
                    % Sometimes the tag ID is different?
                    % Try looking for any string tag if standard fails
                    warning('Could not read ImageDescription directly.');
                    xmlStr = '';
                end
                t.close();

                % Restore warnings immediately
                warning(wState);

                if isempty(xmlStr)
                    error('XML Header is empty.');
                end

                % 1. PARSE DIMENSIONS
                obj.Dimensions.SizeC = 1;
                obj.Dimensions.SizeZ = 1;
                obj.Dimensions.SizeT = 1;

                tok = regexp(xmlStr, 'SizeC="(\d+)"', 'tokens');
                if ~isempty(tok), obj.Dimensions.SizeC = str2double(tok{1}{1}); end
                tok = regexp(xmlStr, 'SizeZ="(\d+)"', 'tokens');
                if ~isempty(tok), obj.Dimensions.SizeZ = str2double(tok{1}{1}); end
                tok = regexp(xmlStr, 'SizeT="(\d+)"', 'tokens');
                if ~isempty(tok), obj.Dimensions.SizeT = str2double(tok{1}{1}); end

                % 2. PARSE VOXEL SIZE
                obj.VoxelSize = struct('x', 0.1, 'y', 0.1, 'z', 1.0, 'unit', 'µm');

                tokX = regexp(xmlStr, 'PhysicalSizeX="([\d\.]+)"', 'tokens');
                if ~isempty(tokX), obj.VoxelSize.x = str2double(tokX{1}{1}); end

                tokY = regexp(xmlStr, 'PhysicalSizeY="([\d\.]+)"', 'tokens');
                if ~isempty(tokY), obj.VoxelSize.y = str2double(tokY{1}{1}); end

                tokZ = regexp(xmlStr, 'PhysicalSizeZ="([\d\.]+)"', 'tokens');
                if ~isempty(tokZ), obj.VoxelSize.z = str2double(tokZ{1}{1}); end

                % Geometry (Suppress here too)
                warning('off', 'all');
                tTemp = Tiff(obj.MasterFile, 'r');
                obj.Geometry.W = double(tTemp.getTag('ImageWidth'));
                obj.Geometry.H = double(tTemp.getTag('ImageLength'));
                tTemp.close();
                warning(wState);

                % 3. PRE-ALLOCATE MAP
                sz = [obj.Dimensions.SizeC, obj.Dimensions.SizeZ, obj.Dimensions.SizeT];
                obj.FrameMap.FileIdx = zeros(sz, 'uint32');
                obj.FrameMap.IFD     = zeros(sz, 'uint32');

                % 4. IDENTIFY ALL FILES
                [masterDir, masterName, ~] = fileparts(obj.MasterFile);
                d = dir(fullfile(masterDir, '*.tif'));

                nameToPath = containers.Map;
                realFiles = {};

                for k=1:length(d)
                    nameToPath(d(k).name) = fullfile(masterDir, d(k).name);
                    realFiles{end+1} = fullfile(masterDir, d(k).name); %#ok<AGROW>
                end
                obj.FileMap = realFiles;

                nameToIdx = containers.Map;
                for k=1:length(realFiles)
                    [~, n, e] = fileparts(realFiles{k});
                    nameToIdx([n e]) = k;
                end

                % 5. REGEX PARSE TIFFDATA
                pat = 'FirstC="(\d+)".*?FirstT="(\d+)".*?FirstZ="(\d+)".*?IFD="(\d+)".*?FileName="([^"]+)"';
                tokens = regexp(xmlStr, pat, 'tokens');

                if isempty(tokens)
                    error('Could not parse TiffData map from XML.');
                end

                fprintf('   Mapping %d frames from XML...\n', length(tokens));

                for k = 1:length(tokens)
                    tk = tokens{k};
                    c = str2double(tk{1}) + 1;
                    t = str2double(tk{2}) + 1;
                    z = str2double(tk{3}) + 1;
                    ifd = str2double(tk{4});
                    fName = tk{5};

                    if isKey(nameToIdx, fName)
                        fIdx = nameToIdx(fName);
                        obj.FrameMap.FileIdx(c, z, t) = fIdx;
                        obj.FrameMap.IFD(c, z, t)     = ifd;
                    else
                        % warning('File in XML not found on disk: %s', fName);
                    end
                end

            catch ME
                warning(wState); % Always restore warnings
                rethrow(ME);
            end

            warning(wState);
        end
    end
end

function img = readStdTiff(fPath, idx)
    % NUCLEAR OPTION: Suppress ALL warnings during Tiff access
    % This is required because the Tiff constructor itself triggers warnings
    % before we can disable specific IDs.

    wState = warning('off', 'all');

    try
        t = Tiff(fPath, 'r');
        t.setDirectory(idx);
        img = t.read();
        t.close();
    catch ME
        warning(wState); % Restore
        rethrow(ME);
    end

    warning(wState); % Restore
end
