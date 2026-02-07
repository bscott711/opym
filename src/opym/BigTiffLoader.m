classdef BigTiffFastLoader < handle
    % BIGTIFFFASTLOADER Reads OME-TIFFs using the embedded XML Map.
    %
    % LOGIC:
    %   1. Reads OME-XML from the master file header.
    %   2. Parses <TiffData> tags to map (T,Z,C) -> (File, IFD).
    %   3. Automatically handles hardware scrambling and multi-file splits.
    %   4. QUIET MODE: Suppresses custom tag warnings from the Tiff library.

    properties
        MasterFile      % Path to the master .ome.tif
        FileMap         % Cell array of file paths
        FrameMap        % Struct containing FileIdx and IFD arrays
        Dimensions      % Struct with fields SizeT, SizeZ, SizeC
        Geometry        % Struct with W, H, BytesPerPixel
        ReaderHandle    % Function handle for the actual reading
    end

    methods
        function obj = BigTiffFastLoader(filePath)
            if ~isfile(filePath)
                error('BigTiffFastLoader:FileNotFound', 'File not found: %s', filePath);
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
                error('BigTiffFastLoader:OutOfBounds', ...
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
                warning(ME.identifier, '%s', ME.message);
                rethrow(ME);
            end
        end
    end

    methods (Access = private)
        function parseXMLMap(obj)
            % SUPPRESS WARNINGS during metadata read
            wState1 = warning('off', 'MATLAB:imagesci:Tiff:libraryWarning');
            wState2 = warning('off', 'MATLAB:imagesci:tifftagsread:expectedTagDataFormat');

            try
                % READ OME HEADER
                t = Tiff(obj.MasterFile, 'r');
                try
                    xmlStr = t.getTag('ImageDescription');
                catch
                    error('Could not read ImageDescription tag.');
                end
                t.close();

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

                % Geometry
                tTemp = Tiff(obj.MasterFile, 'r');
                obj.Geometry.W = double(tTemp.getTag('ImageWidth'));
                obj.Geometry.H = double(tTemp.getTag('ImageLength'));
                tTemp.close();

                % 2. PRE-ALLOCATE MAP
                sz = [obj.Dimensions.SizeC, obj.Dimensions.SizeZ, obj.Dimensions.SizeT];
                obj.FrameMap.FileIdx = zeros(sz, 'uint8');
                obj.FrameMap.IFD     = zeros(sz, 'uint32');

                % 3. IDENTIFY ALL FILES
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

                % 4. REGEX PARSE TIFFDATA
                pat = 'FirstC="(\d+)".*?FirstT="(\d+)".*?FirstZ="(\d+)".*?IFD="(\d+)".*?FileName="([^"]+)"';
                tokens = regexp(xmlStr, pat, 'tokens');

                if isempty(tokens)
                    error('Could not parse TiffData map from XML.');
                end

                fprintf('   Mapping %d frames from XML...\n', length(tokens));

                for k = 1:length(tokens)
                    tk = tokens{k};
                    c = str2double(tk{1}) + 1; % 1-based
                    t = str2double(tk{2}) + 1;
                    z = str2double(tk{3}) + 1;
                    ifd = str2double(tk{4});   % 0-based
                    fName = tk{5};

                    if isKey(nameToIdx, fName)
                        fIdx = nameToIdx(fName);
                        obj.FrameMap.FileIdx(c, z, t) = fIdx;
                        obj.FrameMap.IFD(c, z, t)     = ifd;
                    else
                        warning('File in XML not found on disk: %s', fName);
                    end
                end

            catch ME
                % Restore warnings on error
                warning(wState1);
                warning(wState2);
                rethrow(ME);
            end

            % Restore warnings after success (optional)
            warning(wState1);
            warning(wState2);
        end
    end
end

function img = readStdTiff(fPath, idx)
    % SILENCE WARNINGS for each read operation
    wState1 = warning('off', 'MATLAB:imagesci:Tiff:libraryWarning');
    wState2 = warning('off', 'MATLAB:imagesci:tifftagsread:expectedTagDataFormat');

    try
        t = Tiff(fPath, 'r');
        t.setDirectory(idx);
        img = t.read();
        t.close();
    catch ME
        warning(wState1);
        rethrow(ME);
    end

    warning(wState1);
    warning(wState2);
end
