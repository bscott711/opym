classdef BigTiffFastLoader < handle
    % BIGTIFFFASTLOADER Reads OME-TIFFs using the embedded XML Map.
    %
    % LOGIC:
    %   1. Reads OME-XML from the master file header.
    %   2. Parses <TiffData> tags to map (T,Z,C) -> (File, IFD).
    %   3. Supports scrambled channels, multi-file splitting, and gaps.

    properties
        MasterFile      % Path to the master .ome.tif
        FileMap         % Cell array of file paths
        FrameMap        % Struct Array or Matrix: Map{t,z,c} -> [FileIdx, LocalIFD]
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

            % 1. Parse Metadata & Build Map
            fprintf('📖 Parsing OME-XML Map (this may take 2-3 seconds)...\n');
            obj.parseXMLMap();

            % 2. Default Reader (User should overwrite with petakit reader)
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
            % Map is stored as (C, Z, T) for efficiency or struct
            % We use linear indexing into the mapped arrays

            fIdx = obj.FrameMap.FileIdx(c, z, t);
            ifd  = obj.FrameMap.IFD(c, z, t);

            if fIdx == 0
                error('Frame T%d Z%d C%d is defined in XML but not mapped to a file.', t, z, c);
            end

            targetFile = obj.FileMap{fIdx};

            % 2. Read
            % Note: OME XML IFDs are 0-based. MATLAB Tiff is 1-based.
            % We add +1 here.
            try
                img = obj.ReaderHandle(targetFile, ifd + 1);
            catch ME
                warning(ME.identifier, '%s', ME.message);
                rethrow(ME);
            end
        end
    end

    methods (Access = private)
        function parseXMLMap(obj)
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
            % We use 3D arrays for instant lookup: (C, Z, T)
            sz = [obj.Dimensions.SizeC, obj.Dimensions.SizeZ, obj.Dimensions.SizeT];
            obj.FrameMap.FileIdx = zeros(sz, 'uint8'); % Up to 255 files
            obj.FrameMap.IFD     = zeros(sz, 'uint32');

            % 3. IDENTIFY ALL FILES IN SET
            % Find all siblings to map UUID filenames to real paths
            [masterDir, masterName, ~] = fileparts(obj.MasterFile);
            baseName = regexprep(masterName, '\.ome$', '');
            d = dir(fullfile(masterDir, '*.tif'));

            % Map "FileNameInXML" -> "RealFullPath"
            nameToPath = containers.Map;
            realFiles = {};

            % Populate file list
            for k=1:length(d)
                % Store just the filename as key
                nameToPath(d(k).name) = fullfile(masterDir, d(k).name);
                % Also store sequential index for the lookup array
                realFiles{end+1} = fullfile(masterDir, d(k).name); %#ok<AGROW>
            end
            obj.FileMap = realFiles;

            % Create a helper to map filename string to index in obj.FileMap
            % (Reverse lookup)
            nameToIdx = containers.Map;
            for k=1:length(realFiles)
                [~, n, e] = fileparts(realFiles{k});
                nameToIdx([n e]) = k;
            end

            % 4. REGEX PARSE TIFFDATA
            % Pattern: Looks for FirstC, FirstT, FirstZ, IFD, and UUID FileName
            % Note: This handles explicit TiffData entries (PlaneCount=1)

            % We extract blocks to handle the "FileName" context
            % Regex is complex; we assume standard OME layout:
            % <TiffData ... IFD="X" ...> <UUID FileName="Y">...

            % Faster approach: Extract all TiffData blocks
            pat = '<TiffData\s+FirstC="(\d+)"\s+FirstT="(\d+)"\s+FirstZ="(\d+)"\s+IFD="(\d+)".*?>\s*<UUID\s+FileName="([^"]+)"';
            tokens = regexp(xmlStr, pat, 'tokens');

            if isempty(tokens)
                % Try alternate ordering of attributes if first failed
                pat = 'FirstC="(\d+)".*?FirstT="(\d+)".*?FirstZ="(\d+)".*?IFD="(\d+)".*?FileName="([^"]+)"';
                tokens = regexp(xmlStr, pat, 'tokens');
            end

            if isempty(tokens)
                 error('Could not parse TiffData map. XML format might be unique.');
            end

            % 5. FILL THE MAP
            fprintf('   Mapping %d frames...\n', length(tokens));

            for k = 1:length(tokens)
                tk = tokens{k};
                c = str2double(tk{1}) + 1; % 1-based
                t = str2double(tk{2}) + 1;
                z = str2double(tk{3}) + 1;
                ifd = str2double(tk{4});   % 0-based
                fName = tk{5};

                % Resolve File Index
                if isKey(nameToIdx, fName)
                    fIdx = nameToIdx(fName);

                    % Update Map
                    obj.FrameMap.FileIdx(c, z, t) = fIdx;
                    obj.FrameMap.IFD(c, z, t)     = ifd;
                else
                    warning('File in XML not found on disk: %s', fName);
                end
            end
        end
    end
end

function img = readStdTiff(fPath, idx)
    t = Tiff(fPath, 'r');
    t.setDirectory(idx);
    img = t.read();
    t.close();
end
