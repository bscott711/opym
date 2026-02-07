classdef BigTiffFastLoader < handle
    % BIGTIFFFASTLOADER Maps and reads interleaved OME-TIFFs without slow scanning.
    %
    % Usage:
    %   loader = BigTiffFastLoader('path/to/file.ome.tif');
    %   img = loader.getFrame(t, z, c);
    %   img = loader.getFrame(1, 50, 2); % 1-based indexing

    properties
        MasterFile      % Path to the master .ome.tif
        FileMap         % Struct array of file chunks and their frame ranges
        Dimensions      % Struct with fields SizeT, SizeZ, SizeC
        Geometry        % Struct with W, H, BytesPerPixel
        TotalFrames     % Total frames found across all files
        ReaderHandle    % Function handle for the actual reading
    end

    methods
        function obj = BigTiffFastLoader(filePath)
            % Constructor: Parses metadata and builds the fast map
            if ~isfile(filePath)
                error('BigTiffFastLoader:FileNotFound', 'File not found: %s', filePath);
            end
            obj.MasterFile = filePath;

            % 1. Parse Dimensions & Geometry
            obj.parseMetadata();

            % 2. Map Files (The Fast Heuristic)
            obj.mapFiles();

            % 3. Define the Reader
            % Using an anonymous function to wrap petakit5d or standard Tiff
            % NOTE: Verify if your reader expects 0-based or 1-based index.
            obj.ReaderHandle = @(f, idx) petakit5d.readTiff(f, idx);

            fprintf('✅ Loader initialized. Found %d frames (Exp: %d).\n', ...
                obj.TotalFrames, prod([obj.Dimensions.SizeT, obj.Dimensions.SizeZ, obj.Dimensions.SizeC]));
        end

        function img = getFrame(obj, t, z, c)
            % GETFRAME Retrieve a specific slice (1-based indexing)
            % Mapping: Interleaved [Channel -> Z -> Time]

            % 1. Input Validation
            if t > obj.Dimensions.SizeT || z > obj.Dimensions.SizeZ || c > obj.Dimensions.SizeC
                error('BigTiffFastLoader:OutOfBounds', ...
                      'Index out of bounds. Max: T=%d, Z=%d, C=%d', ...
                      obj.Dimensions.SizeT, obj.Dimensions.SizeZ, obj.Dimensions.SizeC);
            end

            % 2. Calculate Global Linear Index (0-based for internal math)
            % Sequence: Channel changes fastest, then Z, then Time.
            % GlobalIdx = (t-1)*(Z*C) + (z-1)*(C) + (c-1)

            sz_z = obj.Dimensions.SizeZ;
            sz_c = obj.Dimensions.SizeC;

            global_idx_0 = (t-1) * (sz_z * sz_c) + ...
                           (z-1) * (sz_c) + ...
                           (c-1);

            % 3. Find which file contains this index
            targetFile = [];
            localIdx = -1;

            for k = 1:length(obj.FileMap)
                fm = obj.FileMap(k);
                if global_idx_0 >= fm.start_idx && global_idx_0 <= fm.end_idx
                    targetFile = fm.path;
                    % Local index within that file (1-based for MATLAB/Tiff readers)
                    localIdx = (global_idx_0 - fm.start_idx) + 1;
                    break;
                end
            end

            if isempty(targetFile)
                error('BigTiffFastLoader:FrameNotFound', ...
                      'Frame %d (T%d Z%d C%d) not found in file map.', global_idx_0, t, z, c);
            end

            % 4. Call the external reader
            try
                img = obj.ReaderHandle(targetFile, localIdx);
            catch ME
                error('BigTiffFastLoader:ReadError', ...
                      'Read failed on %s (Idx: %d): %s', targetFile, localIdx, ME.message);
            end
        end
    end

    methods (Access = private)
        function parseMetadata(obj)
            % Reads OME-XML headers only
            t = Tiff(obj.MasterFile, 'r');
            try
                desc = t.getTag('ImageDescription');
                % Defaults
                obj.Dimensions.SizeC = 1;
                obj.Dimensions.SizeZ = 1;
                obj.Dimensions.SizeT = 1;

                % Regex Parse
                tokC = regexp(desc, 'SizeC="(\d+)"', 'tokens');
                if ~isempty(tokC), obj.Dimensions.SizeC = str2double(tokC{1}{1}); end

                tokZ = regexp(desc, 'SizeZ="(\d+)"', 'tokens');
                if ~isempty(tokZ), obj.Dimensions.SizeZ = str2double(tokZ{1}{1}); end

                tokT = regexp(desc, 'SizeT="(\d+)"', 'tokens');
                if ~isempty(tokT), obj.Dimensions.SizeT = str2double(tokT{1}{1}); end

                % Geometry
                obj.Geometry.W = double(t.getTag('ImageWidth'));
                obj.Geometry.H = double(t.getTag('ImageLength'));
                bps = double(t.getTag('BitsPerSample'));
                obj.Geometry.BytesPerFrame = obj.Geometry.W * obj.Geometry.H * (bps / 8);

            catch ME
                % FIXED: Added Message Identifier as first argument
                warning('BigTiffFastLoader:MetadataParsing', ...
                        'Metadata parsing failed: %s', ME.message);
            end
            t.close();
        end

        function mapFiles(obj)
            % Identify siblings and map ranges based on file size
            [fDir, fName, ~] = fileparts(obj.MasterFile);
            basePattern = regexprep(fName, '\.ome$', '');
            d = dir(fullfile(fDir, '*.tif'));

            % Find valid files
            files = struct('path', {}, 'idx_suffix', {}, 'bytes', {});
            safeBase = regexptranslate('escape', basePattern);

            count = 0;
            for k = 1:length(d)
                fn = d(k).name;
                isMaster = ~isempty(regexp(fn, ['^' safeBase '\.ome\.tif$'], 'once'));
                tokSib = regexp(fn, ['^' safeBase '_(\d+)\.ome\.tif$'], 'tokens');

                suffix = -1;
                if isMaster, suffix = 0;
                elseif ~isempty(tokSib), suffix = str2double(tokSib{1}{1});
                end

                if suffix >= 0
                    count = count + 1;
                    files(count).path = fullfile(d(k).folder, fn);
                    files(count).idx_suffix = suffix;
                    files(count).bytes = d(k).bytes;
                end
            end

            % Sort by suffix
            [~, I] = sort([files.idx_suffix]);
            files = files(I);

            % Map Ranges
            current_start = 0; % 0-based tracking
            for k = 1:length(files)
                % The Fast Heuristic: FileSize / FrameSize
                num_frames = floor(files(k).bytes / obj.Geometry.BytesPerFrame);

                files(k).start_idx = current_start;
                files(k).end_idx = current_start + num_frames - 1;
                files(k).count = num_frames;

                current_start = current_start + num_frames;
            end

            obj.FileMap = files;
            obj.TotalFrames = current_start;
        end
    end
end
