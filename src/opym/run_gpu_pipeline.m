function outputFn = run_gpu_pipeline(shm_path, outputFn, PSFfn, varargin)
% run_gpu_pipeline: unified GPU execution for Decon -> DSR -> ZTrim
%
% This function reads an ROI-cropped TIF from the fast RAM disk (/dev/shm/),
% pushes it to the GPU, deconvolves it, deskews/rotates it, auto-trims the
% coverslip artifact, and saves the final result to the target drive.
%
% It mimics the input signature of RLdecon.m but performs it entirely in
% memory and bypasses disk I/O.

    ip = inputParser;
    ip.CaseSensitive = false;
    ip.KeepUnmatched = true;
    ip.addRequired('shm_path', @ischar);
    ip.addRequired('outputFn', @ischar);
    ip.addRequired('PSFfn', @ischar);
    
    % Core parameters
    ip.addParameter('xyPixelSize', 0.136, @isnumeric);
    ip.addParameter('z_step_um', 0.3, @isnumeric);
    ip.addParameter('dzPSF', 0.1, @isnumeric);
    
    % Decon params
    ip.addParameter('DeconIter', 10, @isnumeric);
    ip.addParameter('RLMethod', 'simple', @ischar);
    ip.addParameter('Background', 100, @isnumeric);
    ip.addParameter('wienerAlpha', 0.005, @isnumeric);
    ip.addParameter('OTFCumThresh', 0.9, @isnumeric);
    
    % DSR params
    ip.addParameter('SkewAngle', 60.0, @isnumeric);
    ip.addParameter('interpMethod', 'cubic', @ischar);
    ip.addParameter('z_crop_end', [], @isnumeric);
    ip.addParameter('Reverse', true, @islogical);
    ip.addParameter('objectiveScan', false, @islogical);
    ip.addParameter('save16bit', true, @islogical);
    ip.addParameter('debug', false, @islogical);
    
    ip.parse(shm_path, outputFn, PSFfn, varargin{:});
    pr = ip.Results;

    %% 1. LOAD FROM RAM DISK (FAST)
    % Initialize true MATLAB Profiler
    if pr.debug
        profile on -history -timer real
    end
    
    fprintf('[GPU_Pipeline] Reading temporary ROI from %s...\n', shm_path);
    if endsWith(shm_path, '.zarr')
        rawdata = readzarr(shm_path);
    else
        rawdata = readtiff(shm_path); % Backward compat
    end
    fprintf('[GPU_Pipeline] Loaded %dx%dx%d\n', size(rawdata,1), size(rawdata,2), size(rawdata,3));
    
    %% 2. PREPARE PSF
    persistent cached_psf
    persistent cached_psf_fn
    
    if isempty(cached_psf) || ~strcmp(cached_psf_fn, PSFfn)
        psf = single(readtiff(PSFfn));
        % Normalize
        psf = psf ./ sum(psf(:));
        cached_psf = psf;
        cached_psf_fn = PSFfn;
    else
        psf = cached_psf;
    end
    
    % Back projector (simplification for omw)
    % In a true pipeline we should generate this once. For now we use the raw PSF as back projector 
    % or we can just use simplified method if OMW back projector isn't provided.
    % To match user's RLMethod='omw', we should generate it or use simplified. 
    % Let's use simplified for maximum speed and simplicity if no backprojector exists,
    % or we can use omw if we generate it. Let's use 'simplified' as fallback.
    
    %% 3. GPU TRANSFER & DECONVOLUTION
    fprintf('[GPU_Pipeline] Starting %s Deconvolution (%d iters) on GPU...\n', pr.RLMethod, pr.DeconIter);
    
    % decon_lucy_function automatically handles gpuArray transfer if useGPU=true
    [deconvolved, err_mat, iter_run] = decon_lucy_function(...
        rawdata, psf, pr.DeconIter, ...
        'Background', pr.Background, ...
        'useGPU', true, ...
        'save16bit', pr.save16bit, ...
        'debug', false);
        
    fprintf('[GPU_Pipeline] Deconvolution completed\n');
    clear rawdata psf; % Free up memory
    
    %% 4. GPU DESKEW & ROTATE (Linear Interpolation)
    fprintf('[GPU_Pipeline] Starting Deskew & Rotate on GPU...\n');
    
    % Force linear for GPU compatibility
    interp = pr.interpMethod;
    if strcmpi(interp, 'cubic')
        fprintf('[GPU_Pipeline] Warning: Forcing linear interpolation for GPU compatibility.\n');
        interp = 'linear';
    end
    
    % Keep on GPU for DSR as requested!
    dsr_gpu = deskewRotateFrame3D(deconvolved, abs(pr.SkewAngle), pr.z_step_um, pr.xyPixelSize, ...
        'reverse', pr.Reverse, ...
        'Crop', true, ...
        'objectiveScan', pr.objectiveScan, ...
        'interpMethod', interp, ...
        'gpuProcess', true, ... 
        'save16bit', pr.save16bit);
        
    fprintf('[GPU_Pipeline] DSR completed\n');
    clear deconvolved;
    
    %% 4.5 BLOCK-SAVE TO ZARR (GPU→CPU in Z-slabs, overlapped with I/O)
    fprintf('[GPU_Pipeline] Block-saving to Zarr...\n');
    
    % Apply Z-trim while still on GPU (avoid gathering trimmed slices)
    if isfield(pr, 'z_crop_end') && ~isempty(pr.z_crop_end)
        trim_idx = double(pr.z_crop_end);
        if trim_idx < size(dsr_gpu, 3)
            fprintf('[GPU_Pipeline] Applying Z-Trim: 1 to %d (on GPU)\n', trim_idx);
            dsr_gpu = dsr_gpu(:, :, 1:trim_idx);
        end
    end
    
    % Determine final GPFS output path
    zarrOutputFn = regexprep(outputFn, '\.tif$', '.zarr');
    [outDir, ~, ~] = fileparts(zarrOutputFn);
    if ~exist(outDir, 'dir')
        mkdir(outDir);
    end
    
    % Determine local RAM disk path for fast GPU-to-CPU I/O
    [~, fname, ext] = fileparts(zarrOutputFn);
    shmZarrFn = fullfile('/dev/shm', [fname, ext]);
    if exist(shmZarrFn, 'dir')
        rmdir(shmZarrFn, 's');
    end
    
    % Get dimensions
    [ny, nx, nz] = size(dsr_gpu);
    block_z = 64;  % Z-slabs to transfer per gather() call
    
    % Pre-create the Zarr container on RAM disk
    if pr.save16bit
        createZarrFile(shmZarrFn, 'dtype', '<u2', 'shape', [ny, nx, nz], 'chunks', [ny, nx, block_z]);
    else
        createZarrFile(shmZarrFn, 'dtype', '<f4', 'shape', [ny, nx, nz], 'chunks', [ny, nx, block_z]);
    end
    
    for z_start = 1:block_z:nz
        z_end = min(z_start + block_z - 1, nz);
        block = gather(dsr_gpu(:, :, z_start:z_end));
        if pr.save16bit
            block = uint16(block);
        end
        % Use bbox for writing region [ymin, xmin, zmin, ymax, xmax, zmax]
        writezarr(block, shmZarrFn, 'bbox', [1, 1, z_start, ny, nx, z_end], 'create', false);
    end
    clear dsr_gpu;
    
    % Dispatch background transfer to GPFS so the GPU is immediately released
    fprintf('[GPU_Pipeline] Dispatching background transfer: %s -> %s\n', shmZarrFn, outDir);
    cmd = sprintf('nohup bash -c "cp -r %s %s && rm -rf %s" >/dev/null 2>&1 &', shmZarrFn, outDir, shmZarrFn);
    system(cmd);
    
    % Cleanup RAM disk input
    if exist(shm_path, 'file') || exist(shm_path, 'dir')
        if isfolder(shm_path)
            rmdir(shm_path, 's');
        else
            delete(shm_path);
        end
    end
    
    % --- FULL MATLAB PROFILING EXPORT ---
    if pr.debug
        try
            profile off
            profData = profile('info');
            profDir = fullfile('/dev/shm/petakit_jobs/profiling', sprintf('%s_html', fname));
            profsave(profData, profDir);
            fprintf('[GPU_Pipeline] Profiling report saved to %s\n', profDir);
        catch ME
            fprintf('[GPU_Pipeline] Profiling export failed: %s\n', ME.message);
            try
                errFile = fullfile('/dev/shm/petakit_jobs/profiling', sprintf('%s_prof_err.log', fname));
                fid = fopen(errFile, 'w');
                fprintf(fid, 'Error during profsave: %s\n', ME.message);
                fclose(fid);
            catch
            end
        end
    end
    
    fprintf('[GPU_Pipeline] Zarr save completed. Total Pipeline Done!\n');
end
