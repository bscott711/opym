function run_gpu_pipeline_async(activePath, done_dir, fail_dir, val_shm, outFn, psfFn, gpu_lock_dir, lockJobName, varargin)
    % Release the GPU concurrency lock no matter how this function exits
    % (normal return, caught error, or an uncaught crash e.g. CUDA OOM) so a
    % failed job can never permanently starve the queue.
    lockName = fullfile(gpu_lock_dir, [lockJobName '.lock']);
    cleanupLock = onCleanup(@() delete_if_exists(lockName)); %#ok<NASGU>

    try
        run_gpu_pipeline(val_shm, outFn, psfFn, varargin{:});
        [~, jobfname, jobext] = fileparts(activePath);
        finalName = strrep([jobfname jobext], '.active_', '');
        movefile(activePath, fullfile(done_dir, finalName));
        try rmdir(val_shm, 's'); catch; end
    catch ME
        [~, fname, ext] = fileparts(activePath);
        finalName = strrep([fname ext], '.active_', '');
        movefile(activePath, fullfile(fail_dir, finalName));
        errLog = fullfile(fail_dir, [strrep(fname, '.active_', '') '.log']);
        fid = fopen(errLog, 'w');
        fprintf(fid, '%s\n', getReport(ME));
        fclose(fid);
        try rmdir(val_shm, 's'); catch; end
    end
end

function delete_if_exists(p)
    if exist(p, 'file')
        try
            delete(p);
        catch
        end
    end
end
