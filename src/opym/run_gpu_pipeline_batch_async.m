function run_gpu_pipeline_batch_async(activePath, done_dir, fail_dir, items, psfFn, gpu_lock_dir, lockJobName, varargin)
    % Batch counterpart of run_gpu_pipeline_async.m: processes N
    % (shm_path, output_file) pairs that share one PSF inside a single
    % parfeval'd call, holding the GPU concurrency lock once for the whole
    % batch instead of once per frame. Each item still runs through the
    % exact same run_gpu_pipeline() call with identical arguments -- this
    % only changes dispatch granularity, never per-frame math.
    %
    % Per-item failure isolation: one bad item (e.g. corrupt shm input,
    % CUDA OOM) does not prevent its batch-mates from being computed,
    % saved, and reported done -- extending the same "a failed job can
    % never permanently starve the queue" philosophy run_gpu_pipeline_async.m
    % already applies at the whole-ticket level, down to per-item granularity.

    % Release the GPU concurrency lock no matter how this function exits
    % (normal return, caught error, or an uncaught crash e.g. CUDA OOM).
    lockName = fullfile(gpu_lock_dir, [lockJobName '.lock']);
    cleanupLock = onCleanup(@() delete_if_exists(lockName)); %#ok<NASGU>

    [~, jobfname, jobext] = fileparts(activePath);
    finalName = strrep([jobfname jobext], '.active_', '');

    n = numel(items);
    statuses = cell(1, n);
    errors = cell(1, n);

    for i = 1:n
        try
            run_gpu_pipeline(items(i).shm_path, items(i).output_file, psfFn, varargin{:});
            statuses{i} = 'done';
            errors{i} = '';
        catch ME
            statuses{i} = 'failed';
            errors{i} = getReport(ME);
            fprintf('[Batch] Item %d/%d failed (%s): %s\n', i, n, items(i).output_file, ME.message);
        end
        try rmdir(items(i).shm_path, 's'); catch; end
    end

    nFailed = sum(strcmp(statuses, 'failed'));
    if nFailed == n
        % Every item failed -- whole ticket counts as failed.
        dispositionDir = fail_dir;
    else
        % At least one item succeeded -- ticket counts as done, even if
        % some items failed (see partial-failure manifest below).
        dispositionDir = done_dir;
    end

    manifest = struct('shm_path', {items.shm_path}, 'output_file', {items.output_file}, ...
        'status', statuses, 'error', errors);
    manifestPath = fullfile(dispositionDir, [finalName '.manifest.json']);
    try
        fid = fopen(manifestPath, 'w');
        fprintf(fid, '%s', jsonencode(manifest));
        fclose(fid);
    catch
    end

    movefile(activePath, fullfile(dispositionDir, finalName));

    if nFailed > 0 && nFailed < n
        % Partial failure inside an otherwise-successful batch: also flag
        % it under fail_dir so failures aren't silently lost inside a
        % ticket that landed in "done".
        partialPath = fullfile(fail_dir, [finalName '.partial_failures.json']);
        try
            copyfile(manifestPath, partialPath);
        catch
        end
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
