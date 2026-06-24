function run_gpu_pipeline_async(activePath, done_dir, fail_dir, val_shm, outFn, psfFn, varargin)
    try
        [~, fname, ext] = fileparts(outFn);
        shm_outFn = fullfile('/dev/shm/opym_jobs', [fname ext '_final']);
        run_gpu_pipeline(val_shm, shm_outFn, psfFn, varargin{:});
        [~, jobfname, jobext] = fileparts(activePath);
        finalName = strrep([jobfname jobext], '.active_', '');
        movefile(activePath, fullfile(done_dir, finalName));
    catch ME
        [~, fname, ext] = fileparts(activePath);
        finalName = strrep([fname ext], '.active_', '');
        movefile(activePath, fullfile(fail_dir, finalName));
        errLog = fullfile(fail_dir, [strrep(fname, '.active_', '') '.log']);
        fid = fopen(errLog, 'w');
        fprintf(fid, '%s\n', getReport(ME));
        fclose(fid);
    end
end
