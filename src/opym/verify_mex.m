function verify_mex()
    critical = {'readtiff', 'writetiff', 'readzarr', 'writezarr', ...
                'createZarrFile', ...
                'parallelReadTiff', 'parallelWriteTiff', ...
                'parallelReadZarr', 'parallelWriteZarr', ...
                'deskewRotateFrame3D', ...
                'skewed_space_interp_defined_stepsize_mex', ...
                'volume_deskew_rotate_warp_mex'};
    
    all_ok = true;
    for i = 1:numel(critical)
        w = which(critical{i});
        if contains(w, '.mexa64')
            fprintf('✅ %s -> C++ MEX\n', critical{i});
        elseif isempty(w)
            fprintf('❌ %s -> NOT FOUND\n', critical{i});
            all_ok = false;
        else
            fprintf('⚠️  %s -> MATLAB fallback (%s)\n', critical{i}, w);
            all_ok = false;
        end
    end
    
    if all_ok
        fprintf('\n✅ All critical MEX accelerators are active.\n');
    else
        fprintf('\n⚠️  Some MEX accelerators are missing. Performance may be degraded.\n');
    end
end
