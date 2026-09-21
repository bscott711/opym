function rmdirs(varargin)
% Shim for a typo in PetaKit5D's XR_RLdeconFrame3D.m (line ~244), the only
% call site of 'rmdirs' anywhere in the vendored install
% (/mmfs2/cm/shared/apps_local/petakit5d): no such function is defined
% there. MATLAB's real builtin is 'rmdir' (singular). Without this, every
% decon RE-run of a dataset that already has an eroded mask directory on
% disk from a prior run (i.e. exactly the case a parameter retune hits)
% throws 'Undefined function rmdirs' and the ticket fails outright -- a
% first-time run never hits it, since the mask directory doesn't exist yet
% and the guarding `if exist(maskFullPath, 'dir')` short-circuits.
rmdir(varargin{:});
end
