function [J, err_mat] = decon_lucy_omw_function(I, PSF_f, PSF_b, NUMIT, varargin)
% RL decon with alternative backprojector OTF masked wiener (OMW) method
%   save16bit: for GPU compute, convert to 16bit before transfer from GPU to CPU 
%   bbox: for GPU compute, crop the data first before transfer from GPU to CPU
%
% Author: Xiongtao Ruan (11/10/2022)
%
% xruan (08/31/2023): add scaleFactor, deconOffset, dampFactor, and
%   EdgeErosion, and use input parser for parameters.
%
% opym patch (2026-09-27), shadowing the shared PetaKit5D copy (see
% run_petakit_server.m's addpath of patches/): each axis is padded to an
% FFT-friendly size -- the smallest >= n whose prime factors are all 2, 3,
% 5 or 7 -- by repeating its last plane, deconvolved at that size, and
% cropped back before the edge-erosion mask and the 16-bit cast. cuFFT is
% slow on large prime factors: a live volume's z = 161 = 7 x 23; at 162
% (2 x 3^4) an FFT pair takes 16.6 ms instead of 18.2 ms on the RTX PRO
% 6000. Every caller (RLdecon: the live and the batch paths) gets it, so
% they stay bit-identical to each other; results differ slightly from
% the unpadded ones in earlier archives (approved 2026-09-27). Otherwise
% unchanged.


ip = inputParser;
ip.CaseSensitive = false;
ip.addRequired('I', @isnumeric);
ip.addRequired('PSF_f', @isnumeric);
ip.addRequired('PSF_b', @isnumeric);
ip.addRequired('NUMIT', @isnumeric);
ip.addParameter('Background', [], @isnumeric);
ip.addParameter('useGPU', true, @islogical); % use GPU processing
ip.addParameter('save16bit', true, @islogical);
ip.addParameter('dampFactor', 1, @isnumeric); % damp factor for decon result
ip.addParameter('scaleFactor', 1, @isnumeric); % scale factor for result
ip.addParameter('deconOffset', 0, @isnumeric); % offset for decon result
ip.addParameter('EdgeErosion', 0, @isnumeric); % edge erosion for decon result
ip.addParameter('bbox', [], @isnumeric); % bounding box to crop data after decon
ip.addParameter('debug', false, @islogical);
ip.addParameter('debug_folder', './debug/', @ischar);
ip.addParameter('saveStep', 1, @isnumeric); % save intermediate results every given iterations

ip.parse(I, PSF_f, PSF_b, NUMIT, varargin{:});

pr = ip.Results;
Background = pr.Background;
useGPU = pr.useGPU;
save16bit = pr.save16bit;
dampFactor = pr.dampFactor;
scaleFactor = pr.scaleFactor;
deconOffset = pr.deconOffset;
EdgeErosion = pr.EdgeErosion;
bbox = pr.bbox;
debug = pr.debug;
debug_folder = pr.debug_folder;
saveStep = pr.saveStep;

if isempty(Background)
    Background = 0;
end

PSF_b = single(PSF_b);
sizeI0 = size(I);
sizeI = arrayfun(@fftFriendly, sizeI0);

% 1. Prepare PSFs
persistent OTF_f OTF_b psz sq_b gpuNum last_useGPU
if isempty(psz) || isempty(OTF_f) || ~(numel(psz) == numel(sizeI) && all(psz == sizeI) && last_useGPU == useGPU ...
        && sq_b == sum(PSF_b .^ 2, 'all')) || (useGPU && ~isempty(OTF_f) && gpuNum > 0 && ~existsOnGPU(OTF_f))
    psz = sizeI;
    
    gpuNum = 0;
    last_useGPU = useGPU;
    if useGPU
        gpuNum = gpuDeviceCount("available");
    end

    PSF_f = single(PSF_f);    
    sq_b = sum(PSF_b .^ 2, 'all');
    if useGPU && gpuNum > 0
        PSF_f = gpuArray(PSF_f);
        PSF_b = gpuArray(PSF_b);
    end
    OTF_f = decon_psf2otf(PSF_f ./ sum(PSF_f(:)), double(sizeI));
    OTF_b = decon_psf2otf(PSF_b ./ sum(PSF_b(:)), double(sizeI));
end
clear PSF_f PSF_b;

if gpuNum == 0
    useGPU = false;
end

if useGPU
    I = gpuArray(I);
    if deconOffset ~= 0 || EdgeErosion > 0
        I_mask = I ~= 0;
    end
    if strcmp(underlyingType(I), 'uint16')
        I = I - uint16(Background);            
        I = single(I);
    else
        I = max(I - Background, 0);    
    end
else
    if ~isa(I, 'double')
        I = single(I);
    end
    if deconOffset ~= 0 || EdgeErosion > 0
        I_mask = I ~= 0;
    end    
    I = max(I - Background, 0);
end
% Pad to the FFT-friendly size by repeating the last plane of each axis.
if any(sizeI ~= sizeI0)
    I = I(padIndex(sizeI0(1), sizeI(1)), padIndex(sizeI0(2), sizeI(2)), ...
        padIndex(sizeI0(3), sizeI(3)));
end
J = I;

% 2. L_R Iterations
estep = saveStep;
err_mat = zeros(NUMIT, 4);

for k = 1 : NUMIT
    % valina RL update step
    CX = real(ifftn(fftn(J) .* OTF_f));
    % CX = CX + (CX < eps) * eps;
    CX = max(CX, eps);
    J = real(ifftn(fftn(I ./ CX) .* OTF_b)) .* J;
    % J = J .* (J > 0);
    J = max(J, 0);
            
    if debug
        err_mat(k, 1:2) = [k, mean((double(J) - double(I)) .^ 2, 'all')];
        err_mat(k, 3) = err_mat(k, 2) ./ err_mat(1, 2); 
        if rem(k, estep) == 0 && k > estep * 2
            istp = k/estep;
            err_mat(k, 4) = min(abs(err_mat(istp, 3) - err_mat(istp-1, 3)) / 10, abs(err_mat(istp, 3) + err_mat(istp-2, 3) - 2 * err_mat(istp-1, 3)) / 2);
        else
            if k < 3 * estep
                err_mat(k, 4) = 1;
            else
                err_mat(k, 4) = err_mat(istp * estep, 4);
            end
        end
        
        if k == 1
            mkdir(debug_folder);
        end
        if rem(k, estep) == 0
            if useGPU 
                writetiff(uint16(gather(J)), sprintf('%s/Iter_%04d.tif', debug_folder, k));
            else
                writetiff(uint16(J), sprintf('%s/Iter_%04d.tif', debug_folder, k));
            end
        end
    end
end

if debug
    err_mat = err_mat(1 : floor(NUMIT / estep), :);
end

% post-processing of deconvolved result
if dampFactor > 1
    J = J .* (J <= dampFactor * I) + min(J, dampFactor * I .* (J >= dampFactor * I));
end
clear I CX;
if any(sizeI ~= sizeI0)
    J = J(1 : sizeI0(1), 1 : sizeI0(2), 1 : sizeI0(3));
end

if EdgeErosion > 0
    I_mask = decon_mask_edge_erosion(I_mask, EdgeErosion);
end

if scaleFactor ~= 1 || deconOffset ~= 0 || EdgeErosion > 0
    if deconOffset ~= 0 || EdgeErosion > 0
        J = (J * scaleFactor + deconOffset) .* I_mask;
    else
        J = J * scaleFactor;
    end
end

if save16bit
    J = uint16(J);
end

if useGPU
    if ~isempty(bbox)
        J = J(bbox(1) : bbox(4), bbox(2) : bbox(5), bbox(3) : bbox(6));
    end    
    J = gather(J);
else
    if ~isempty(bbox)
        J = crop3d(J, bbox);
    end
end

end


function m = fftFriendly(n)
% The smallest m >= n whose prime factors are all 2, 3, 5 or 7.
m = n;
while true
    r = m;
    for f = [2, 3, 5, 7]
        while mod(r, f) == 0
            r = r / f;
        end
    end
    if r == 1
        return;
    end
    m = m + 1;
end
end


function idx = padIndex(n, m)
% Indices 1..n, then n repeated up to length m.
idx = [1 : n, repmat(n, 1, m - n)];
end

