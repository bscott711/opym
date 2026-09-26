% FFT_SIZES  How much of the live decon is the z size's FFT cost?
%
% The live decon runs 3D FFTs on (ny, nx, nz) = (490, 1458, 161) volumes;
% 161 = 7 x 23, and 23 is a slow radix for cuFFT. This times single-
% precision gpuArray fftn + ifftn at z = 161 and at the nearby FFT-friendly
% sizes 162 (2 x 3^4) and 168 (2^3 x 3 x 7), as the decon's OTF
% multiplication uses them. Measurement only: nothing here touches the
% live path or its output.
%
% Run on a test GPU, e.g.:
%   CUDA_VISIBLE_DEVICES=1 matlab -batch "run('scripts/live_bench/fft_sizes.m')"
% Prints one JSON line per size.

g = gpuDevice(1);
sizes = [161, 162, 168];
ny = 490; nx = 1458; reps = 10;
for nz = sizes
    x = gpuArray.rand(ny, nx, nz, 'single');
    f = fftn(x); y = real(ifftn(f)); wait(g);    % plan + warm
    t = zeros(1, reps);
    for k = 1:reps
        t0 = tic;
        f = fftn(x);
        y = real(ifftn(f .* f));
        wait(g);
        t(k) = toc(t0);
    end
    fprintf('{"nz": %d, "fft_ifft_pair_s_median": %.4f, "min": %.4f}\n', ...
        nz, median(t), min(t));
    clear x f y
end
