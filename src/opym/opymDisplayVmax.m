function vmax = opymDisplayVmax(peak)
% The square-root view curve's ceiling for a channel whose first volume
% peaks at `peak` counts: 2^k - 1 >= 2 * peak, within [4095, 65535] -- as
% opym.live_view.DisplayEncoding.sqrt_for picks it (see run_live_zarr.m's
% displayLut). The deconvolved data can pass 12 bits.
top = 2 * double(peak) + 1;
vmax = min(65535, max(4095, 2 ^ ceil(log2(max(top, 1))) - 1));
end
