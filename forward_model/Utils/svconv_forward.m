function [conv_result] = svconv_forward(label, nmf_map, nmf_psf)
% svconv_forward - svconv_forward propagation

% Written by:
% Jiachen, 06/10/2021

[hh, ww, channel, depth] = size(label);
label = gpuArray(label);

% svconv forward projection
conv_result = gpuArray.zeros(size(label));
conv_result_buf = gpuArray.zeros(hh, ww, channel, depth);
for n = 1 : channel
    for z = 1 : depth
        fprintf('start convolving label with depth_%d / %d \n', z, depth);
        disp("====================");
        for m = 1 : size(nmf_map, 3)
            buf = conv2fft(label(:, :, n, z).* squeeze(nmf_map(:, :, m)), squeeze(nmf_psf(:, :, m, n, z)), 'same');
            conv_result_buf(:, :, n, z) = conv_result_buf(:, :, n, z) + buf;
        end
    end
end
conv_result = sum(conv_result_buf, 4) / depth;
% keep positive
conv_result(conv_result < 0) = 0;

end