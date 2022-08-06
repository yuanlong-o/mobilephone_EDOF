function [psf_crop_sum_normalized] = crop_psf(dataset, N, first_scale_factor, second_scale_factor, crop_box_h, crop_box_w)
% CROP_PSF - crop psf to fit the support box

% input:
%   dataset   :   h*w*N psf matrix to be downsampled and cropped
%   N   :   field number of psf stack
%   first_scale_factor   :   first downsample
%   second_scale_factor   :   second downsample
%   crop_box_h   :   the height of support box
%   crop_box_w   :   the width of support box

% output:
%   psf_crop_sum_normalized   :   crop_box_h*crop_box_w*N matrix, cropped psf

% Written by:
% Jiachen, 06/10/2021

psf_temp = dataset(:, :, 1);
size_h = size(psf_temp, 1);
size_w = size(psf_temp, 2);

for kk = 1 : N
    %             psf_temp = double(load(sprintf('%s\\%s', psf_list(kk).folder, psf_list(kk).name)));
    psf_temp = dataset(:, :, kk);
    % first downsampling step for DELTA with sensor pixelsize_2.45um
    psf_temp = imresize(psf_temp, first_scale_factor);
    % second downsampling step for sensor image 2*2 binning(1k * 1k raw label data),
    % or for those color datasets which have no obvious features
    psf_temp = imresize(psf_temp, second_scale_factor);
    size_h = size(psf_temp, 1);
    size_w = size(psf_temp, 2);
    psf_crop(:, :, kk) = psf_temp(floor(size_h/2) - crop_box_h/2 + 1 : floor(size_h/2) + crop_box_h/2, ...
        floor(size_w/2) - crop_box_w/2 + 1 : floor(size_w/2) + crop_box_w/2);
%     psf_crop_sum_normalized(:, :, kk) = psf_crop(:, :, kk);
    psf_crop_sum_normalized(:, :, kk) = psf_crop(:, :, kk) / sum(sum(psf_crop(:, :, kk)));
end
%         psf_crop_sum_normalized = psf_crop_sum_normalized * sum(sum(psf_crop(:, :, 1)));
% attention that the energy of the R channel is very high,
% simply * sum(sum(psf_crop(:, :, 1))) will bring errors, because this varible is changeable with time, you should fix it.

end