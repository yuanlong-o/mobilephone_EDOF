clear all; close all; clc;
addpath('.\\Utils');
gpuDevice(1);

%% PART 2
% This script is test to read RGB image and generate corresponding dataset
% Written by: Jiachen, 07/21/2021

%% directory constant
% label images dir_depth fusion
input_label_dir = '.\\dataset\\TIFF_AfterReg';
input_fusion_label_dir = '.\\dataset\\TIFF_AfterFusion';
% output_dir
output_blur_dir = '.\\dataset\\TIFF_AfterConv';

lambda_list = [590, 532, 488];
depth_list = 1 : 11;

% pre-allocated memory for nmf_psf and coef_map
input_nmf_dir = sprintf('.\\dataset\\nmf_dir\\Tifffile_config2\\Lambda_%3d\\Depth_0%d\\', 488, 1);
psf_size = importdata([input_nmf_dir, 'test_eigen_psf.mat']);
map_size = importdata([input_nmf_dir, 'test_coef_map.mat']);
nmf_psf = gpuArray.zeros(size(psf_size, 1), size(psf_size, 2), size(psf_size, 3), length(lambda_list), length(depth_list));

%% directory variable
for i = 1 : length(lambda_list)
    lambda = lambda_list(i)
    for j = 1 : length(depth_list)
        depth = depth_list(j)
        if (depth < 10)
            input_nmf_dir = sprintf('.\\dataset\\nmf_dir\\Tifffile_config2\\Lambda_%3d\\Depth_0%d\\', lambda, depth);   
        else
            input_nmf_dir = sprintf('.\\dataset\\nmf_dir\\Tifffile_config2\\Lambda_%3d\\Depth_%d\\', lambda, depth);
        end
        
        % load eigen_psf
        nmf_psf(:, :, :, i, j) = importdata([input_nmf_dir, 'test_eigen_psf.mat']);
    end
end
% load eigen_psf
nmf_map = gpuArray(map_size);

% compensate for pre-calibration
all_ones = gpuArray.ones(1944, 1944, length(lambda_list), length(depth_list));
[compensate_result] = svconv_forward(all_ones, nmf_map, nmf_psf);

% Convert label images to the nature order
[label_dir] = natsort_label_dir(input_label_dir);
[fusion_label_dir] = natsort_label_dir(input_fusion_label_dir);


tic;
for k = 1 : length(label_dir)-2
    fprintf('start generating blur_image_%d / %d in total \n', k, length(label_dir));
    disp("====================");
    label_list = dir(sprintf('%s\\%s', label_dir{k, 2}, label_dir{k, 1}));
    for kk = 3 : length(label_list)
        label_temp = imread(sprintf('%s\\%s', label_list(kk).folder, label_list(kk).name));
        label_temp = double(label_temp);
        label(:, :, :, kk - 2) = imresize(label_temp, [1944, 1944]);
    end
    
    % svconv_forward propagation
    [conv_result] = svconv_forward(label, nmf_map, nmf_psf);
    conv_result = conv_result ./ compensate_result;
    
    conv_result = gather(conv_result);
    
    % imhistmatch
    fusion_label = loadtiff(sprintf('%s\\%s', fusion_label_dir{k+2, 2}, fusion_label_dir{k+2, 1})); % k / k+2
    fusion_label = double(fusion_label);
    fusion_label = imresize(fusion_label, [1944, 1944]);
    fusion_label = (fusion_label - min(fusion_label(:))) / (max(fusion_label(:)) - min(fusion_label(:)));
    fusion_label = im2uint16(fusion_label);
    
    conv_result = double(conv_result);
    conv_result = (conv_result - min(conv_result(:))) / (max(conv_result(:)) - min(conv_result(:)));
    conv_result = im2uint16(conv_result);
%     nbins = length(unique(fusion_label(:)));
    nbins = 255;
    conv_result = imhistmatch(conv_result, fusion_label, nbins); 
    
    saveastiff(conv_result, ...
        sprintf('%s\\img_%d.tiff', output_blur_dir, k));
    toc;
end
