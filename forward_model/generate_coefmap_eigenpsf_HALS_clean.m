clear all; close all; clc;
addpath('.\\Utils');

%% PART 1
% ---- This file contains the following sections:
% 1 ----- output cropped psf stack from .txt file generated from zemax
% 1.1 ------ import .txt data and tranform to .mat (,tiff for show)
% 1.2 ------ first downsample to align DELTA(zemax) with the sensor pixelsize_2.45um
% 1.3 ------ second downsample for sensor image 2*2 binning(eg. 1k*1k raw label data), set 1 for 2k*2k data
% 1.4 ------ crop the psf to the proper support box
% 2 ----- generate coeffcient map and eigen psf from cropped psf stack
% 2.1 ------ generate 2D dct basis as coefficient map
% 2.2 ------ update eigen psf by HALS algorithm
% 2.3 ------ finish NMF step and save all results

% Written by: Jiachen, 06/21/2021
% Modified by: Jiachen, 10/14/2021, add parameter target pixel pitch
% Modified by: Jiachen, 10/24/2021, consider difference between actual PSF(crop included, set global_size as 1944*1944) and Zemax PSF 

%% Some file and directory constants
lambda_list = [488, 532, 590];
depth_list = 1 : 11;
% data spacing list from zemax
first_scale_factor_list = zeros(length(lambda_list), length(depth_list));

% generate_dct_basis, coefficient map
ind = 0;
size_h_global = 1944; %%
size_w_global = 1944; %%
k_h = 3; %%
k_w = 3; %%
[global_H, kept_N] = generate_dct_basis(size_h_global, size_w_global, k_h, k_w);

%%
for i = 1 : length(lambda_list)
    lambda = lambda_list(i)
    for j = 1 : length(depth_list)
        depth = depth_list(j)
        ind = ind + 1;
        if (depth < 10)
            input_dir = sprintf('.\\dataset\\txtfile\\Lambda_%d\\Depth_0%d\\', lambda, depth);
            output_psfdir = sprintf('.\\dataset\\Tifffile\\Lambda_%3d\\Depth_0%d', lambda, depth);
            output_eigen_dir = output_psfdir;
            output_map_dir = output_psfdir;
            mkdir(output_psfdir);
        else
            input_dir = sprintf('.\\dataset\\txtfile\\Lambda_%d\\Depth_%d\\', lambda, depth);
            output_psfdir = sprintf('.\\dataset\\Tifffile\\Lambda_%3d\\Depth_%d', lambda, depth);
            output_eigen_dir = output_psfdir;
            output_map_dir = output_psfdir;
            mkdir(output_psfdir);
        end
        
        tic;
        disp("start generating mat from txt file");
        disp("====================");
        % convert txt to matrix or other format
        target_pixel_spacing = 1.12; % add parameter target pixel pitch
        [dataset, first_scale_factor] = txt2mat_crop(input_dir, target_pixel_spacing);
        
        % load PSF .mat data and generate psf_crop_stack
        N = 25; %%
        sqrt_N = sqrt(N);
        crop_box_h = 90; %% default: 30(second_scale_factor = 0.5)
        crop_box_w = 90; %% default: 30
%         first_scale_factor = first_scale_factor_list(i, j); % default: 0.1053 withoutmask: 0.7918
        second_scale_factor = 1; % default: 0.5 withoutmask: 0.5, 1: color_dataset
        
        % crop psf to fit the support box
        psf_crop_sum_normalized(:,:,:,i,j) = crop_psf(dataset, N, first_scale_factor, second_scale_factor, crop_box_h, crop_box_w);

        save(sprintf('%s\\psf_crop_stack.mat', output_psfdir), 'psf_crop_sum_normalized');
        toc;
        disp("finish generating mat from txt file");
        disp("====================");
    end
end
%         saveastiff(im2uint16(psf_crop / max(psf_crop(:))), ...
%             sprintf('%s\\psf_crop_stack.tiff', output_psfdir));
        
         %% Load crop_psf_stack
%         load(sprintf('%s\\psf_crop_stack.mat', output_psfdir));

for grid = 1 : size(psf_crop_sum_normalized, 3)
    for depth = 1 : size(psf_crop_sum_normalized, 4)
        psf_crop_sum_normalized(:,:,grid,depth,:) = psf_crop_sum_normalized(:,:,grid,depth,:) ...
            / sum(sum(psf_crop_sum_normalized(:,:,grid,depth,2)));
    end
end

ind = 0;

for i = 1 : length(lambda_list)
    lambda = lambda_list(i)
    for j = 1 : length(depth_list)
        depth = depth_list(j)
        ind = ind + 1;
        if (depth < 10)
            input_dir = sprintf('.\\dataset\\txtfile\\Lambda_%d\\Depth_0%d\\', lambda, depth);
            output_psfdir = sprintf('.\\dataset\\Tifffile\\Lambda_%3d\\Depth_0%d', lambda, depth);
            output_eigen_dir = output_psfdir;
            output_map_dir = output_psfdir;
            mkdir(output_psfdir);
        else
            input_dir = sprintf('.\\dataset\\txtfile\\Lambda_%d\\Depth_%d\\', lambda, depth);
            output_psfdir = sprintf('.\\dataset\\Tifffile\\Lambda_%3d\\Depth_%d', lambda, depth);
            output_eigen_dir = output_psfdir;
            output_map_dir = output_psfdir;
            mkdir(output_psfdir);
        end
        
        psf_crop_stack = psf_crop_sum_normalized(:,:,:,i,j);
        [crop_box_h, crop_box_w, N] = size(psf_crop_stack);
        sqrt_N = sqrt(N);
        d1 = crop_box_h * sqrt_N;
        d2 = crop_box_w * sqrt_N;

        %% Generate fixed coef map using 2D DCT basis functions
        %% generate nmf_coef_map
        if ind == 1
            % imresize the map to fit the size of blurred image
            global_map =reshape(global_H.', size_h_global, size_w_global, []); % note here is transposed
            % global_map = global_map - min(global_map(:));
            save(sprintf('%s\\test_coef_map.mat', output_map_dir), 'global_map');
            saveastiff(im2uint16(global_map/ max(global_map(:))), sprintf('%s\\test_coef_map.tiff', output_map_dir));
        end
        
        %% generate nmf_eigen_psf
        size_h = sqrt_N;
        size_w = sqrt_N;
        k_h = 3; %%
        k_w = 3; %%
        [chosen_H, kept_N] = generate_dct_basis(size_h, size_w, k_h, k_w);
        
        psf_trans = reshape(psf_crop_stack, [], size(psf_crop_stack, 3));
        [U, S, V] = svd(psf_trans, 'econ');
%         w0 = U(:, 1 : kept_N) * S(1 : kept_N, 1 : kept_N)  * (V(:, 1 : kept_N) .');
        w0 = U(:, 1 : kept_N) * S(1 : kept_N, 1 : kept_N);
        w0(w0 < 0) = 0;
        
%         w0 = w0(:, 1 : kept_N);
%         w0 = ones(size(psf_crop_stack, 1) * size(psf_crop_stack, 2), kept_N);
%         params = struct('maxIter', 1);
        IND = true(size(w0));
        W = HALS_spatial(psf_trans, w0, chosen_H, IND, 15); 
%         A = HALS_only_A(psf_crop_stack, A, C, params);

        chosen_PSF = reshape(W, size(psf_crop_stack, 1), size(psf_crop_stack, 2), []);
        chosen_PSF = chosen_PSF / sum(chosen_PSF(:));
        save(sprintf('%s\\test_eigen_psf.mat', output_eigen_dir), 'chosen_PSF');
        saveastiff(im2uint16(chosen_PSF / max(chosen_PSF(:))), sprintf('%s\\test_eigen_psf.tiff', output_eigen_dir));

     end
end