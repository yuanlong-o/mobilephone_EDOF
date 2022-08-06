function [dataset, first_scale_factor] = txt2mat_crop(input_dir, target_pixel_spacing)
% TXT2MAT - Convert txt to matrix or other format
% A simple function for converting .txt to .mat from zemax

% input:
%   input_dir: a string of input directory
%   target_pixel_spacing: actual sensor pixel pitch value

% output:
%   dataset: h * w * N matrix, N is field number
%   first_scale_factor: first downsampling factors captured from DELTA

% Written by: Jiachen, 06/10/2021
% Modified by: Jiachen, 10/14/2021, add parameter target pixel pitch
% Modified by: Jiachen, 10/24/2021, add PSF_field crop

txt_list = dir([input_dir, '*.txt']);
delimiterIn = ' ';
headerlinesIn =22;

field_list = [8,9,10,11,12,15,16,17,18,19,23,24,25,26,27,31,32,33,34,35,38,39,40,41,42];
% ii for serial number identifier
for ii = 1 : 25
    filename = sprintf('%s\\%s', txt_list(field_list(ii)).folder, txt_list(field_list(ii)).name);
    dataset_cell = importdata(filename, delimiterIn, headerlinesIn);
    dataset(:, :, ii) = dataset_cell.data;
    %             save(sprintf('%s\\%s.mat', output_matrixdir, txt_list(ii).name(1:end-4)), 'dataset');
    %             saveastiff(im2uint16(dataset/ max(dataset(:))), sprintf('%s\\%s.tiff', output_tiffdir, txt_list(ii).name(1:end-4)));
end
spacing_tmp = dataset_cell.textdata{6};
first_scale_factor = cell2mat(textscan(spacing_tmp, '%*s %*s %*s %f %*s', 'Delimiter', ' '))/target_pixel_spacing;

end
