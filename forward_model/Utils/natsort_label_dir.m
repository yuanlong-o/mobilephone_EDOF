function [label_dir] = natsort_label_dir(input_label_dir)
% load_label_natsort - arrange image files of image directory based on natural order

% input:
%   input_dir   :   a string of label directory

% output:
%   label_dir   :   natsorted label dir

% Written by:
% Jiachen, 06/10/2021

% load label images for ground truth
label_dir = dir(input_label_dir);
label_dir = struct2cell(label_dir).';
label_dir = [label_dir(:,1:3), cellfun(@num2str, label_dir(:, 4:end), 'UniformOutput', false)];
label_dir = natsortrows(label_dir);

end