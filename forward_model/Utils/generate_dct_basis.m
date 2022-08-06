function [C, kept_N] = generate_dct_basis(size_h, size_w, k_h, k_w)
    x_axis = 1 : size_h;
    x_axis = x_axis / size_h;
    y_axis = 1 : size_w;
    y_axis = y_axis / size_w;
    [Y, X] = meshgrid(y_axis, x_axis);
    phi_shift = 2;
    img = [];
    img_global = [];
    for i = 1 : k_h
        for j = 1 : k_w
            for p = 1 : phi_shift
                f_h = i - 1;
                f_w = j - 1;
                buf = 1 + cos(f_h * X + f_w * Y + (-1)^p * pi / 2);
                img(:, :,i, j, p) =  buf;
                %                     buf2 = imresize(buf, [sqrt_N, sqrt_N], 'nearest');
                %                     img_global(:, :,i, j, p) = buf2;
            end
        end
    end
    C_temp = reshape(img, size(img, 1)*size(img, 2), []);
    kept_N = size(C_temp, 2);
    C = C_temp.';
end

% clc, clear
% close all
% 
% %% this file is sued to genrate DCT-like base
% %  last update: 6/5/2021. YZ
% 
% 
% size_h = 4;
% size_w = 4;
% 
% k_h = 4;
% k_w = 4;
% 
% size_h_global = 200;
% size_w_global = 200;
% 
% x_axis = 1 : size_h; 
% x_axis = x_axis / size_h;
% y_axis = 1 : size_w;
% y_axis = y_axis / size_w;
% [Y, X] = meshgrid(y_axis, x_axis);
% 
% phi_shift = 2;
% 
% img = [];
% img_global = [];
% for i = 1 : k_h
%    for j = 1 : k_w
%        for p = 1 : phi_shift
%             f_h = i - 1;
%             f_w = j - 1;
%             buf = 1 + cos(f_h * X + f_w * Y + (-1)^p * pi / 2);
%             img(:, :,i, j, p) =  buf;
%             buf2 = imresize(buf, [size_h_global, size_w_global]);
% 
%             img_global(:, :,i, j, p) = buf2;
%             pause(0.1)
%             figure(101), imagesc(buf2), axis equal, axis off
%            
%        end 
%    end
% end
% 
% %% summary
% global_img = sum(img_global, [3, 4, 5]);
% figure, imshow(global_img, [])