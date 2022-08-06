function [Y,ndx,dbg] = natsortrows(X,rgx,varargin)
% Natural-order / alphanumeric sort the rows of a cell array or string array.
%
% (c) 2014-2021 Stephen Cobeldick
%
% Sort text by character code and by number value.
% SORTROWS <col> input is supported, to select columns to sort by.
%
%%% Example:
% >> X = {'x2','10';'x10','0';'x1','0';'x2','2'};
% >> sortrows(X) % Wrong numeric order:
% ans =
%     'x1'     '0'
%     'x10'    '0'
%     'x2'     '10'
%     'x2'     '2'
% >> natsortrows(X)
% ans =
%     'x1'     '0'
%     'x2'     '2'
%     'x2'     '10'
%     'x10'    '0'
%
%%% Syntax:
%  Y = natsortrows(X)
%  Y = natsortrows(X,rgx)
%  Y = natsortrows(X,rgx,<options>)
% [Y,ndx,dbg] = natsortrows(X,...)
%
% To sort the elements of a string/cell array use NATSORT (File Exchange 34464)
% To sort any file-names or folder-names use NATSORTFILES (File Exchange 47434)
%
%% File Dependency %%
%
% NATSORTROWS requires the function NATSORT (File Exchange 34464). The optional
% arguments <options> are passed directly to NATSORT (except for the SORTROWS-
% style <col> numeric vector, which is parsed internally). See NATSORT for case-
% sensitivity, sort direction, number substring matching, and other options.
%
%% Examples %%
%
% >> A = {'B','2','X';'A','100','X';'B','10','X';'A','2','Y';'A','20','X'};
% >> sortrows(A) % wrong number order:
% ans =
%    'A'  '100'  'X'
%    'A'    '2'  'Y'
%    'A'   '20'  'X'
%    'B'   '10'  'X'
%    'B'    '2'  'X'
% >> natsortrows(A)
% ans =
%    'A'    '2'  'Y'
%    'A'   '20'  'X'
%    'A'  '100'  'X'
%    'B'    '2'  'X'
%    'B'   '10'  'X'
% >> natsortrows(A,[],'descend')
% ans =
%     'B'    '10'     'X'
%     'B'    '2'      'X'
%     'A'    '100'    'X'
%     'A'    '20'     'X'
%     'A'    '2'      'Y'
%
% >> sortrows(A,[2,-3]) % Wrong number order:
% ans =
%    'B'   '10'  'X'
%    'A'  '100'  'X'
%    'A'    '2'  'Y'
%    'B'    '2'  'X'
%    'A'   '20'  'X'
% >> natsortrows(A,[],[2,-3])
% ans =
%    'A'    '2'  'Y'
%    'B'    '2'  'X'
%    'B'   '10'  'X'
%    'A'   '20'  'X'
%    'A'  '100'  'X'
%
% >> B = {'ABCD';'3e45';'67.8';'+Inf';'-12';'+9';'NaN'};
% >> sortrows(B) % wrong number order:
% ans =
%    '+9'
%    '+Inf'
%    '-12'
%    '3e45'
%    '67.8'
%    'ABCD'
%    'NaN'
% >> natsortrows(B,'[-+]?(NaN|Inf|\d+\.?\d*(E[-+]?\d+)?)')
% ans =
%    '-12'
%    '+9'
%    '67.8'
%    '3e45'
%    '+Inf'
%    'NaN'
%    'ABCD'
%
%% Input and Output Arguments %%
%
%%% Inputs (**=default):
% X   = Array of size MxN, with atomic rows to be sorted. Can be a
%       string array, or a cell array of char vectors, or a categorical
%       array, or any other array type supported by NATSORT.
% rgx = Regular expression to match number substrings, '\d+'**
%     = [] uses the default regular expression.
% <options> can be supplied in any order:
%     = SORTROWS <col> argument is supported: a numeric vector where
%       each integer specifies which column of X to sort by, and
%       negative integers indicate that the sort order is descending.
%     = all remaining options are passed directly to NATSORT.
%
%%% Outputs:
% Y   = Array X with rows sorted into alphanumeric order.
% ndx = NumericVector, size Mx1. Row indices such that Y = X(ndx,:).
% dbg = CellVectorOfCellArrays, size 1xN. Each cell contains the debug cell array
%       for one column of input X. Helps debug the regular expression (see NATSORT).
%
% See also SORT SORTROWS NATSORT NATSORTFILES CELLSTR REGEXP IREGEXP SSCANF
%% Input Wrangling %%
%
assert(ndims(X)<3,...
	'SC:natsortrows:X:NotMatrix',...
	'First input <X> must be a matrix (2D).') %#ok<ISMAT>
%
if nargin>1
	varargin = [{rgx},varargin];
end
%
%% Columns to Sort %%
%
[nmr,nmc] = size(X);
ndx = 1:nmr;
drn = {'descend','ascend'};
dbg = {};
isn = cellfun(@isnumeric,varargin);
isn(1) = false; % rgx
%
if any(isn)
	assert(nnz(isn)<2,...
		'SC:natsortrows:col:Overspecified',...
		'The <col> input is over-specified (only one numeric input is allowed).')
	col = varargin{isn};
	assert(isvector(col)&&isreal(col)&&all(~mod(col,1))&&all(col)&&all(abs(col)<=nmc),...
		'SC:natsortrows:col:IndexMismatch',...
		'The <col> input must be a vector of column indices into the first input <X>.')
	sgn = (3+sign(col))/2;
	idc = abs(col);
else
	idc = 1:nmc;
end
%
%% Sort Columns %%
%
for k = numel(idc):-1:1
	if any(isn)
		varargin{isn} = drn{sgn(k)};
	end
	if nargout<3 % faster:
		[~,ids] = natsort(X(ndx,idc(k)),varargin{:});
	else % for debugging:
		[~,ids,tmp] = natsort(X(ndx,idc(k)),varargin{:});
		[~,idd] = sort(ndx);
		dbg{idc(k)} = tmp(idd,:);
	end
	ndx = ndx(ids);
end
%
ndx = ndx(:);
Y = X(ndx,:);
%
end
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%natsortrows