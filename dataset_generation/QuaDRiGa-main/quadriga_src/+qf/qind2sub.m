function [ i1,i2,i3,i4 ] = qind2sub( sic, ndx )
%QIND2SUB Calculates subscripts from linear indices.
%
% Description:
%   The qind2sub command determines the equivalent subscript values corresponding to a single index
%   into an array. The output is identical to the MATLAB / Octave ind2sub command. However, the
%   qind2sub command offers slightly better performance.
%
% Input:
%   sic
%   A vector describing the layout of the object. For two-dimensional objects, siz(1) is the
%   number of rows and siz(2) is the number of columns.
%
%   ndx
%   A vector containing the linear indices of the array.
%
% Output:
%   i1
%   Row-index
%
%   i2
%   Column-index
%
%   i3
%   Page-index
%
%   i4
%   Index of the 4-th dimension
%
%
% QuaDRiGa Copyright (C) 2011-2019
% Fraunhofer-Gesellschaft zur Foerderung der angewandten Forschung e.V. acting on behalf of its
% Fraunhofer Heinrich Hertz Institute, Einsteinufer 37, 10587 Berlin, Germany
% All rights reserved.
%
% e-mail: quadriga@hhi.fraunhofer.de
%
% This file is part of QuaDRiGa.
%
% The Quadriga software is provided by Fraunhofer on behalf of the copyright holders and
% contributors "AS IS" and WITHOUT ANY EXPRESS OR IMPLIED WARRANTIES, including but not limited to
% the implied warranties of merchantability and fitness for a particular purpose.
%
% You can redistribute it and/or modify QuaDRiGa under the terms of the Software License for
% The QuaDRiGa Channel Model. You should have received a copy of the Software License for The
% QuaDRiGa Channel Model along with QuaDRiGa. If not, see <http://quadriga-channel-model.de/>.

sic = [ sic, ones(1,4-numel(sic)) ];
sik = [ 1, cumprod( sic(1:3) ) ];

vi = rem(ndx-1, sik(4)) + 1;
i4 = (ndx - vi)/sik(4) + 1;
ndx = vi;
vi = rem(ndx-1, sik(3)) + 1;
i3 = (ndx - vi)/sik(3) + 1;
ndx = vi;
vi = rem(ndx-1, sik(2)) + 1;
i2 = (ndx - vi)/sik(2) + 1;
ndx = vi;
vi = rem(ndx-1, sik(1)) + 1;
i1 = (ndx - vi)/sik(1) + 1;

end

