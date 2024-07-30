function out = reshapeo( obj, shape )
%RESHAPEO Reshapes the input handle object array to an output object array
%
% Description:
%   Octave 4.0 does not implement the "reshape" function and is very sensitive to incorrect
%   indexing of object arrays. This function provides the required functionality for QuaDRiGa.
%
% Input:
%   obj
%   Object array with up to 4 dimensions.
%
%   shape
%   A vector describing the desired layout of the object array. For two-dimensional array, shape(1)
%   is the number of rows and shape(2) is the number of columns.
%
% Output:
%   out
%   Reshaped object array with up to 4 dimensions.
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

if numel( obj ) ~= prod( shape )
   error('Size does not match');
end

sic = size( obj );
out = obj(1,1,1,1);
for n = 2 : numel( obj )
    [ i1,i2,i3,i4 ] = qf.qind2sub( sic, n );
    [ j1,j2,j3,j4 ] = qf.qind2sub( shape, n );
    out( j1,j2,j3,j4 ) = obj( i1,i2,i3,i4 );
end

end
