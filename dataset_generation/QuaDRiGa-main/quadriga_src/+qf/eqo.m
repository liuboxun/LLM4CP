function iseq = eqo( obj, obj_array )
%EQO Determines if object handles are equal
%
% Description:
%   Octave 4.0 does not implement the "eq" function and is very sensitive to incorrect indexing of
%   object arrays. This function provides the required functionality for QuaDRiGa. For this to
%   work, the corresponding classes must have a writable property "OctEq" which may be hidden.
%
% Input:
%   obj
%   A single object handle.
%
%   obj_array
%   A (multi-dimensional) array of object handles with up to four dimensions.
%
% Output:
%   iseq
%   A boolean variable having the same size as "obj_array". True (1) values indicate that the
%   corresponding handles in "obj_array"  point to the same object as the "obj"-handle does.
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

if numel( obj ) > 1
    error('obj must be scalar.');
else
    obj = obj(1,1);
end

sic = size( obj_array );
N   = prod( sic );

% Set all properties "OctEq" in obj_array to false
for n = 1 : N
    [ i1,i2,i3,i4 ] = qf.qind2sub( sic, n );
    if numel(sic) == 4
        obj_array( i1,i2,i3,i4 ).OctEq = false;
    elseif numel(sic) == 3
        obj_array( i1,i2,i3 ).OctEq = false;
    else
        obj_array( i1,i2 ).OctEq = false;
    end
end

% Set property "OctEq" in obj to true
obj.OctEq = true;

% Read all properties "OctEq" in obj_array
iseq = false( sic );
for n = 1 : N
    [ i1,i2,i3,i4 ] = qf.qind2sub( sic, n );
    iseq( i1,i2,i3,i4 ) = obj_array( i1,i2,i3,i4 ).OctEq;
end

end
