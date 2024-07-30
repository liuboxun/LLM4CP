function S = map( h_sos, xc, yc, zc )
%MAP Generates a map at the given coordinates in x,y and z direction
%
% This method generates a multi-dimensional array (of up to 3 dimensions) of spatially correlated
% random variables.  
%
% Input:
%   xc      Vector of x coordinates in [m]
%   yc      Vector of y coordinates in [m]
%   zc      Vector of z coordinates in [m]
%
% Output:
%   S       Array of spatially correlated random variables
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

if numel( h_sos ) > 1 
   error('QuaDRiGa:qd_sos:map','map not definded for object arrays.');
else
    h_sos = h_sos(1,1); % workaround for octave
end


ndim_out = nargin-1;
if ndim_out > h_sos.dimensions
    error( 'Number of requested dimensions is not supported.' );
end

if ~exist( 'yc','var' )
    yc = 0;
end

if ~exist( 'zc','var' )
    zc = 0;
end

nx = numel(xc);
ny = numel(yc);
nz = numel(zc);

ox =  ones(nx,1,'uint8');
oy =  ones(ny,1,'uint8');
oz =  ones(nz,1,'uint8');

x = reshape( single(xc) , 1, [] );
x = x( oy,:,oz );
x = x(:);

y = reshape( single(yc) , [] , 1 );
y = y( :,ox,oz );
y = y(:);

z = reshape( single(zc) , 1 , 1, []  );
z = z( oy,ox,: );
z = z(:);

switch ndim_out
    case 1
        S = h_sos.val( x.' );
    case 2
        S = h_sos.val( [x,y].' );
    case 3
        S = h_sos.val( [x,y,z].' );
end

S = reshape( S, ny, nx, nz );

end
