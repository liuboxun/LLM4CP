function map = get_lsp_map( h_builder, xc, yc, zc )
%GET_LSP_MAP Calculates the spatial map of the correlated LSPs
%
% Calling object:
%   Single object
%
%
% Input:
%   xc
%   A vector containing the map sample positions in [m] in x-direction
%
%   yc
%   A vector containing the map sample positions in [m] in y-direction
%
%   zc
%   A vector containing the map sample positions in [m] in z-direction
%
% Output:
%   map
%   An array of size [ nx, ny, nz, 8 ] containing the values of the LSPs at the sample positions.
%   The indices of the fourth dimension are:
%      * Delay spread [s]
%      * K-factor [linear]
%      * Shadow fading [linear]
%      * Azimuth angle spread of departure [rad]
%      * Azimuth angle spread of arrival [rad]
%      * Elevation angle spread of departure [rad]
%      * Elevation angle spread of arrival [rad]
%      * Cross-polarization ratio [linear]
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

if numel( h_builder ) > 1
    error('QuaDRiGa:qd_builder:ObjectArray','??? "get_lsp_map" is only defined for scalar objects.')
else
    h_builder = h_builder(1,1); % workaround for octave
end

if h_builder.dual_mobility == -1
    h_builder.check_dual_mobility;
end
if h_builder.dual_mobility ~= 0
    error('QuaDRiGa:qd_builder:get_lsp_map','Parameter-maps are meaningless for dual-mobility setups.')
end

if numel( h_builder.simpar(1,1).center_frequency ) > 1
    error('QuaDRiGa:qd_builder:get_lsp_map','Parameter-maps cannot be generated for multi-frequency simulations.')
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

[ ds, kf, sf, asD, asA, esD, esA, xpr ] = gen_lsf_parameters( h_builder, 3, 0, h_builder.tx_position, [x,y,z].' );

map = [ ds; kf; sf; asD; asA; esD; esA; xpr ].';
map = reshape( map, ny, nx, nz, [] );

end
