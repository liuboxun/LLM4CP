function [ phi_o, theta_o ] = rotate_sphere( phi_i, theta_i, phi_r, theta_r )
%ROTATE_SPERE Rotation of the unit sphere
%
% This function rotates the input angles without changing the angle offset.
%
% Input:
%   phi_i           Azimuth angles (rad)                                [ N x L ]
%   theta_i         Elevation angles (rad)                              [ N x L ]
%   phi_r           Azimuth rotation angles                             [ N x 1 ]
%   theta_r         Elevation rotation angles                           [ N x 1 ]
%
% Derived from input:
%   N   = number of users
%   L   = number of paths
%
% Output:
%   phi_o           Rotated Azimuth angles (rad)                      	[ N x L ]
%   theta_o         Rotated Elevation angles (rad)                    	[ N x L ]
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

N = size( phi_i,1 );
L = size( phi_i,2 );

if ~exist( 'theta_i','var' ) || isempty( theta_i ) || size(theta_i,1) ~= N ||  size(theta_i,2) ~= L
    error('QuaDRiGa:qf:rotate_sphere','Size of "theta_i" does not match size of "phi_i".');
end
if ~exist( 'phi_r','var' ) || isempty( phi_r ) || size(phi_r,1) ~= N ||  size(phi_r,2) ~= 1
    error('QuaDRiGa:qf:rotate_sphere','Size of "phi_r" does not match size of "phi_i".');
end
if ~exist( 'theta_r','var' ) || isempty( theta_r ) || size(theta_r,1) ~= N ||  size(theta_r,2) ~= 1
    error('QuaDRiGa:qf:rotate_sphere','Size of "theta_r" does not match size of "phi_i".');
end

% Wrap angles around the unit circle
phi_i  = mod( real(phi_i) + pi, 2*pi) - pi;

% Transform departure angles to Cartesian coordinates
phi_i = permute( phi_i, [3,2,1] );
theta_i = permute( theta_i, [3,2,1] );

C = zeros( 3,L,N );
C(1,:,:) = cos( phi_i ) .* cos( theta_i );
C(2,:,:) = sin( phi_i ) .* cos( theta_i );
C(3,:,:) = sin( theta_i );

% Build rotation matrix
R = zeros( 3,3,N );
ce = cos( theta_r );
se = sin( theta_r );
ca = cos( phi_r );
sa = sin( phi_r );
R(1,1,:) = ce.*ca;
R(1,2,:) = -sa;
R(1,3,:) = -se.*ca;
R(2,1,:) = ce.*sa;
R(2,2,:) = ca;
R(2,3,:) = -se.*sa;
R(3,1,:) = se;
R(3,2,:) = 0;
R(3,3,:) = ce;

% Apply rotation in Cartesian coordinates
for n = 1 : N
    C(:,:,n) = R(:,:,n) * C(:,:,n);
end

% Calculate rotated departure angles
hypotxy = hypot( C(1,:,:),C(2,:,:) );
theta_o = atan2(C(3,:,:),hypotxy);
phi_o = atan2(C(2,:,:),C(1,:,:));
theta_o = permute( theta_o, [3,2,1] );
phi_o = permute( phi_o, [3,2,1] );

end
