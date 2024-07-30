function [ theta, phi, B, d_phi ] = pack_sphere( N )
%PACK_SPHERE Creates equally distributed points on the unit sphere
%
% Description:
%   This function equally distributes points on a unit sphere. The actual number of placed points
%   (M) might be smaller then N due to rounding offsets.
%
% Input:
%   N
%   The number of points to place.
%
% Output:
%   theta
%   Vector of elevation angles in [rad] having values from -pi/2 to pi/2; size [ M x 1 ]
%
%   phi
%   Vector of azimuth angles in [rad] having values from -pi to pi; size [ M x 1 ]
%
%   B
%   Positions of the points on Cartesian coordinates; size [ 3 x M ]
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

a = 4*pi/N;                 % Sphere surface
d = sqrt(a);                % Distance between points

M_theta = round( pi/d );    % No. Latitudes
d_theta = pi / M_theta;     % Resolution
d_phi   = a / d_theta;

A = zeros(N+50,2);             % The output coordinates

N_count = 1;                % Number of placed points
for m = 1 : M_theta
    theta = pi * ( m-0.5 ) / M_theta;           % Current latitude
    M_phi = round( 2*pi*sin(theta)/d_phi );     % No. points on current latitude
    phi   = 2*pi*(0:M_phi-1) / M_phi;           % Longitude points
    
    N_new = numel( phi );
    A( N_count : N_count + N_new - 1 , 1 ) = theta;
    A( N_count : N_count + N_new - 1 , 2 ) = phi;
    
    N_count = N_count + N_new;
end
N_count = N_count - 1;

% Format output
theta = A(1:N_count,1)-pi/2;
phi   = angle( exp( 1j.*A(1:N_count,2) ) );

if nargout > 2
    B = zeros( 3,N_count );
    B(1,:) = cos(theta).*cos(phi);
    B(2,:) = cos(theta).*sin(phi);
    B(3,:) = sin(theta);
end

end
