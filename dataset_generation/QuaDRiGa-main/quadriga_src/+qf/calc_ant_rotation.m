function [ R, phiL, thetaL, gamma ] = calc_ant_rotation( zrot, yrot, xrot, phi, theta )
%CALC_ANT_ROTATION Calculates the polarimetric antenna rotation in geographic coordinates
%
%   This function implements the antenna rotation, including the polarization. For each rotation
%   angle "zrot", "yrot", and "xrot", a 3D rotation matrix (Rz, Ry, Rx) is calculated (assuming a
%   right-handed Cartesian coordinate system). The matrices are combined in the order "Rz * Ry *
%   Rx", i.e. the x-rotation is applied first and the z-rotation is applied last. The input angles
%   "phi" (azimuth) and "theta" (elevation) are relative to the global coordinate system. They are
%   converted into the local antenna coordinates "phiL" and "thetaL" which takes the antenna
%   orientation into account. The angle "gamma" is the polarization rotation angle. The rotation
%   angles "zrot", "yrot", and "xrot" can be empty, scalar, or match the size of "phi and "theta".
%   This function is used by the classes "qd_arrayant", "qd_layout", and "qd_builder".
%
%   See: Jaeckel, S.; "Quasi-deterministic channel modeling and experimental validation in
%   cooperative and massive MIMO deployment topologies"; PHD thesis; TU Ilmenau, 2017
%   Sec. 2.5.2, pp. 33
%
%   Input:
%       zrot    Rotation angle around z-axis in [rad], Default = 0
%       yrot    Rotation angle around y-axis in [rad], Default = 0
%       xrot    Rotation angle around x-axis in [rad], Default = 0
%       phi     Azimuth input angles in [rad], Default = 0
%       theta   Elevation input angles in [rad], Default = 0
%
%   Output:
%       R       Antenna rotation matrix (Rz * Ry * Rx)
%       phiL    Azimuth angles for pattern interpolation in [rad]
%       thetaL  Elevation angles for pattern interpolation in [rad]
%       gamma   The polarization rotation angle in [rad]
%
%
% QuaDRiGa Copyright (C) 2011-2020
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

% Assign defaults
precision = 'double';
use_single_precision = false;
if ~exist( 'phi','var' ) || isempty( phi )
    if ~exist( 'theta','var' ) || isempty( theta )
        theta = 0;
    elseif isa( theta , 'single' )
        precision = 'single';
        use_single_precision = true;
    end
    phi = zeros( size( theta ), precision );
elseif ~exist( 'theta','var' ) || isempty( theta )
    if isa( phi , 'single' ) 
        precision = 'single';
        use_single_precision = true;
    end
    theta =  zeros( size( phi ),precision);
elseif isa( phi , 'single' )
    precision = 'single';
    use_single_precision = true;
    theta = single( theta );
end

if ~exist( 'zrot','var' ) || isempty( zrot )
    zrot = zeros(1,precision);
end
if ~exist( 'yrot','var' ) || isempty( yrot )
    yrot = zeros(1,precision);
end
if ~exist( 'xrot','var' ) || isempty( xrot )
    xrot = zeros(1,precision);
end

% Save dimensions of the angles
input_size = size( phi );

% Put all angles in a vector
phi   = phi(:);
theta = theta(:);
if use_single_precision
    zrot  = single( zrot(:)' );
    yrot  = single( yrot(:)' );
    xrot  = single( xrot(:)' );
else
    zrot  = zrot(:)';
    yrot  = yrot(:)';
    xrot  = xrot(:)';
end

% Store the number of values
no_angles = numel( phi );
no_dir    = [ numel( zrot ), numel( yrot ), numel( xrot ) ];

if any( no_dir ~= no_angles & no_dir ~= 1 ) && nargout > 1
    error('Number of orientations must be empty, scalar, or match the number of angles.')
end

zd = zeros( 1, max(no_dir),precision);
od = ones(  1, max(no_dir),precision);

% Generate z-rotation matrix
if all( zrot == 0 )
    Rz = eye(3,precision);
    no_dir(1) = 0;
else
    co = cos(zrot);
    si = sin(zrot);
    if no_dir(1) == 1
        Rz = [  co;si;0 ;  -si;co;0 ;    0;0;1 ];
    else
        Rz = [ co;si;zd ; -si;co;zd ; zd;zd;od ];
    end
    Rz = reshape( Rz, 3,3,[] );
end

% Generate y-rotation matrix
if all( yrot == 0 )
    Ry = eye(3,precision);
    no_dir(2) = 0;
else
    co = cos(yrot);
    si = sin(yrot);
    if no_dir(2) == 1
        Ry = [  co;0;-si ;    0;1;0 ;  si;0;co ];
    else
        Ry = [ co;zd;-si ; zd;od;zd ; si;zd;co ];
    end
    Ry = reshape( Ry, 3,3,[] );
end

% Generate x-rotation matrix
if all( xrot == 0 )
    Rx = eye(3,precision);
    no_dir(3) = 0;
else
    co = cos(xrot);
    si = sin(xrot);
    if no_dir(3) == 1
        Rx = [    1;0;0 ;  0;co;si ;  0;-si;co ];
    else
        Rx = [ od;zd;zd ; zd;co;si ; zd;-si;co ];
    end
    Rx = reshape( Rx, 3,3,[] );
end

% Generate combined rotation matrix
if all( no_dir == 0 )
    R = eye(3,precision);
elseif all( no_dir([1,2]) == 0 )
    R = Rx;
elseif all( no_dir([1,3]) == 0 )
    R = Ry;
elseif all( no_dir([2,3]) == 0 )
    R = Rz;
else
    R = zeros( 3,3,max(no_dir),precision );
    for n = 1 : max(no_dir)
        ii = uint16( no_dir>1 )*(n-1) + 1;
        R(:,:,n)  = Rz(:,:,ii(1)) * Ry(:,:,ii(2)) * Rx(:,:,ii(3)) ;
    end
end
no_dir = max( no_dir );

% Calculate the angles for the antenna interpolation
if nargout > 1
    
    % Transform input angles from geographic coordinates to Cartesian coordinates
    C = [ cos( phi ), cos( theta ), sin( theta ) ];
    C(:,1) = C(:,1) .* C(:,2);
    C(:,2) = C(:,2) .* sin( phi );

    % Same as, but faster:
    %   C = zeros( no_angles,3,precision );
    %   C(:,1) = cos( theta ) .* cos( phi );
    %   C(:,2) = cos( theta ) .* sin( phi );
    %   C(:,3) = sin( theta );
    
    % Apply the rotation
    % Note: This is equal to: "R.' * C'", but avoids the trnspose operations to save time
    if no_dir == 1
        C = C * R;
    else
        for n = 1 : no_dir
            C(n,:) = C(n,:) * R(:,:,n);
        end
    end
    
    % Transform back to spheric coordinates
    C(C(:,3)> 1,3) = 1;         % Possible numeric instability
    C(C(:,3)<-1,3) = -1;        % Possible numeric instability
    thetaL = asin( C(:,3) );
    phiL   = atan2( C(:,2),C(:,1) );
end

% Calculate the polarization rotation angle "gamma"
if nargout > 3
    
    % Avoid calculating sin and cos multiple times
    Eth_o = sin(theta);
    Eph_o = cos(phi);
    Eth_n = sin(phi);
    
    % Geographic basis vectors in theta direction (original angles)
    Eth_o = [ Eth_o.*Eph_o , Eth_o.*Eth_n, -cos(theta) ];
    
    % Spherical/geographical basis vector in phi direction (original angles)
    Eph_o = [ -Eth_n , Eph_o , zeros(no_angles,1,precision) ];
    
    % Geographic basis vector in theta direction (new angles)
    Eth_n = sin(thetaL);
    Eth_n = [ Eth_n.*cos(phiL) , Eth_n.*sin(phiL), -cos(thetaL) ];
    
    % Apply rotation to new angles
    if no_dir == 1
        Eth_n = Eth_n * R';
    else
        for n = 1 : no_dir
            Eth_n(n,:) = Eth_n(n,:) * R(:,:,n)';
        end
    end
    cos_gamma = sum( Eth_o .* Eth_n , 2 );
    sin_gamma = sum( Eph_o .* Eth_n , 2 );
    
    gamma = atan2( sin_gamma, cos_gamma );
    gamma = reshape( gamma, input_size );
end

% Reshape output to match input size
if nargout > 1
    thetaL = reshape( thetaL, input_size );
    phiL = reshape( phiL, input_size );
end

end
