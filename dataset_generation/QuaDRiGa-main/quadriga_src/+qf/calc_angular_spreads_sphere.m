function [ as, es, orientation, phi, theta ] = calc_angular_spreads_sphere( az, el, pow, calc_bank_angle, quantize )
%CALC_ANGULAR_SPREADS_SPHERE Calculates azimuth and elevation angular spreads with spherical wrapping
%
% Calling object:
%   None (static method)
%
% Description:
%   This method calculates the RMS azimuth and elevation angular spreads from a given set of
%   angles. The wrapping operation is done for spherical coordinates in the following way:
%
%   1. Calculate the power-weighted mean-angle in geographic coordinates
%
%   2. Apply the power-weighted mean-angle to the given "az" and "el" angles by a rotation
%      operation in 3D-Cartesian coordinates
%
%   3. Find the optimal bank angle such that the resulting azimuth angle spread is maximized and
%      the elevation angle spread is minimized
%   4. Apply the bank angle by a rotation operation in 3D-Cartesian coordinates
%
%   5. Calculate the RMS azimuth and elevation the angular spreads
%
%   Without spherical wrapping, the azimuth spread depends on the elevation spread. At the poles
%   of the sphere (e.g. if there are many paths arriving from above or below the receiver), a
%   large azimuth spread might be calculated despite the fact that energy is focused into a small
%   surface-area of the sphere. This is not considered by the default calculation method, which
%   treats azimuth and elevation angles independently. However, with spherical wrapping, the
%   average angle is calculated such that the power-weighted average always lies on the "equator"
%   of the geographic coordinate system. In this case the obtained values for the azimuth and
%   elevation spread more accurately reflect the power-distribution.
%
% Input:
%   az
%   A vector of azimuth angles in [rad] ranging from -π to π. Dimensions: [ n_ang x n_path ]
%
%   el
%   A vector of elevation angles in [rad] ranging from -π/2 to π/2. Dimensions: [ n_ang x n_path ]
%
%   pow
%   A vector of path powers in [W]. Dimensions: [ n_ang x n_path ] or [ 1 x n_path ]
%
%   calc_roll_angle
%   A logical variable. If set to (1, default), the optimal roll angle is calculated such that the
%   azimuth angle spread is maximized and the elevation angle spread is minimized. If set to 0, no
%   roll angle is calculated.
%
%   quantize
%   This parameter in units of [deg] (scalar value) can be used to group paths in the angular
%   domain. For example: The resolution of an array antenna might be 3 degrees. However, several
%   paths might be estimated from different snapshots at slightly different angles (e.g. one at 10
%   deg and another at 10.3 deg). Since the angular difference (0.3 deg) is below the array
%   resolution (3 deg), they might belong to the same path. Thus, setting the angular quantization
%   to 3 deg will sum up the powers of the two paths and treat them as one. Default: 0 deg (all
%   paths are treated as independent)
%
% Output:
%   as
%   The RMS azimuth angular spread in [rad] for each angle vector. Dimensions: [ n_ang x 1 ]
%
%   es
%   The RMS elevation angular spread in [rad] for each angle vector. Dimensions: [ n_ang x 1 ]
%
%   orientation
%   This 3-element vector describes direction of the power-weighted mean-angle. The reference
%   system for aircraft principal axes is used. The first value describes the "bank angle", i.e.
%   the orientation around an axis drawn through the body of the vehicle from tail to nose in the
%   normal direction of movement. Positive rotation is clockwise (seen from the pilot/drivers
%   perspective). The second value describes the "tilt angle", i.e. the vertical angle relative to
%   the horizontal plane; positive values point upwards. The third value describes the bearing or
%   "heading angle", in mathematic sense. Dimensions: [ 3 x n_ang ]
%
%   phi
%   The rotated azimuth angles in [rad]. Dimensions: [ n_ang x n_path ]
%
%   theta
%   The rotated elevation angles in [rad]. Dimensions: [ n_ang x n_path ]
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

if ~exist('calc_bank_angle','var') || isempty( calc_bank_angle )
    calc_bank_angle = true;
end

if ~exist('quantize','var') || isempty( quantize )
    quantize = 0;
else
    quantize = quantize * pi/180; % Transform to [rad]
end

N = size( az,1 );
L = size( az,2 );
oL = ones( 1,size(az,2 ));

if size( el,1 ) < N
    el = zeros( size( az ));
end
if size( pow,1 ) < N
    pow = pow( ones(1,N), : );
end

% Normalize powers
pt = sum( pow,2 );
pow = pow./pt( :,oL  );

% Pre-process data
powX  = permute( pow, [3,2,1] );
phi   = permute( az,  [3,2,1] );
theta = permute( el,  [3,2,1] );

% Placeholder
C     = zeros( 3,L,N );
R     = zeros( 3,3,N );
heading   = angle( sum( powX.*exp( 1j*phi ) , 2 ) );              	% Calculate heading angle
tilt = zeros( 1,1,N );
bank  = zeros( 1,1,N );
orientation = zeros( 3,1,N );

upd = true(1,N);
lp  = 1;
while any( upd ) && lp < 10
    
    phi(:,:,upd) = phi(:,:,upd) - heading( :,oL,upd );              % Apply heading angle (z-rotation)
    orientation(3,1,upd) = orientation(3,1,upd) + heading(1,1,upd); % Save orientation change
    
    tilt(:,:,upd) = angle( sum( powX(:,:,upd).*exp( 1j*theta(:,:,upd) ) , 2 ) );  % Calculate tilt angle
    
    R(:) = 0;                                                   % Calculate rotation matrix
    co = cos( tilt(:,:,upd) );
    si = sin( tilt(:,:,upd) );
    R(1,1,upd) = co;
    R(1,3,upd) = si;
    R(3,1,upd) = -si;
    R(3,3,upd) = co;
    R(2,2,upd) = 1;
    
    C(1,:,upd) = cos( phi(:,:,upd) ) .* cos( theta(:,:,upd) ); % Transform to Cartesian coordinates
    C(2,:,upd) = sin( phi(:,:,upd) ) .* cos( theta(:,:,upd) );
    C(3,:,upd) = sin( theta(:,:,upd) );
    
    for n = 1 : N                                               % Apply tilt angle
        if upd( n )
            C(:,:,n) = R(:,:,n) * C(:,:,n);
        end
    end
    
    theta(:,:,upd) = atan2(C(3,:,upd),hypot( C(1,:,upd),C(2,:,upd) ));  % Transform to Spheric coordinates
    phi(:,:,upd)   = atan2(C(2,:,upd),C(1,:,upd));
    orientation(2,1,upd) = orientation(2,1,upd) + tilt(1,1,upd);       % Save orientation change
    
    heading(:,:,upd) = angle( sum( powX(:,:,upd).*exp( 1j*phi(:,:,upd) ) , 2 ) );      % Calculate heading angle
    tilt(:,:,upd) = angle( sum( powX(:,:,upd).*exp( 1j*theta(:,:,upd) ) , 2 ) );  % Calculate tilt angle
    
    upd = abs(heading(1,:)) > 0.001 | abs(tilt(1,:)) > 0.001;
    lp = lp + 1;
end

% Calculate angular spreads
as = qf.calc_angular_spreads( permute(phi,[3,2,1]),   pow, 1, quantize );
es = qf.calc_angular_spreads( permute(theta,[3,2,1]), pow, 1, quantize );

% Try to find the bank angle that maximized the AS and minimized the ES
if calc_bank_angle
    
    cst   = es./as;                                 % Initial cost function at 0 deg bank angle
    cstN  = cst;                                    % Placeholder for cost function updated
    bankN = bank;                                   % Placeholder for bank angle updates
    thetaN = theta;
    phiN  = phi;
    dir   = ones(size(bank));                       % Initial search direction
    step  = ones(size(bank)) * 10 * pi/180;         % Initial step-size
    
    % Transform to Cartesian coordinates
    C(1,:,:) = cos( phi(1,:,:) ) .* cos( theta(1,:,:) );
    C(2,:,:) = sin( phi(1,:,:) ) .* cos( theta(1,:,:) );
    C(3,:,:) = sin( theta(1,:,:) );
    D = C;                                          % Placeholder for rotated angles
    
    lp = 1;                                         % Loop counter
    upd = true(N,1);                                % Indicator which values nnet to be updated
    while any( upd ) && lp < 50
        
        % Set new bank angle
        bankN(:,:,upd) = bank(:,:,upd) + dir(:,:,upd) .* step(:,:,upd);
        
        R(:) = 0;                                  	% Calculate rotation matrix
        co = cos( bankN(:,:,upd) );
        si = sin( bankN(:,:,upd) );
        R(1,1,upd) = 1;
        R(2,2,upd) = co;
        R(2,3,upd) = -si;
        R(3,2,upd) = si;
        R(3,3,upd) = co;
        
        for n = 1 : N                               % Apply bank angle
            if upd( n )
                D(:,:,n) = R(:,:,n) * C(:,:,n);
            end
        end
        
        % Transform to Spheric coordinates and calculate updated costs
        thetaN(:,:,upd) = atan2(D(3,:,upd),hypot( D(1,:,upd),D(2,:,upd) ));
        phiN(:,:,upd) = atan2(D(2,:,upd),D(1,:,upd));
        asN = qf.calc_angular_spreads( permute(phiN(:,:,upd),[3,2,1]), pow(upd,:), 1, quantize );
        esN = qf.calc_angular_spreads( permute(thetaN(:,:,upd),[3,2,1]), pow(upd,:), 1, quantize );
        
        cstN(upd) = esN ./ asN;
        
        ind = cstN > cst & upd;                     % Results is worse
        dir(:,:,ind)  = -dir(:,:,ind);              % Direction change
        step(:,:,ind) = 0.37 * step(:,:,ind);       % Reduce step size
        
        ind = cstN < cst & upd;                     % Result is better
        bank(:,:,ind) = bankN(:,:,ind);             % Set new bank
        theta(:,:,ind) = thetaN(:,:,ind);           % Store new theta values
        phi(:,:,ind) = phiN(:,:,ind);               % Store new phi values
        cst(ind) = cstN(ind);                       % Update cost
        
        upd = step(:) > 1e-6;                       % Resolution limit
        lp = lp + 1;
    end
    
    orientation(1,1,:) = bank;                      % Save bank angle to orientation
    
    if 0 % Visualize alignment
        % Test angles
        test = [ 0 , -90 : 1 : 90 ];
        nT = numel( test );
        oT = ones( 1,nT );
        
        % Extend variables
        phi   = phi( oT,:,: );
        theta = theta( oT,:,: );
        as = as(:,oT);
        es = es(:,oT);
        
        for t = 2 : nT
            R(:) = 0;                                                   % Calculate rotation matrix
            co = cos( test(t)*pi/180 );
            si = sin( test(t)*pi/180 );
            R(1,1,:) = 1;
            R(2,2,:) = co;
            R(2,3,:) = -si;
            R(3,2,:) = si;
            R(3,3,:) = co;
            
            C(1,:,:) = cos( phi(t,:,:) ) .* cos( theta(t,:,:) ); % Transform to Cartesian coordinates
            C(2,:,:) = sin( phi(t,:,:) ) .* cos( theta(t,:,:) );
            C(3,:,:) = sin( theta(t,:,:) );
            
            for n = 1 : N                                               % Apply tilt angle
                C(:,:,n) = R(:,:,n) * C(:,:,n);
            end
            
            theta(t,:,:) = atan2(C(3,:,:),hypot( C(1,:,:),C(2,:,:) ));  % Transform to Spheric coordinates
            phi(t,:,:)   = atan2(C(2,:,:),C(1,:,:));
            
            as(:,t) = qf.calc_angular_spreads( permute(phi(t,:,:),[3,2,1]), pow, 1, quantize );
            es(:,t) = qf.calc_angular_spreads( permute(theta(t,:,:),[3,2,1]), pow, 1, quantize );
        end
        imagesc(as./es);
    end
    
    % Calculate angular spreads
    as = qf.calc_angular_spreads( permute(phi,[3,2,1]),   pow, 1, quantize );
    es = qf.calc_angular_spreads( permute(theta,[3,2,1]), pow, 1, quantize );
end

orientation = permute( orientation, [1,3,2] );
orientation = angle(exp( 1j*orientation ));

end
