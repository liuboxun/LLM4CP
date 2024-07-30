function [ as, mean_angle ] = calc_angular_spreads( ang, pow, wrap_angles, quantize )
%CALC_ANGULAR_SPREADS Calculates the angular spread in [rad]
%
% Description:
%   This function calculates the RMS angular spread from a given set of angles. It is used by the
%   "qd_builder" to map the path powers to angles.
%
% Input:
%   ang
%   A vector of angles in [rad]. Dimensions: [ n_ang x n_path ]
%
%   pow
%   A vector of path powers in [W]. Dimensions: [ n_ang x n_path ] or [ 1 x n_path ]
%
%   wrap_angles
%   A logical variable. If set to 1, angles will be wrapped around +/- pi. If set to 0, no wrapping
%   is applied. Default: 1 (with wrapping)
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
%   The RMS angular spread in [rad] for each angle vector. Dimensions: [ n_ang x 1 ]
%
%   mean_angle
%   The mean angles in [rad]. Dimensions: [ n_ang x 1 ]
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

if ~exist('wrap_angles','var') || isempty( wrap_angles )
    wrap_angles = true;
end

if ~exist('quantize','var') || isempty( quantize )
    quantize = 0;
else
    quantize = quantize * pi/180; % Transform to [rad]
    wrap_angles = true;
end

N  = size( ang,1 );
oN = ones(1,N);
if size( pow,1 ) < N
    pow = pow( oN, : );
end

% Normalize powers
pt = sum( pow,2 );
pow = pow./pt( :,ones(1,size(pow,2))  );

if wrap_angles
    mean_angle = angle( sum( pow.*exp( 1j*ang ) , 2 ) );
else
    mean_angle = sum( pow.*ang,2 ); 
end

phi = ang - mean_angle(:,ones( 1,size(ang,2) ) );

if wrap_angles
    phi = angle( exp( 1j*phi ) );
end

if quantize > 0
    
    % Quantized angle grid
    phiN = floor( -pi/quantize ) : ceil( pi/quantize );
    phiN = phiN*quantize;
    phiN = phiN( phiN > -pi+1e-10 );
    
    % Accumulate power values
    powN = zeros( N, numel(phiN) );
    for n = 1 : N
        for m = 1 : numel(phiN)
            if m ~= numel(phiN)
                ii = ( phi(n,:) >  phiN(m) - quantize/2 ) & ( phi(n,:) <= phiN(m) + quantize/2 );
            else
                ii = ( phi(n,:) >  phiN(m) - quantize/2 ) | ( phi(n,:) <= phiN(1) - quantize/2 );
            end
            powN(n,m) = sum(pow(n,ii));
        end
    end
    
    % Calculate AS
    as = qf.calc_angular_spreads( phiN, powN, 1, 0 );
    
else
    % Calculate AS
    as = sqrt( sum(pow.*(phi.^2),2) - sum( pow.*phi,2).^2 );
end

end
