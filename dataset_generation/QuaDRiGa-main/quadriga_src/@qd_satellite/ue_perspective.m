function [ xyzU, visible, orientation ] = ue_perspective( h_qd_satellite, ue_pos, t, i_sat )
%UE_PERSPECTIVE Predicts the satellite positions and orientation in local QuaDRiGa coordinates
%
% Calling object:
%   Single object
%
% Description:
%   The QuaDRiGa reference coordinate system is defined by Cartesian (x,y,z) coordinates. This
%   method attaches a tangential plane at the UE positions given by geographic coordinates
%   (longitude, latitude). The origin (0,0,0) of the local QuaDRiGa coordinates is is placed at the
%   UE position and the satellite coordinates at time t are transformed into local coordinates. The
%   satellite is oriented such that the x-axis of its local coordinate system points into the
%   direction of flight and its z-axis points to the center of the Earth (for circular orbits).
%   Hence, it rotates at a constant rate of one revolution per orbit. This orientation is
%   transformed into local QuaDRiGa coordinates as well.
%
% Input:
%   ue_pos
%   A two-element vector describing defining origin of the tangential plane on the surface of the
%   Earth in geographic coordinates (latitude,longitude) in [degree]
%
%   t
%   Vector containing the T time points in seconds relative to the epoch (simulation start).
%   Alternatively, it is possible to use the current clock-time by providing a string describing
%   the clock offset relative to UTC time, e.g. 'utc+02' stands for Central European Summer Time
%   (CEST).
%
%   i_sat
%   Vector containing the S satellite indices for which the orbit should be predicted (optional).
%   By default, all satellites are used.
%
% Output:
%   xyzU
%   Positions of the satellite in local QuaDRiGa coordinates (relative to the tangential plane,
%   including Earths rotation);  Values are in [km]; Dimensions: [ 3 x T x S ]
%
%   visible
%   Logic vector indicating positions above the horizon; Dimensions: [ T x S ]
%
%   orientation
%   This vector describes the orientation of the satellite. The reference system for aircraft
%   principal axes is used. The first value describes the "bank angle", i.e. the orientation around
%   an axis drawn through the body of the satellite from tail to nose. The second value describes
%   the "tilt angle", i.e. the vertical angle relative to the horizontal plane. The third value
%   describes the bearing or "heading angle", in mathematical sense. All values are given in
%   [degree]; Dimensions: [ 3 x T x S ]
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

if numel( h_qd_satellite ) > 1
    error('QuaDRiGa:h_qd_satellite:ue_perspective','ue_perspective not defined for object arrays.');
else
    h_qd_satellite = h_qd_satellite(1,1); % workaround for octave
end

% Parse input arguments
if ~exist('ue_pos','var') || isempty( ue_pos )
    ue_pos = [ 13.3249472 , 52.5163194 ];  % HHI
end
lonUE = ue_pos( 1 );
latUE = ue_pos( 2 );

if ~exist('t','var') || isempty( t )
    t = 0;
elseif ischar(t) && numel(t) == 6 && strcmp(t(1:3),'utc') && ~isempty( h_qd_satellite.epoch )
    t = ( now - str2double(t(4:6))/24 - h_qd_satellite.epoch)*86400;
elseif ~( isnumeric(t) && isreal(t) )
    error('QuaDRiGa:h_qd_satellite:init_layout','Invalid time offset or epoch not given.');
else
    t = reshape( t,[],1 );
end

if ~exist('i_sat','var') || isempty( i_sat )
    i_sat = 1 : h_qd_satellite.n_satellites;
end

nT = numel( t );
nS = numel( i_sat );
oO = ones( 1,nS*nT );
oS = ones( 1,nS );

% Get the UE positions in XYZ coordinates
xyzUE = qd_satellite.R_e.*[  cosd(lonUE).*cosd(latUE); sind(lonUE).*cosd(latUE); sind(latUE) ];

% Rotate to north pole around y-axis and by longitude angle around z-axis
% Align with QuaDRiGa X-Y-Z coordinates (east = 0, counter-clockwise)
sLo = sind( lonUE );
cLo = cosd( lonUE );
sLa = sind( latUE );
cLa = cosd( latUE );
Rr = [ -sLo, cLo, 0 ; -sLa*cLo, -sLa*sLo, cLa ; cLa*cLo, cLa*sLo, sLa  ];

% Get the satellite positions in rotating XYZ coordinates at time t
[ xyzI, xyzR ] = orbit_predictor( h_qd_satellite, t, i_sat );
xyzR = reshape( xyzR, 3,[] );                	% Single vector

% Calculate relative satellite positions from UE perspective
xyzU = xyzR - xyzUE(:,oO);
xyzU = Rr * xyzU;                               % QuaDRiGa Frame

% Format output
xyzU = reshape( xyzU, 3, nT, nS );

% Determine visibility
visible = reshape( xyzU(3,:,:) > 0 , nT,nS );

if nargout > 2                                  % Get the orientation
    
    % Calculate the satellite positions 0.1 seconds later (inertial frame)
    DxyzI = orbit_predictor( h_qd_satellite, t+0.1, i_sat );
    D = DxyzI - xyzI;                          	% Speed vector in inertial frame
    
    % Calculate the z-rotation matrix for the Earths rotation
    GMST = h_qd_satellite.Omega_e*t.';          % Angular of rotation of the Earth [deg]
    sLo = sind( GMST );
    cLo = cosd( GMST );

    % Transform the speed vector into the rotating reference frame
    tmp = D(1,:,:);
    D(1,:,:) =  cLo(:,:,oS).*tmp + sLo(:,:,oS).*D(2,:,:);
    D(2,:,:) = -sLo(:,:,oS).*tmp + cLo(:,:,oS).*D(2,:,:);
    
    D = reshape( D, 3,[] );                     % Single vector
    D = D ./ ([1;1;1] * sqrt(sum(D.^2,1)));    	% Normalize to obtain direction
    
    % Calculate the bank angle
    iU = xyzUE./sqrt(sum(xyzUE.^2));
    iS = xyzR ./ ([1;1;1]*sqrt(sum(xyzR.^2,1)));
    bank = asind( iU' * cross( iS, D ) );     	% Bank angle
    
    % Rotate D into QuaDRiga Frame
    D = Rr * D;                                	% QuaDRiGa Frame
    
    % Calculate heading and tilt
    heading = angle( D(1,:) + 1j*D(2,:) )*180/pi;
    tilt = angle(sqrt(D(1,:).^2 + D(2,:).^2) + 1j*D(3,:))*180/pi;
    
    % Assemble orientation vector
    orientation = [bank ; tilt ; heading];
    orientation = mod( orientation +180,360)-180;   % Map to interval [-180 , 180 )
    
    % Format output
    orientation = reshape( orientation, 3, nT, nS );
end

end
