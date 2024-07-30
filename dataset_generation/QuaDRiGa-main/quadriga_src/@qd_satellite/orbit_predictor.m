function [ xyzI, xyzR, r, lat, lonI, lonR, pq, Omega, omega, v ] = orbit_predictor( h_qd_satellite, t, i_sat )
%ORBIT_PREDICTOR Non-GSO satellite orbit predictor
%
% Calling object:
%   Single object
%
% Description:
%   Given the orbital elements given by the qd_satellite' object, standard orbit mechanics are used
%   to predict the position of the satellite at future times.
%
% Input:
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
%   xyzI
%   Positions of the satellite in Cartesian coordinates (inertial, non-rotating reference frame);
%   Values are in [km]; Dimensions: [ 3 x T x S ].
%
%   xyzR
%   Positions of the satellite in Cartesian coordinates (rotating reference frame taking Earths
%   rotation into account);  Values are in [km]; Dimensions: [ 3 x T x S ]
%
%   r
%   Distance from center of Earth to the satellite in [km]. Dimensions: [ T x S ]
%
%   lat
%   Geographic latitude in [degree]. Dimensions: [ T x S ]
%
%   lonI
%   Geographic longitude in [degree] (inertial, non-rotating reference frame). Dimensions: [ T x S
%   ]
%
%   lonR
%   Geographic longitude in [degree] (rotating reference frame taking Earths rotation into
%   account). Dimensions: [T x S ]
%
%   pq
%   The position of the satellite within the orbital plane in (P, Q) coordinates; Values are in
%   [km]; Dimensions: [ 2 x T x S ]
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
    error('QuaDRiGa:h_qd_satellite:orbit_predictor','orbit_predictor not defined for object arrays.');
else
    h_qd_satellite = h_qd_satellite(1,1); % workaround for octave
end

% Parse input arguments
if ~exist('t','var')
    error('QuaDRiGa:h_qd_satellite:orbit_predictor','Input variable t is not defined.');
elseif ischar(t) && numel(t) == 6 && strcmp(t(1:3),'utc') && ~isempty( h_qd_satellite.epoch )
    t = ( now - str2double(t(4:6))/24 - h_qd_satellite.epoch)*86400;
elseif ~( isnumeric(t) && isreal(t) )
    error('QuaDRiGa:h_qd_satellite:orbit_predictor','Invalid time offset or epoch not given.');
else
    t = reshape( t,[],1 );
end

if ~exist('i_sat','var') || isempty( i_sat )
    i_sat = 1 : h_qd_satellite.n_satellites;
end

nT = numel( t );
oT = ones( 1,nT );
nS = numel( i_sat );
oS = ones( 1,nS );

% For the relevant t in seconds since simulation start, calculate the values of the precessing
% terms (omega, Omega, M)

% Ascending node longitude at timepoint t in [degrees]
Omega = h_qd_satellite.POmega_0(oT,i_sat) + t * h_qd_satellite.Omega_r(1,i_sat);

% Perigee argument at timepoint t in [degrees]
omega = h_qd_satellite.Pomega_0(oT,i_sat) + t * h_qd_satellite.omega_r(1,i_sat);

% Mean anomaly at timepoint t in [degrees]
M = h_qd_satellite.M_0(oT,i_sat) + t * h_qd_satellite.n_bar(1,i_sat);
M = mod( M +180,360)-180;                               % Map to interval [-180 , 180 )

% From M calculate the eccentric anomaly E using an iteration for elliptical orbits
e = h_qd_satellite.Pe( oT,i_sat );                      % The orbital eccentricity
E = M;                                                  % Eccentric anomaly for circular orbits at timepoint t in [degrees]
upd = e > 1e-12;                                        % Determine which values must be solved by iteration

% Solve "M = E - e * sind( E )" for elliptical orbits
En = E;                                                 % Initial En
Mn = En - e .* sind( En ) .* 180/pi;                    % Updated M
dM = mod( M - Mn, 360 );                                % The angle difference between Mn and M in range [0,180]
dM( dM > 180 ) = 360 - dM( dM > 180 );
dMn = dM;                                               % The initial difference
dE = 10*ones(nT,nS);                                    % Initial step size in degrees

while any( upd(:) )                                     % Iterate until M == Mn
    En(upd) = E(upd) + dE(upd);                         % Updated En
    Mn(upd) = En(upd) - e(upd) .* sind( En(upd) ) .* 180/pi;  % Updated Mn
    
    dM(upd) = M(upd) - Mn(upd);
    dM(upd) = mod( M(upd) - Mn(upd), 360 );             % The angle difference
    dM( dM > 180 & upd ) = 360 - dM( dM > 180 & upd );
    
    ii = dM < dMn & upd;                                % Estimate improved
    E(ii) = En(ii);                                     % Update E
    dMn(ii) = dM(ii);                                   % Update dMn

    ii = dM > dMn & upd;                                % Estimate got worse
    dE(ii) = -0.382 * dE(ii);                           % Change step size and direction

    upd(upd) = dMn(upd) > 1e-7;                         % Continue until M ~ Mn
end
E = mod( E +180,360)-180;                               % Map to interval [-180 , 180 )

% From E calculate the true anomaly v
v = 2*atand(sqrt((1+e)./(1-e)).*tand(E/2));
v = mod( v +180,360)-180;                               % Map to interval [-180 , 180 )

% Calculate the radius vector r
p = h_qd_satellite.p( oT, i_sat );
r = p ./ ( 1 + e.*cosd(v) );

% Calculate the position of the satellite within the orbital plane in (P, Q) coordinates
pq = cat( 3 , r.*cosd(v) , r.* sind(v) );
pq = permute( pq, [3,1,2] );

% Calculate the position of the satellite in xyz coordinates (inertial frame)
inclination =  h_qd_satellite.inclination( oT, i_sat );
xyzI = zeros( nT, nS, 3 );
if 0 % ITU proposal
    cO = cosd(Omega);
    sO = sind(Omega);
    co = cosd(omega);
    so = sind(omega);
    ci = cosd(inclination);
    si = sind(inclination);
    cv = cosd(v);
    sv = sind(v);
    xyzI(:,:,1) = cv.*( cO.*co - sO.*so.*ci ) + sv.*( -cO.*so - sO.*co.*ci );
    xyzI(:,:,2) = cv.*( sO.*co + cO.*so.*ci ) + sv.*( -sO.*so + cO.*co.*ci );
    xyzI(:,:,3) = cv.*( so.*si ) + sv.*( co.*si );
else % Alternative method using less operations
    cO = cosd(Omega);
    sO = sind(Omega);
    co = cosd(omega+v);
    so = sind(omega+v);
    ci = cosd(inclination);
    si = sind(inclination);
    xyzI(:,:,1) = co.*cO - sO.*ci.*so;
    xyzI(:,:,2) = co.*sO + cO.*ci.*so;
    xyzI(:,:,3) = so.*si;
end
xyzI = permute( r(:,:,[1 1 1]).*xyzI, [3,1,2] );

% Calculate the angular of rotation of the Earth
if isempty( h_qd_satellite.epoch )  
    % Earths 0-meridian is aligned with vernal point at the simulation start
    GMST = h_qd_satellite.Omega_e*t.';
else % Take Earths reference orientation into account
    dT = ( h_qd_satellite.epoch - h_qd_satellite.Epoch_eR ) * 86400; % Time difference in [s]
    GMST = h_qd_satellite.Omega_e*t.' + h_qd_satellite.Omega_e*dT + h_qd_satellite.Omega_eR;
end
    
% Calculate the longitude and latitude positions 
lat  = atand( xyzI(3,:,:)./sqrt( sum( xyzI(1:2,:,:).^2,1) )); 	% Latitude
lonI = angle( xyzI(1,:,:) + xyzI(2,:,:)*1j )*180/pi;            % Longitude (inertial frame)
lonR = lonI - GMST(:,:,oS);                                    	% Longitude (rotating frame)
lonR = mod( lonR +180,360)-180;                                 % Map to interval [-180 , 180 )

% Calculate the position of the satellite in xyz coordinates (rotating frame)
xyzR = zeros( 3, nT, nS );
xyzR(1,:,:) = cosd(lonR).*cosd(lat);
xyzR(2,:,:) = sind(lonR).*cosd(lat);
xyzR(3,:,:) = sind(lat);
xyzR = permute( r(:,:,[1 1 1]), [3,1,2] ) .* xyzR;

% Format output
lat  = permute( lat, [2,3,1] );
lonI = permute( lonI, [2,3,1] );
lonR = permute( lonR, [2,3,1] );

end
