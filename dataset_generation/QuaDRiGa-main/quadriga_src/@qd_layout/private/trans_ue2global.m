function [ lon, lat, hnn ] = trans_ue2global( pos, ReferenceCoord )
%TRANS_UE2GLOBAL Transforms local QuaDRiGa positions to global WGS84 positions
%
% Input:
%   pos                 Positions in local Cartesian coordinates [ 3 x nP ]
%   ReferenceCoord      Reference positions on Earth [lon, lat]
%
% Output:
%   lon                 Longitude [ 1 x nP ]
%   lat                 Latitude [ 1 x nP ]
%   hnn                 Height above sea level [ 1 x nP ]
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

% Parse input arguments
if ~exist('ReferenceCoord','var') || isempty( ReferenceCoord )
    ReferenceCoord = [ 13.3249472 , 52.5163194 ];  % HHI
end
lonUE = ReferenceCoord( 1 );
latUE = ReferenceCoord( 2 );
if numel( ReferenceCoord ) == 3
    hnnUE = ReferenceCoord(3);
else
    hnnUE = 0;
end

nP = size( pos,2 );
oP = ones( 1,nP );

% Radius of the Earth in [m]
R_e = 6378.137*1e3;

% Get the UE positions in XYZ coordinates
xyzUE = (R_e + hnnUE).*[  cosd(lonUE).*cosd(latUE); sind(lonUE).*cosd(latUE); sind(latUE) ];

% QuaDRiGa X-Y-Z coordinates (east = 0, counter-clockwise) with WGS 84 coordinates
Rzz = [cosd(90) -sind(90) 0; sind(90) cosd(90) 0; 0 0 1];
Ry = [cosd(-latUE+90) 0 sind(-latUE+90); 0 1 0; -sind(-latUE+90) 0 cosd(-latUE+90)];
Rz = [cosd(lonUE) -sind(lonUE) 0; sind(lonUE) cosd(lonUE) 0; 0 0 1];
Rr = Rz*Ry*Rzz;

% Transform positions to Global xyz coordinates
xyzG = Rr * pos;
xyzG = xyzG + xyzUE(:,oP);

% Calculate Latitude and Longitude
lat = atand( xyzG(3,:,:)./sqrt( sum( xyzG(1:2,:,:).^2,1) )); 	% Latitude
lon = angle( xyzG(1,:,:) + xyzG(2,:,:)*1j )*180/pi;             % Longitude (inertial frame)

% Height above sea level
hnn = sqrt(sum(xyzG.^2,1))-R_e;                                 
hnn( abs( hnn ) < 1e-7 ) = 0;                                   % Special case when hnn is close to 0

end
