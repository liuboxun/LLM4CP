function pos = trans_global2ue( lon, lat, hnn, ReferenceCoord )
%TRANS_UE2GLOBAL Transforms global WGS84 positions to local QuaDRiGa positions
%
% Input:
%   lon                 Longitude in degree [ 1 x nP ]
%   lat                 Latitude in degree  [ 1 x nP ]
%   hnn                 Height above sea level in meters [ 1 x nP ]
%   ReferenceCoord      Reference positions on Earth [lon, lat] or [lon, lat, hnn]
%
% Output:
%   pos                 Positions in local Cartesian coordinates in meters [ 3 x nP ]
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

if ~exist('lon','var') || isempty( lon )
    error('??? Longitude not given');
end
lon = reshape( lon,1,[] );

nP = numel(lon);
oP = ones( 1,nP );

if ~exist('lat','var') || isempty( lat ) || numel(lat) ~= nP
    error('??? Latitude not given');
end
lat = reshape( lat,1,[] );

if ~exist('hnn','var') || isempty( hnn )
    hnn = zeros(1,nP);
end

% Radius of the Earth in [m]
R_e = 6378.137*1e3;

% Get the UE positions in XYZ coordinates
xyzUE = (R_e + hnnUE).*[  cosd(lonUE).*cosd(latUE); sind(lonUE).*cosd(latUE); sind(latUE) ];

% Align with QuaDRiGa X-Y-Z coordinates (east = 0, counter-clockwise)
Rz = [cosd(-lonUE) -sind(-lonUE) 0; sind(-lonUE) cosd(-lonUE) 0; 0 0 1];
Ry = [cosd(latUE-90) 0 sind(latUE-90); 0 1 0; -sind(latUE-90) 0 cosd(latUE-90)];
Rzz = [cosd(-90) -sind(-90) 0; sind(-90) cosd(-90) 0; 0 0 1];
Rr = Rzz*Ry*Rz;

% Calculate the positions in Cartesian coordinates
pos = zeros( 3, nP );
pos(1,:) = cosd(lon).*cosd(lat);
pos(2,:) = sind(lon).*cosd(lat);
pos(3,:) = sind(lat);
pos = [1;1;1]*(R_e + hnn) .* pos;

% Calculate relative satellite positions from UE perspective
pos = pos - xyzUE(:,oP);
pos = Rr * pos;                               % QuaDRiGa Frame

end
