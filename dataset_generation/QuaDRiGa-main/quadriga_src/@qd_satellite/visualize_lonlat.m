function visualize_lonlat( h_qd_satellite, observation_time, i_sat )
%VISUALIZE_LONLAT Plots the satellite ground track
%
% Calling object:
%   Single object
%
% Description:
%   A ground track or ground trace is the path on the surface of the Earth directly below the
%   satellite. It is the projection of the satellite's orbit onto the surface of the Earth. A
%   satellite ground track may be thought of as a path along the Earth's surface which traces the
%   movement of an imaginary line between the satellite and the center of the Earth. In other
%   words, the ground track is the set of points at which the satellite will pass directly
%   overhead, or cross the zenith, in the frame of reference of a ground observer.
%
% Input:
%   observation_time
%   Vector containing the T time points in seconds relative to the epoch (simulation start).
%   Alternatively, it is possible to use the current clock-time by providing a string describing
%   the clock offset relative to UTC time, e.g. 'utc+02' stands for Central European Summer Time
%   (CEST). When clock-time is used, satellite names are plotted next to the satellite position. If
%   'observation_time' is not given, satellite trajectories are plotted for one full orbit and
%   names are not plotted.
%
%   i_sat
%   Vector containing the S satellite indices that should be included in the plot. By default, all
%   satellites are used. If only one satellite position requested for a single time point (e.g.
%   clock-time), the orbital track is shown in addition to the satellite position.
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
    error('QuaDRiGa:h_qd_satellite:visualize_lonlat','visualize_lonlat not defined for object arrays.');
else
    h_qd_satellite = h_qd_satellite(1,1); % workaround for octave
end

if ~exist('i_sat','var') || isempty( i_sat )
    i_sat = 1 : h_qd_satellite.n_satellites;
end
nS = numel( i_sat );

% If observation time is not given, use the smallest orbital period
if ~exist('observation_time','var') || isempty( observation_time )
    tmp = min(h_qd_satellite.orbit_period(i_sat));
    observation_time = (0:0.01:1)*tmp;
elseif ischar(observation_time) && numel(observation_time) == 6 && strcmp(observation_time(1:3),'utc') && ~isempty( h_qd_satellite.epoch )
    observation_time = ( now - str2double(observation_time(4:6))/24 - h_qd_satellite.epoch)*86400;
end
observation_time = reshape( observation_time,[],1 );
nT = numel( observation_time );

% Load earth texture
information = what('quadriga_src');
earth_texture = [information.path,filesep,'@qd_satellite',filesep,'private',filesep,'Land_ocean_ice_2048.jpg'];
cdata = imread(earth_texture);

% Plot earth
figure
image([-180, 180], [90, -90], cdata,'AlphaData',0.7);
set(gca,'YDir','normal')
hold on

% Get the observation time
tim = observation_time;

% Draw orpit-path for single satellite
if nS == 1 && numel( tim ) == 1
    tt = (-h_qd_satellite.orbit_period(i_sat)/2 : 60 : h_qd_satellite.orbit_period(i_sat)/2) + tim;
    [ ~, ~, ~, lat_orb, ~, lon_orb ] = orbit_predictor( h_qd_satellite, tt, i_sat );
    
    ii = [0; find( abs( diff( lon_orb ) ) > 180 ); numel(lon_orb) ];
    for n = 1 : numel(ii) - 1
        iii = ii(n)+1 : ii(n+1);
        plot( lon_orb(iii), lat_orb(iii),'-m');
     end
end

[ ~, xyz_pos, ~, lat, ~, lon ] = orbit_predictor( h_qd_satellite, tim, i_sat );
[ ~, Dxyz_pos ] = orbit_predictor( h_qd_satellite, tim+0.1, i_sat );

s_min = Inf;
s_max = 0;
for n = 1 : nS
    speed_v = sqrt(sum((Dxyz_pos(:,:,n) - xyz_pos(:,:,n)).^2,1))*10;
    ii = [ 0 ; find( abs(diff( lon(:,n) )) > 180 ) ; nT ];
    for iS = 1 : numel(ii)-1
        ind = ii(iS)+1:ii(iS+1);
        plot( lon(ind,n).', lat(ind,n).','-m');
    end
    if numel( speed_v ) < 4
        scatter( lon(:,n).', lat(:,n).', [], 'g','filled');
    else
        scatter( lon(:,n).', lat(:,n).', [], speed_v,'filled');
    end
    if numel(tim) == 1 && ~isempty( h_qd_satellite.sat_name )
        text( lon(1,n), lat(1,n), [' ',h_qd_satellite.sat_name{i_sat(n)}], 'Color','g' );
    end
    s_min = min( [s_min,speed_v] );
    s_max = max( [s_max,speed_v] );
end

caxis([s_min*0.99999,s_max*1.00001])

xlabel('longitude in degrees')
ylabel('latitude in degrees')
grid on
hold off
axis image
zoom on

end
