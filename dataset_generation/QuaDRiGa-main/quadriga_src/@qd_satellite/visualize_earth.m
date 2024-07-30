function visualize_earth( h_qd_satellite, observation_time, i_sat )
%VISUALIZE_EARTH Plots the satellite position in the rotating reference frame
%
% Calling object:
%   Single object
%
% Description:
%   Plots the satellite position in 3D Cartesian coordinates with the Earth at the center. Earth's
%   rotation is taken into account.
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
    error('QuaDRiGa:h_qd_satellite:visualize_earth','visualize_earth not defined for object arrays.');
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

% Load earth texture
information = what('quadriga_src');
earth_texture = [information.path,filesep,'@qd_satellite',filesep,'private',filesep,'800px-Nasa_land_ocean_ice_8192.jpg'];
cdata = imread(earth_texture);

% add Earth
phi = -180:10:180;
theta = (0:10:180);
[Phi, Theta] = meshgrid(phi, theta);
earth_x = (qd_satellite.R_e*cosd(Phi).*sind(Theta));
earth_y = (qd_satellite.R_e*sind(Phi).*sind(Theta));
earth_z = (qd_satellite.R_e*ones(size(Phi)).*cosd(Theta));

figure
s = surf(earth_x, earth_y, earth_z, ones(size(earth_z)),'EdgeColor','b','FaceColor','b');
set(s, 'FaceColor', 'texturemap', 'CData', double(cdata)/255, 'FaceAlpha', 1, 'EdgeColor', 'none');

axis equal
hold on

% Get the observation time
tim = observation_time;

% Draw orpit-path for single satellite
if nS == 1 && numel( tim ) == 1
    tt = (-h_qd_satellite.orbit_period(i_sat)/2 : 60 : h_qd_satellite.orbit_period(i_sat)/2) + tim;
    [ ~, xyz_pos_orb] = orbit_predictor( h_qd_satellite, tt, i_sat );
    plot3( xyz_pos_orb(1,:), xyz_pos_orb(2,:), xyz_pos_orb(3,:),'-m');
end

[ ~, xyz_pos ] = orbit_predictor( h_qd_satellite, tim, i_sat );
[ ~, Dxyz_pos ] = orbit_predictor( h_qd_satellite, tim+0.1, i_sat );

s_min = Inf;
s_max = 0;
for n = 1 : nS
    speed_v = sqrt(sum((Dxyz_pos(:,:,n) - xyz_pos(:,:,n)).^2,1))*10;
    if numel( speed_v ) < 4
        scatter3( xyz_pos(1,:,n), xyz_pos(2,:,n), xyz_pos(3,:,n), 'g','filled');
    else
        scatter3( xyz_pos(1,:,n), xyz_pos(2,:,n), xyz_pos(3,:,n), [], speed_v,'filled');
    end
    plot3(xyz_pos(1,:,n), xyz_pos(2,:,n), xyz_pos(3,:,n),'-k');
    s_min = min( [s_min,speed_v] );
    s_max = max( [s_max,speed_v] );
    if numel(tim) == 1 && ~isempty( h_qd_satellite.sat_name )
        text( xyz_pos(1,1,n), xyz_pos(2,1,n), xyz_pos(3,1,n), [' ',h_qd_satellite.sat_name{i_sat(n)}], 'Color','k' );
    end
end

caxis([s_min*0.99999,s_max*1.00001])

xlabel('x in km')
ylabel('y in km')
zlabel('z in km')
hold off
axis equal
view([110, 10])
rotate3d on

end
