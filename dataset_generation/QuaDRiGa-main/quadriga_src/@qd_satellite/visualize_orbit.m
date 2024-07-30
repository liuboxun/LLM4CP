function visualize_orbit( h_qd_satellite, i_sat )
%VISUALIZE_ORBIT Plots the satellite orbit in the inertial frame of reference
%
% Calling object:
%   Single object
%
% Description:
%   Plots the satellite orbit in 3D Cartesian coordinates with the Earth at the center. Earth's
%   rotation is not taken into account.
%
% Input:
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
    error('QuaDRiGa:h_qd_satellite:visualize_orbit','visualize_orbit not defined for object arrays.');
else
    h_qd_satellite = h_qd_satellite(1,1); % workaround for octave
end

if ~exist('i_sat','var') || isempty( i_sat )
    i_sat = 1 : h_qd_satellite.n_satellites;
end
nS = numel( i_sat );

% Plot a sphere
phi = 0:10:360;
theta = (0:10:180);
[Phi, Theta] = meshgrid(phi, theta);
earth_x = (qd_satellite.R_e*cosd(Phi).*sind(Theta));
earth_y = (qd_satellite.R_e*sind(Phi).*sind(Theta));
earth_z = (qd_satellite.R_e*ones(size(Phi)).*cosd(Theta));

figure
surf(earth_x, earth_y, earth_z, ones(size(earth_z)),'EdgeColor','b','FaceColor','b','FaceAlpha',0.2);
axis equal
hold on

x = [1,-1,-1,1] * max(h_qd_satellite.semimajor_axis);
y = [1,1,-1,-1] * max(h_qd_satellite.semimajor_axis);
z = [0,0,0,0];

try
    fill3(x,y,z,z,'FaceAlpha',0.3,'FaceColor','g','EdgeColor','g')
end

s_min = Inf;
s_max = 0;
for n = 1 : nS
    t_max = h_qd_satellite.orbit_period(1,i_sat(n));
    delta_t = t_max/180;
    tim = 0:delta_t:t_max;

    xyz_pos  = h_qd_satellite.orbit_predictor( tim , i_sat(n) );
    
    % Calculate speed 
    Dxyz_pos = h_qd_satellite.orbit_predictor( tim+0.1 , i_sat(n) );
    speed_v = sqrt(sum((Dxyz_pos - xyz_pos).^2,1))*10;
    
    scatter3(xyz_pos(1,:), xyz_pos(2,:), xyz_pos(3,:), [], speed_v,'filled');
    plot3(xyz_pos(1,:), xyz_pos(2,:), xyz_pos(3,:),'-k');
          
    s_min = min( [s_min,speed_v] );
    s_max = max( [s_max,speed_v] );
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
