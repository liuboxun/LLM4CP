function h_qd_satellite = gen_constellation_walker_delta( semimajor_axis, ...
    inclination, no_planes, no_sat_plane, phase_offset )
%GEN_CONSTELLATION_WALKER_DELTA Generates equally spaced GSO satellite
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
 
% Default  for semimajor_axis
if ~exist('semimajor_axis','var') || isempty( semimajor_axis )
    semimajor_axis = qd_satellite.R_e + 600;
elseif numel( semimajor_axis ) ~= 1
    error('QuaDRiGa:qd_satellite:gen_constellation_gso',...
        '??? "semimajor_axis" must be scalar.');
end

% Default  for inclination
if ~exist('inclination','var') || isempty( inclination )
    inclination = 60;
elseif numel( inclination ) ~= 1
    error('QuaDRiGa:qd_satellite:gen_constellation_gso',...
        '??? "inclination" must be scalar.');
end

% Default  for no_planes
if ~exist('no_planes','var') || isempty( no_planes )
    no_planes = 12;
elseif numel( no_planes ) ~= 1
    error('QuaDRiGa:qd_satellite:gen_constellation_gso',...
        '??? "no_planes" must be scalar.');
end

% Default  for no_sat_plane
if ~exist('no_sat_plane','var') || isempty( no_sat_plane )
    no_sat_plane = no_planes;
elseif numel( no_sat_plane ) ~= 1
    error('QuaDRiGa:qd_satellite:gen_constellation_gso',...
        '??? "no_sat_plane" must be scalar.');
end

% Default  for no_sat_plane
if ~exist('phase_offset','var') || isempty( phase_offset )
    phase_offset = 1 * 360/(no_sat_plane*no_planes);
elseif numel( phase_offset ) ~= 1
    error('QuaDRiGa:qd_satellite:gen_constellation_gso',...
        '??? "phase_offset" must be scalar.');
end

n_satellites = no_sat_plane * no_planes;
z_satellites = zeros( 1,n_satellites );
o_satellites = ones( 1,n_satellites );

o_sat_plane = ones(1,no_sat_plane);

% Longitude of the ascending node (Omega) in [degree]
lon_asc_node = 0 : 360 / no_planes : 359.99999999999;
lon_asc_node = reshape( lon_asc_node( o_sat_plane,: ),1,[] );

% Distribution of satellits within a plane
true_anomaly = 0 : 360 / no_sat_plane : 359.99999999999;
true_anomaly = true_anomaly( ones(1,no_planes),: );
true_anomaly = true_anomaly + (phase_offset*(0:no_planes-1)')*o_sat_plane;
true_anomaly = reshape( true_anomaly',1,[] );

% Create qd_satellite object
h_qd_satellite = gen_constellation_custom( semimajor_axis(1,o_satellites),...
    z_satellites, inclination(1,o_satellites), lon_asc_node, z_satellites, true_anomaly  );

end
