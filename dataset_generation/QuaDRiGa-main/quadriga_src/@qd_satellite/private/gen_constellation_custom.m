function h_qd_satellite = gen_constellation_custom( semimajor_axis, eccentricity, ...
    inclination, lon_asc_node, arg_periapsis, true_anomaly  )
%GEN_CONSTELLATION_CUSTOM Generates a custom satellite constellation
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

% Set defaults
if ~exist('semimajor_axis','var') || isempty( semimajor_axis )
    semimajor_axis = qd_satellite.R_geo;
end
semimajor_axis = semimajor_axis(:).';
z = zeros( size( semimajor_axis ));
if ~exist('eccentricity','var') || isempty( eccentricity )
    eccentricity = z;
end
if ~exist('inclination','var') || isempty( inclination )
    inclination = z;
end
if ~exist('lon_asc_node','var') || isempty( lon_asc_node )
    lon_asc_node = z;
end
if ~exist('arg_periapsis','var') || isempty( arg_periapsis )
    arg_periapsis = z;
end
if ~exist('true_anomaly','var') || isempty( true_anomaly )
    true_anomaly = z; 
end

h_qd_satellite = qd_satellite([]);
h_qd_satellite.name = 'custom constellation';
h_qd_satellite.semimajor_axis   = semimajor_axis;
h_qd_satellite.eccentricity     = eccentricity(:).';
h_qd_satellite.inclination      = inclination(:).';
h_qd_satellite.lon_asc_node     = lon_asc_node(:).';
h_qd_satellite.arg_periapsis    = arg_periapsis(:).';
h_qd_satellite.true_anomaly     = true_anomaly(:).';

for n = 1 : h_qd_satellite.n_satellites
    h_qd_satellite.sat_name{1,n} = ['sat',num2str(n,'%04d')];
end

end
