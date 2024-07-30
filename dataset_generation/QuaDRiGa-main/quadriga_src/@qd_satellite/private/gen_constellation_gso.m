function h_qd_satellite = gen_constellation_gso( n_satellites, phase_offset )
%GEN_CONSTELLATION_GSO Generates equally spaced GSO satellite
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
if ~exist('n_satellites','var') || isempty( n_satellites )
    n_satellites = 3;
elseif numel( n_satellites ) ~= 1
    error('QuaDRiGa:qd_satellite:gen_constellation_gso',...
        '??? "n_satellites" must be a scalar.');
end
if ~exist('phase_offset','var') || isempty( phase_offset )
    phase_offset = 0;
end

semimajor_axis = ones(1, n_satellites)*qd_satellite.R_geo;
true_anomaly = (0:360/n_satellites:359) + phase_offset;

h_qd_satellite = gen_constellation_custom( semimajor_axis, [], [], [], [], true_anomaly);
h_qd_satellite.name = 'gso constellation';
h_qd_satellite.station_keeping = 1;

end
