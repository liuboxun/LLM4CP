function pos = set_satellite_pos( h_layout, rx_latitude, sat_el, sat_az, sat_height, tx_no )
%SET_SATELLITE_POS Calculates the Tx position from a satellite orbit.
%
% Calling object:
%   Single object
%
% Description:
%   QuaDRiGas reference coordinate system is on the surface of the earth. In order to use QuaDRiGa
%   for satellite links, the satellite position must be set. Normally, this position is given in
%   azimuth and elevation relative to the users position. This function takes a satellite orbital
%   position and calculates the corresponding transmitter coordinates.
%
% Input:
%   rx_latitude
%   The receiver latitude coordinate on the earth surface in [deg]. Default is 52.5
%
%   sat_el
%   Satellite elevation seen from the receiver positions in [deg]. Default is 31.6
%
%   sat_az
%   Satellite azimuth in [deg] given in compass coordinates. Default is 180° (south)
%
%   sat_height
%   Satellite height in [km] relative to earth surface. Default is 35786 (GEO orbit)
%
%   tx_no
%   The 'tx_no' in the layout object for which the position should be set. Default is 1
%
% Output:
%   pos
%   The satellite positions in the metric QuaDRiGa coordinate system
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

if numel( h_layout ) > 1 
   error('QuaDRiGa:qd_layout:set_scenario','set_scenario not defined for object arrays.');
else
    h_layout = h_layout(1,1); % workaround for octave
end

if ~exist( 'rx_latitude' , 'var' ) || isempty( rx_latitude )
    rx_latitude = 52.5;     % Berlin :-)
elseif ~( isnumeric(rx_latitude) &&...
        isreal(rx_latitude) &&...
        all( size(rx_latitude) == 1 ) &&...
        rx_latitude >= -90 && rx_latitude <= 90 )
    error('??? "rx_latitude" has wrong format.');
end
rx_latitude = abs( rx_latitude );   % Northern and southern hemisphere are symmetric

if ~exist( 'sat_el' , 'var' ) || isempty( sat_el )
    sat_el = 31.6;
elseif ~( isnumeric(sat_el) &&...
        isreal(sat_el) &&...
        all( size(sat_el) == 1 ) &&...
        sat_el >= 0 && sat_el <= 90 )
    error('??? "sat_el" has wrong format.');
end

if ~exist( 'sat_az' , 'var' ) || isempty( sat_az )
    sat_az = 180;           % South
elseif ~( isnumeric(sat_az) &&...
        isreal(sat_az) &&...
        all( size(sat_az) == 1 ) &&...
        sat_az >= 0 && sat_az <= 360 )
    error('??? "sat_az" has wrong format.');
end

if ~exist( 'sat_height' , 'var' ) || isempty( sat_height )
    sat_height = 35786;     % GEO
elseif ~( isnumeric(sat_height) &&...
        isreal(sat_height) &&...
        all( size(sat_height) == 1 ) &&...
        sat_height >= 0 )
    error('??? "sat_height" has wrong format.');
end

if ~exist( 'tx_no' , 'var' ) || isempty( tx_no )
    tx_no = 1;
elseif ~( isnumeric(tx_no) &&...
        isreal(tx_no) &&...
        all( size(tx_no) == 1 ) &&...
        sum( tx_no == (1:h_layout.no_tx) ) == 1 )
    error('??? "tx_no" has wrong format.');
end



R_e = 6378.145; % radius of the Earth in [km]
sat_dist = (sqrt(R_e^2*sind(sat_el).^2 + sat_height.^2 + 2*sat_height*R_e) - R_e*sind(sat_el))*1e3; % distance between user and satellite in [m]
        

sat_x = sat_dist * cosd(sat_el) * cosd( -sat_az+90 );
sat_y = sat_dist * cosd(sat_el) * sind( -sat_az+90 );
sat_z = sat_dist * sind(sat_el);

pos = [ sat_x ; sat_y ; sat_z ];

h_layout.tx_position(:,tx_no) = pos;

end
