function angles = get_angles( h_builder )
%GET_ANGLES Calculates the departure- and arrival angles of the LOS path between Tx and Rx
%
% Calling object:
%   Single object
%
% Output:
%   angles
%   A matrix containing the four angles:
%      * Azimuth of Departure at the Tx (AoD, row 1)
%      * Azimuth of Arrival at the Rx (AoA, row 2)
%      * Elevation of Departure at the Tx (EoD, row 3)
%      * Elevation of Arrival at the Rx (EoA, row 4)
%    The number of columns corresponds to the number of rx-positions.
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

if numel( h_builder ) > 1
    error('QuaDRiGa:qd_builder:ObjectArray','??? "get_angles" is only defined for scalar objects.')
else
    h_builder = h_builder(1,1); % workaround for octave
end

% Initialize memory
angles = zeros( 4,h_builder.no_rx_positions );

% ThetaBs 
angles(1,:) = atan2( h_builder.rx_positions(2,:) - h_builder.tx_position(2,:) , ...
    h_builder.rx_positions(1,:) - h_builder.tx_position(1,:) );

% ThetaMs 
angles(2,:) = pi + angles(1,:);

% EaBs
angles(3,:) = atan( ( h_builder.rx_positions(3,:) - h_builder.tx_position(3,:) ) ./ ...
    sqrt( (h_builder.tx_position(1,:) - h_builder.rx_positions(1,:)).^2 +...
    (h_builder.tx_position(2,:) - h_builder.rx_positions(2,:)).^2 ) );

% When Rx and Tx are at the same position, the angle is NaN
% We set it to 0 instead.
angles(3, isnan( angles(3,:) ) ) = 0;

% EaMs
angles(4,:) = -angles(3,:);

% Mapping to (-180,180) degrees
angles = angles * 180/pi;
angles = mod( angles , 360);
angles = angles - 360*floor(angles/180);

end

