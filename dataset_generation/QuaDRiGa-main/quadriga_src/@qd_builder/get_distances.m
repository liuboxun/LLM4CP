function dist = get_distances( h_builder )
%GET_DISTANCES Calculates the distances between Rx and Tx
%
% Calling object:
%   Single object
%
% Output:
%   dist
%   A vector containing the distances between each Rx and the Tx in [m]
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
    error('QuaDRiGa:qd_builder:ObjectArray','??? "get_distances" is only defined for scalar objects.')
else
    h_builder = h_builder(1,1); % workaround for octave
end

dist = sqrt( (h_builder.rx_positions(1,:) - h_builder.tx_position(1,:) ).^2 +...
    ( h_builder.rx_positions(2,:) - h_builder.tx_position(2,:) ).^2 );

end
