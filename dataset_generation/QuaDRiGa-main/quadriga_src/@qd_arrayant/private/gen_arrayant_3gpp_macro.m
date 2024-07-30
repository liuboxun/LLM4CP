function h_qd_arrayant = gen_arrayant_3gpp_macro( phi_3dB, theta_3dB, rear_gain, electric_tilt )
%GEN_ARRAYANT_3GPP_MACRO
%
%   An antenna with a custom gain in elevation and azimuth. See. 3GPP TR 36.814 V9.0.0 (2010-03),
%   Table A.2.1.1-2, Page 59
%      * phi_3dB - Half-Power in azimuth direction (default = 70 deg)
%      * theta_3dB - Half-Power in elevation direction (default = 10 deg)
%      * rear_gain - Front-to back ratio (default = 25 dB)
%      * electric_tilt - Electrical downtilt (default = 15 deg)
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

% Set input variables
if ~exist('phi_3dB','var') || isempty( phi_3dB )
    phi_3dB = 70;
end

if ~exist('theta_3dB','var') || isempty( theta_3dB )
    theta_3dB = 10;
end

if ~exist('rear_gain','var') || isempty( rear_gain )
    rear_gain = 25;
end

if ~exist('electric_tilt','var') || isempty( electric_tilt )
    electric_tilt = 15;
end

if ~( size(phi_3dB,1) == 1 && isnumeric(phi_3dB) && isreal(phi_3dB) &&...
        all(size(phi_3dB) == [1 1]) )
    error('Azimuth HPBW (phi_3dB) has invalid value.')
end

if ~( size(theta_3dB,1) == 1 && isnumeric(theta_3dB) && isreal(theta_3dB) &&...
        all(size(theta_3dB) == [1 1]) )
    error('Elevation HPBW (theta_3dB) has invalid value.')
end

if ~( size(rear_gain,1) == 1 && isnumeric(theta_3dB) && isreal(rear_gain) && ...
        all(size(rear_gain) == [1 1]) && rear_gain>=0 )
    error('Front-to-back ratio (rear_gain) has invalid value.')
end

if ~( size(electric_tilt,1) == 1 && isnumeric(electric_tilt) && isreal(electric_tilt) &&...
        all(size(electric_tilt) == [1 1]) )
    error('Electric tilt has invalid value.')
end

% Generate omni antenna as default
h_qd_arrayant = gen_arrayant_omni;

phi = h_qd_arrayant.azimuth_grid*180/pi;
Ah  = -min( 12*(phi./phi_3dB).^2 , rear_gain );

theta = h_qd_arrayant.elevation_grid.'*180/pi;
Av  = -min( 12*((theta+electric_tilt)./theta_3dB).^2 , rear_gain-5 );

A = -min( -Ah(ones(h_qd_arrayant.no_el,1),:) - Av(:,ones(h_qd_arrayant.no_az,1)) , rear_gain );

h_qd_arrayant.Fa = sqrt( 10.^(0.1*A) );
h_qd_arrayant.normalize_gain;

end
