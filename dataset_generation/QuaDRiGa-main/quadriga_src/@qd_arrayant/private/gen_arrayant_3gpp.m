function h_qd_arrayant = gen_arrayant_3gpp( phi_3dB, theta_3dB, SLA_v, A_m, G_dBi )
%GEN_ARRAYANT_3GPP
%
%   An antenna with a custom gain in elevation and azimuth. See. 3GPP TR 36.873 V12.7.0 (2017-12),
%   Table 7.1-1, Page 18
%      * phi_3dB - Half-Power in azimuth direction (phi_3dB), default = 65 deg
%      * theta_3dB - Half-Power in elevation direction (theta_3dB), default = 65 deg
%      * SLA_v - Side-lobe attenuation in vertical cut (SLA_v), default = 30 dB
%      * A_m - Maximum attenuation (A_m), default = 30 dB
%      * G_dBi - Antenna gain in dBi (G_dBi), default = 8 dBi
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
    phi_3dB = 65;
end

if ~exist('theta_3dB','var') || isempty( theta_3dB )
    theta_3dB = 65;
end

if ~exist('SLA_v','var') || isempty( SLA_v )
    SLA_v = 30;
end

if ~exist('A_m','var') || isempty( A_m )
    A_m = 30;
end

if ~exist('G_dBi','var') || isempty( G_dBi )
    G_dBi = 8;
end

if ~( size(phi_3dB,1) == 1 && isnumeric(phi_3dB) && isreal(phi_3dB) &&...
        all(size(phi_3dB) == [1 1]) )
    error('Azimuth HPBW (phi_3dB) has invalid value.')
end

if ~( size(theta_3dB,1) == 1 && isnumeric(theta_3dB) && isreal(theta_3dB) &&...
        all(size(theta_3dB) == [1 1]) )
    error('Elevation HPBW (theta_3dB) has invalid value.')
end

if ~( size(SLA_v,1) == 1 && isnumeric(SLA_v) && isreal(SLA_v) && ...
        all(size(SLA_v) == [1 1]) )
    error('Side-lobe attenuation in vertical cut (SLA_v) has invalid value.')
end

if ~( size(A_m,1) == 1 && isnumeric(A_m) && isreal(A_m) &&...
        all(size(A_m) == [1 1]) )
    error('Maximum attenuation (A_m) has invalid value.')
end

if ~( size(G_dBi,1) == 1 && isnumeric(G_dBi) && isreal(G_dBi) &&...
        all(size(G_dBi) == [1 1]) )
    error('Antenna gain (G_dBi) has invalid value.')
end

% Generate omni antenna as default
h_qd_arrayant = qd_arrayant.generate('omni');

phi = h_qd_arrayant.azimuth_grid*180/pi;
Ah  = -min(12*(phi./phi_3dB).^2, A_m);

theta = h_qd_arrayant.elevation_grid.'*180/pi;
Av  = -min(12*(theta./theta_3dB).^2, SLA_v);

A = -min(-Ah(ones(h_qd_arrayant.no_el, 1), :) - Av(:, ones(h_qd_arrayant.no_az, 1)), A_m);
A = A - max(A(:)) + G_dBi;
h_qd_arrayant.Fa = 10.^(A./20);

end
