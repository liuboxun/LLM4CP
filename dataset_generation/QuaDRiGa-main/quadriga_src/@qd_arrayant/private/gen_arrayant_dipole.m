function h_qd_arrayant = gen_arrayant_dipole
%GEN_ARRAYANT_DIPOLE 
%
%   A short dipole radiating with vertical polarization
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

h_qd_arrayant = gen_arrayant_omni;

theta_grid = h_qd_arrayant.elevation_grid( ones(1,h_qd_arrayant.no_az),: )';

% Short dipole
E_theta = cos( (1 - 1e-6) * theta_grid );
E_phi = zeros(size(E_theta));

P = E_theta.^2 + E_phi.^2;      % Calculate radiation power pattern
P_max = max(max(P));            % Normalize by max value
P = P ./ P_max;

% Calculate the Gain
gain_lin = sum(sum( cos(theta_grid) )) / sum(sum( P.*cos(theta_grid) ));

% Normalize by Gain
E_theta = E_theta .* sqrt(gain_lin./P_max);
E_phi = E_phi .* sqrt(gain_lin./P_max);

h_qd_arrayant.Fa = E_theta;
h_qd_arrayant.Fb = E_phi;

end
