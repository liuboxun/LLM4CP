function h_qd_arrayant = gen_arrayant_parametric( Ain, Bin, Cin, Din )
%GEN_ARRAYANT_PARAMETRIC
%
%   An antenna with the radiation pattern set to
%        E_theta = A·sqrt(B+(1-B)·(cos(theta))^C·exp(-D·phi^2))
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

% Set inputs if not given
if ~exist('Ain','var') || isempty( Ain )
    Ain = 1.9;
end
if ~exist('Bin','var') || isempty( Bin )
    Bin = 0.1;
end
if ~exist('Cin','var') || isempty( Cin )
    Cin = 1;
end
if ~exist('Din','var') || isempty( Din )
    Din = 1.3;
end

% Generate omni antenna as default
h_qd_arrayant = gen_arrayant_omni;

phi = h_qd_arrayant.azimuth_grid;
theta = h_qd_arrayant.elevation_grid;

C = cos(theta).^Cin;
D = exp(-Din * phi.^2);

P = zeros(numel( theta ),numel(phi));
for a = 1:numel( theta )
    for b = 1:numel(phi)
        P(a, b) = C(a) * D(b);
    end
end
P = Bin + (1-Bin)*P;

h_qd_arrayant.Fa = Ain * sqrt(P);
h_qd_arrayant.Fb = zeros(h_qd_arrayant.no_el, h_qd_arrayant.no_az);

end
