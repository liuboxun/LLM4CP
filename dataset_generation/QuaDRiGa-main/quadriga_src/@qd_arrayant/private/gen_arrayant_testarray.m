function h_qd_arrayant = gen_arrayant_testarray( res )
%GEN_ARRAYANT_TESTARRAY
%
%   An array antenna with near-optimal angular resolution for testing the spatial properties of the
%   channel model. This antenna can be used either as a transmit or receive antenna. The generated
%   channel coefficients can be used by 'qf.calc_angels' to obtain the departure and
%   arrival angles of clusters. The first 28 elements sample the whole sphere in vertical
%   polarization. Element 29 is ideally horizontally polarized to calculate the XPR per path.
%   Elements 30 and 31 are circularly polarized to obtain the XPR for circular and elliptic
%   polarization.
%   
%       res - Angular sampling resolution in [deg] - Default is 1 degree
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


if ~exist('res','var') || isempty(res)
    res = 1;
end
    
[ theta, phi ] = qf.pack_sphere( 27 );                      % Generate equidistant directions
N = numel( theta );                                         % Store number of directions
h_qd_arrayant = qd_arrayant('custom',20,20,0.05);           % Main beam opening and front-back ratio
h_qd_arrayant.single;

if res ~= 1
    h_qd_arrayant.set_grid( (-180:res:180)*pi/180, (-90:res:90)*pi/180 );
end

h_qd_arrayant.element_position(1) = 1.6;                    % Element distance from array phase-center
h_qd_arrayant.copy_element(1,2:N+3);                        % Set number of elements
for n = 1:N                                                 % Create sub-elements
    h_qd_arrayant.rotate_pattern( [0,theta(n),phi(n)]*180/pi,'xyz',n,1);
end
h_qd_arrayant.combine_pattern;                              % Apply far field transformation
P = sum( abs(h_qd_arrayant.Fa(:,:,1:N)).^2,3 );             % Normalize to unit power
h_qd_arrayant.Fa(:,:,1:N) = h_qd_arrayant.Fa(:,:,1:N) ./ sqrt(P(:,:,ones(1,N)));

h_qd_arrayant.Fb(:,:,N+1) = 1;                              % Add horizontal polarization
h_qd_arrayant.Fa(:,:,N+1) = 0;

h_qd_arrayant.Fb(:,:,N+2) = 1/sqrt(2);                      % Add LHCP receive polarization
h_qd_arrayant.Fa(:,:,N+2) = 1j/sqrt(2);
h_qd_arrayant.Fb(:,:,N+3) = 1/sqrt(2);                      % Add RHCP receive polarization
h_qd_arrayant.Fa(:,:,N+3) = -1j/sqrt(2);

h_qd_arrayant.double;

end
