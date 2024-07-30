function h_qd_arrayant = gen_arrayant_omni
%GEN_ARRAYANT_OMNI 
%
%   An isotropic radiator with vertical polarization.
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

h_qd_arrayant = qd_arrayant( [] );
h_qd_arrayant.name              = 'omni';
h_qd_arrayant.center_frequency  = qd_simulation_parameters.speed_of_light;
h_qd_arrayant.elevation_grid    = (-90:90)*pi/180;
h_qd_arrayant.azimuth_grid      = (-180:180)*pi/180;
h_qd_arrayant.no_elements       = 1;
h_qd_arrayant.element_position  = zeros( 3,1 );
h_qd_arrayant.Fa                = ones( 181,361);
h_qd_arrayant.Fb                = zeros( 181,361);
h_qd_arrayant.coupling          = 1;

end
