function h = checksum( h_simpar )
%CHECKSUM Calculates a checksum of all parameters to identify changes in the simulation settings
%
% Calling object:
%   Single object
%
% Output:
%   h
%   The checksum (uint64 number).
%
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

h_simpar = h_simpar(1,1);

h = h_simpar.samples_per_meter + sum( h_simpar.center_frequency/1e7 );
h = uint32( mod( abs(h)*100, 2147483647 ) );
h = h + uint32( numel(h_simpar.center_frequency) + h_simpar.use_3GPP_baseline + ...
    h_simpar.use_absolute_delays + h_simpar.use_random_initial_phase  );
h = h + sum( cast( h_simpar.autocorrelation_function,'uint32' ) );

h = uint64( h );

end
