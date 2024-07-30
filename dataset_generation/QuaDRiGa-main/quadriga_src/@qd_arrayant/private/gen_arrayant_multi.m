function [ h_qd_arrayant, par ] = gen_arrayant_multi( no_elements, spacing, tilt, Fa, Fb  )
%GEN_ARRAYANT_MULTI
%
%   A multi-element antenna with adjustable electric downtilt.
%      * no_elements - Number of elements stacked in elevation direction
%      * spacing - Element spacing in [λ]
%      * tilt - Electric downtilt in [deg]
%      * Fa - Individual element pattern "Fa" for the vertical polarization (default: patch)
%      * Fb - Individual element pattern "Fb" for the horizontal polarization
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

% Set inputs
if ~exist('no_elements','var') || isempty( no_elements )
    no_elements = 8;
end

s = qd_simulation_parameters;
s.center_frequency = s.speed_of_light;
if ~exist('spacing','var') || isempty( spacing )
    spacing = 0.5 * s.wavelength;
end

if ~exist('tilt','var') || isempty( tilt )
    tilt = 0;
else
    % Convert tilt to [rad]
    tilt = tilt * pi/180;
end

if ~exist('Fa','var') || isempty( Fa )
    tmp = qd_arrayant.generate('patch');
    Fa = tmp.Fa;
end
no_el_in = size( Fa, 3);

if ~exist('Fb','var') || isempty( Fb )
    Fb = zeros( 181, 361, no_el_in);
end

% Generate omni antenna as default
h_qd_arrayant = gen_arrayant_omni;
if no_el_in > 1
    h_qd_arrayant.copy_element( 1,2:no_el_in );
end
h_qd_arrayant.Fa = Fa;
h_qd_arrayant.Fb = Fb;

% Copy the basic elements
for n = 2:no_el_in
    h_qd_arrayant.copy_element( n, (n-1)*no_elements+1 );
end

% Set the element spacing
el_pos = (0:no_elements-1) * spacing;
el_pos = el_pos - mean(el_pos);
for n = 1:no_el_in
    ind = (n-1)*no_elements + (1:no_elements);
    h_qd_arrayant.copy_element( (n-1)*no_elements+1 , ind );
    h_qd_arrayant.element_position(3,ind) = el_pos;
end

% Set the coupling
C = 2*pi*sin(tilt) * h_qd_arrayant.element_position(3,:) / s.wavelength;
C = exp( 1j*C );
C = reshape( C, no_elements , no_el_in );
coupling = C * sqrt(1/no_elements);
coupling_array = zeros( no_el_in*no_elements, no_el_in );
for n = 1:no_el_in
    ind = (n-1)*no_elements + (1:no_elements);
    coupling_array(ind,n) = coupling(:,n);
end
par = C;
h_qd_arrayant.coupling = coupling_array;

% calculate the effective pattern
h_qd_arrayant.combine_pattern( s.center_frequency );

end
