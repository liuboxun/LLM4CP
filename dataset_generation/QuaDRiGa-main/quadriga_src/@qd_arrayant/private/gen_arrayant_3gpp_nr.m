function h_qd_arrayant = gen_arrayant_3gpp_nr( M, N, center_freq, pol, tilt, spacing, Mg, Ng, dgv, dgh )
%GEN_ARRAYANT_3GPP_NR
%
%   Antenna model for the 3GPP-NR channel model (TR 38.901, v15.0.0, pp.21).
%
%      * M   - Number of vertical elements (M)
%      * N   - Number of horizontal elements (N)
%      * center_freq - The center frequency in [Hz]
%      * pol - Polarization indicator
%           1. K=1, vertical polarization only
%           2. K=1, H/V polarized elements
%           3. K=1, +/-45 degree polarized elements
%           4. K=M, vertical polarization only
%           5. K=M, H/V polarized elements
%           6. K=M, +/-45 degree polarized elements
%      * tilt - The electric downtilt angle in [deg] for pol = 4,5,6
%      * spacing - Element spacing in [λ], Default: 0.5
%      * Mg  - Number of nested panels in a column (Mg)
%      * Ng  - Number of nested panels in a row (Ng)
%      * dgv - Panel spacing in vertical direction (dg,V) in [λ], Default: 0.5 M
%      * dgh - Panel spacing in horizontal direction (dg,H) in [λ], Default: 0.5 N
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
if ~exist('M','var') || isempty( M )
    M = 10;
end

if ~exist('N','var') || isempty( N )
    N = 10;
end

if ~exist('center_freq','var') || isempty( center_freq )
    center_freq = 299792458;
end

if ~exist('pol','var') || isempty( pol )
    pol = 1;
end

if ~exist('tilt','var') || isempty( tilt )
    tilt = 0;
end

if ~exist('spacing','var') || isempty( spacing )
    spacing = 0.5;
end

if ~exist('Mg','var') || isempty( Mg )
    Mg = 2;
end

if ~exist('Ng','var') || isempty( Ng )
    Ng = 2;
end

if ~exist('dgv','var') || isempty( dgv )
    dgv = 0.5 * M;
end

if ~exist('dgh','var') || isempty( dgh )
    dgh = 0.5 * N;
end

% Generate single 3GPP panel
h_qd_arrayant = gen_arrayant_3gpp_3d( M, N, center_freq, pol, tilt, spacing );

if Mg > 1 || Ng > 1
    % Calculate the wavelength
    s = qd_simulation_parameters;
    s.center_frequency = center_freq;
    lambda = s.wavelength;
    
    % Copy the array antenna
    hs = h_qd_arrayant.copy;
    
    % Nested panels in a column
    element_position = hs.element_position;
    tmp = element_position(3,:);
    for iR = 2 : Mg
        spc = (iR-1)*dgv*lambda;
        element_position(3,:) = tmp + spc;
        hs.element_position = element_position;
        h_qd_arrayant.append_array( hs );
    end
    
    if Ng > 1
        hs = h_qd_arrayant.copy;
    end
    
    % Nested panels in a row
    element_position = hs.element_position;
    tmp = element_position(2,:);
    for iC = 2 : Ng
        spc = (iC-1)*dgh*lambda;
        element_position(2,:) = tmp + spc;
        hs.element_position = element_position;
        h_qd_arrayant.append_array( hs );
    end
    
    % Center the element positions
    tmp = mean( h_qd_arrayant.element_position,2 );
    h_qd_arrayant.element_position = h_qd_arrayant.element_position - tmp(:,ones( 1,h_qd_arrayant.no_elements ));
end

end
