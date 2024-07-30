function h_qd_arrayant = gen_arrayant_3gpp_3d( M, N, center_freq, pol, tilt, spacing )
%GEN_ARRAYANT_OMNI
%
%   The antenna model for the 3GPP-3D channel model (TR 36.873, v12.5.0, pp.17).
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

% Generate omni antenna as default
h_qd_arrayant = gen_arrayant_omni;

% Antenna element vertical radiation pattern (dB)
theta = h_qd_arrayant.elevation_grid.'*180/pi;
Av  = -min( 12*(theta./65).^2 , 30 );

% Antenna element horizontal radiation pattern (dB)
phi = h_qd_arrayant.azimuth_grid*180/pi;
Ah  = -min( 12*(phi./65).^2 , 30 );

% Combining method for 3D antenna element pattern (dB)
A = -min( -Av(:,ones(h_qd_arrayant.no_az,1)) -Ah(ones(h_qd_arrayant.no_el,1),:)  , 30 );

% Set pattern
h_qd_arrayant.Fa = sqrt( 10.^(0.1*A) );

% Maximum directional gain of an antenna element is 8 dB
h_qd_arrayant.normalize_gain(1,8);

% Polarization
switch pol
    case {2,5} % H / V polarization (0, 90 deg slant)
        h_qd_arrayant.copy_element(1,2);
        h_qd_arrayant.rotate_pattern(90,'x',2,2);
        
    case {3,6} % +/- 45 deg polarization
        h_qd_arrayant.copy_element(1,2);
        h_qd_arrayant.rotate_pattern(45,'x',1,2);
        h_qd_arrayant.rotate_pattern(-45,'x',2,2);
end

% Coupling of vertically stacked elements
if pol >= 4
    h_qd_arrayant = gen_arrayant_multi( M, spacing, tilt, h_qd_arrayant.Fa, h_qd_arrayant.Fb );
    M = 1;
    h_qd_arrayant.no_elements
end

% Assign center frequency
h_qd_arrayant.center_frequency = center_freq;

if N > 1 || M > 1
    % Calculate the wavelength
    s = qd_simulation_parameters;
    s.center_frequency = center_freq;
    lambda = s.wavelength;
    
    % Copy elements
    T = h_qd_arrayant.no_elements;
    h_qd_arrayant.no_elements = T*N*M;
    
    for t=2:T
        ii = T+t : T : T*N*M;
        ij = ones(1,numel(ii))*t;
        h_qd_arrayant.Fa(:,:,ii) = h_qd_arrayant.Fa(:,:,ij);
        h_qd_arrayant.Fb(:,:,ii) = h_qd_arrayant.Fb(:,:,ij);
    end
    
    % Set vertical positions
    tmp = (0:M-1) * lambda*spacing;
    posv = tmp - mean(tmp);
    tmp = reshape( posv(ones(1,N),:).' , 1 , [] );
    h_qd_arrayant.element_position(3,:) = reshape( tmp(ones(T,1),:) ,1,[] );
    
    % Set horizontal positions
    tmp = (0:N-1) * lambda*spacing;
    posh = tmp - mean(tmp);
    tmp = reshape( posh(ones(1,M),:) , 1 , [] );
    h_qd_arrayant.element_position(2,:) = reshape( tmp(ones(T,1),:) ,1,[] );
end

end
