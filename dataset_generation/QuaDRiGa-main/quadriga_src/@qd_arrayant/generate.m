function [ h_qd_arrayant, par ] = generate( array_type, Ain, Bin, Cin, Din, Ein, Fin, Gin, Hin, Iin, Jin )
%GENERATE Generates predefined array antennas
%
% Calling object:
%   None (static method)
%
% Array types:
%   omni
%   An isotropic radiator with vertical polarization.
%
%   dipole
%   A short dipole radiating with vertical polarization.
%
%   half-wave-dipole
%   A half-wave dipole radiating with vertical polarization.
%
%   patch
%   A vertically polarized patch antenna with 90° opening in azimuth and elevation.
%
%   custom
%   An antenna with a custom gain in elevation and azimuth. The values A,B,C and D for the
%   parametric antenna are returned.
%      * Ain - 3dB beam width in azimuth direction
%      * Bin - 3dB beam width in elevation direction
%      * Cin - Isotropic gain (linear scale) at the back of the antenna
%
%   parametric
%   An antenna with the radiation pattern set to
%        E_theta = A·sqrt(B+(1-B)·(cos(theta))^C·exp(-D·phi^2))
%
%   multi
%   A multi-element antenna with adjustable electric downtilt.
%      * Ain - Number of elements stacked in elevation direction
%      * Bin - Element spacing in [λ]
%      * Cin - Electric downtilt in [deg]
%      * Din - Individual element pattern "Fa" for the vertical polarization
%      * Ein - Individual element pattern "Fb" for the horizontal polarization
%
%   3gpp-macro
%   An antenna with a custom gain in elevation and azimuth. See. 3GPP TR 36.814 V9.0.0 (2010-03),
%   Table A.2.1.1-2, Page 59
%      * Ain - Half-Power in azimuth direction (default = 70 deg)
%      * Bin - Half-Power in elevation direction (default = 10 deg)
%      * Cin - Front-to back ratio (default = 25 dB)
%      * Din - Electrical downtilt (default = 15 deg)
%
%   3gpp
%   An antenna with a custom gain in elevation and azimuth. See. 3GPP TR 36.873 V12.7.0 (2017-12),
%   Table 7.1-1, Page 18
%      * Ain - Half-Power in azimuth direction (phi_3dB), default = 65 deg
%      * Bin - Half-Power in elevation direction (theta_3dB), default = 65 deg
%      * Cin - Side-lobe attenuation in vertical cut (SLA_v), default = 30 dB
%      * Din - Maximum attenuation (A_m), default = 30 dB
%      * Ein - Antenna gain in dBi (G_dBi), default = 8 dBi
%
%   3gpp-3d
%   The antenna model for the 3GPP-3D channel model (TR 36.873, v12.5.0, pp.17).
%      * Ain - Number of vertical elements (M)
%      * Bin - Number of horizontal elements (N)
%      * Cin - The center frequency in [Hz]
%      * Din - Polarization indicator
%           1. K=1, vertical polarization only
%           2. K=1, H/V polarized elements
%           3. K=1, +/-45 degree polarized elements
%           4. K=M, vertical polarization only
%           5. K=M, H/V polarized elements
%           6. K=M, +/-45 degree polarized elements
%      * Ein - The electric downtilt angle in [deg] for Din = 4,5,6
%      * Fin - Element spacing in [λ], Default: 0.5
%
%   3gpp-mmw
%   Antenna model for the 3GPP-mmWave channel model (TR 38.901, v14.1.0, pp.21). The parameters
%   "Ain" - "Fin" are identical to the above model for the "3gpp-3d" channel model. Additional
%   parameters are:
%      * Gin - Number of nested panels in a column (Mg)
%      * Hin - Number of nested panels in a row (Ng)
%      * Iin - Panel spacing in vertical direction (dg,V) in [λ], Default: 0.5 M
%      * Jin - Panel spacing in horizontal direction (dg,H) in [λ], Default: 0.5 N
%
%   parabolic
%   An ideal parabolic reflector antenna with input parameters:
%      * Ain - Radius of the antenna aperture in [meters]
%      * Bin - Center frequency in [Hz]
%      * Cin - Min. sidelobe power relative to directivity in [dB] (default: -40 dB)
%      * Din - Polarization indicator
%           1. vertical (E-theta) polarization (default)
%           2. horizontal (E-phi) polarization
%           3. LHCP
%           4. RHCP
%           5. dual-polarized two-port antenna (LHCP,RHCP)
%      * Ein - number of beams for a multibeam antenna (hexagonal layout)
%      * Fin - beam separation in [deg], default is the FWHM
%      * Gin - Satellite Tx max Gain in (dBi)
%
%   xpol
%   Two elements with ideal isotropic patterns (vertical polarization). The second element is
%   slanted by 90°.
%
%   rhcp-dipole
%   Two crossed dipoles with one port. The signal on the second element (horizontal) is shifted by
%   -90° out of phase. The two elements thus create a RHCP signal.
%
%   lhcp-dipole
%   Two crossed dipoles with one port. The signal on the second element (horizontal) is shifted by
%   90° out of phase. The two elements thus create a LHCP signal.
%
%   lhcp-rhcp-dipole
%   Two crossed dipoles. For input port 1, the signal on the second element is shifted by +90° out
%   of phase. For input port 2, the the signal on the second element is shifted by -90° out of
%   phase. Port 1 thus transmits a LHCP signal and port 2 transmits a RHCP signal.
%
%   testarray
%   An array antenna with near-optimal angular resolution for testing the spatial properties of the
%   channel model. This antenna can be used either as a transmit or receive antenna. The generated
%   channel coefficients can be used by 'qf.calc_angels' to obtain the departure and
%   arrival angles of clusters. The first 28 elements sample the whole sphere in vertical
%   polarization. Element 29 is ideally horizontally polarized to calculate the XPR per path.
%   Elements 30 and 31 are circularly polarized to obtain the XPR for circular and elliptic
%   polarization.
%      * Ain - Angular sampling resolution in [deg] - Default is 1 degree
%
%   ula2
%   Uniform linear arrays composed of 2 omni-antennas (vertical polarization) with 10 cm element
%   distance.
%
%   ula4
%   Uniform linear arrays composed of 4 omni-antennas (vertical polarization) with 10 cm element
%   distance.
%
%   ula8
%   Uniform linear arrays composed of 8 omni-antennas (vertical polarization) with 10 cm element
%   distance.
%
%   vehicular
%   Generates array antennas for vehicle UEs according to 3GPP TR 37.885 V15.1.0
%      * Ain - vehicle type
%           1: passenger vehicle w/ bumper antennas
%           2: passenger vehicle w/ rooftop antennas
%           3: bus/truck w/ rooftop antennas
%      * Bin - frequency range
%           1: below 6 GHz
%           2: above 6 GHz
%      * Cin - model option
%           1: antennas based on macro BS antenna pattern
%           2: antenna patterns based on simulated vehicle mounted antennas
%
% Input:
%   array_type
%   One of the above array types.
%
%   Ain - Jin
%   Additional parameters for the array antenna (see above).
%
% Output:
%   par
%   The parameters A, B, C, and D for the "parametric" antenna type.
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

% Default return for par
par = [];

% Initialize all input variables
var_names = {'Ain', 'Bin', 'Cin', 'Din', 'Ein', 'Fin', 'Gin', 'Hin', 'Iin', 'Jin'};
for n = 1:numel( var_names )
    if ~exist( var_names{n},'var' )
        eval([ var_names{n},' = [];' ]);
    end
end

array_type = lower( array_type );
switch array_type
    case 'omni'
        h_qd_arrayant = gen_arrayant_omni;
        
    case {'short-dipole', 'dipole'}
        h_qd_arrayant = gen_arrayant_dipole;
        
    case 'half-wave-dipole'
        h_qd_arrayant = gen_arrayant_half_wave_dipole;
        
    case 'custom'
        [ h_qd_arrayant, par ] = gen_arrayant_custom( Ain, Bin, Cin );
        
    case 'patch'
        [ h_qd_arrayant, par ] = gen_arrayant_custom( 90, 90, 0 );
        
    case 'parametric'
        h_qd_arrayant = gen_arrayant_parametric( Ain, Bin, Cin, Din );
        
    case 'multi'
        [ h_qd_arrayant, par ] = gen_arrayant_multi( Ain, Bin, Cin, Din, Ein  );
        
    case '3gpp-macro'
        h_qd_arrayant = gen_arrayant_3gpp_macro( Ain, Bin, Cin, Din );
        
    case '3gpp'
        h_qd_arrayant = gen_arrayant_3gpp( Ain, Bin, Cin, Din, Ein );
        
    case '3gpp-3d'
        h_qd_arrayant = gen_arrayant_3gpp_3d( Ain, Bin, Cin, Din, Ein, Fin );
        
    case { '3gpp-mmw', '3gpp-nr' }
        h_qd_arrayant = gen_arrayant_3gpp_nr( Ain, Bin, Cin, Din, Ein, Fin, Gin, Hin, Iin, Jin );
        
    case 'xpol'
        h_qd_arrayant = gen_arrayant_omni;
        h_qd_arrayant.copy_element(1,2);
        h_qd_arrayant.Fa(:,:,2) = 0;
        h_qd_arrayant.Fb(:,:,2) = 1;
        
    case 'rhcp-dipole'
        
        h_qd_arrayant = gen_arrayant_dipole;
        copy_element( h_qd_arrayant,1,2 );
        rotate_pattern( h_qd_arrayant,90,'x',2 );
        h_qd_arrayant.coupling = 1/sqrt(2) * [1;-1j];
        
    case 'lhcp-dipole'
        h_qd_arrayant = qd_arrayant.generate('rhcp-dipole');
        h_qd_arrayant.coupling = 1/sqrt(2) * [1;1j];
        
    case 'lhcp-rhcp-dipole'
        h_qd_arrayant = qd_arrayant.generate('rhcp-dipole');
        h_qd_arrayant.coupling = 1/sqrt(2) * [1 1;1j -1j];
        
    case 'ula2'
        h_qd_arrayant = gen_arrayant_omni;
        h_qd_arrayant.no_elements = 2;
        h_qd_arrayant.element_position(2,:) = [-0.05 0.05];
        
    case 'ula4'
        h_qd_arrayant = gen_arrayant_omni;
        h_qd_arrayant.no_elements = 4;
        h_qd_arrayant.element_position(2,:) = -0.15 :0.1: 0.15;
        
    case 'ula8'
        h_qd_arrayant = gen_arrayant_omni;
        h_qd_arrayant.no_elements = 8;
        h_qd_arrayant.element_position(2,:) = -0.35 :0.1: 0.35;
        
    case 'vehicular'
        h_qd_arrayant = gen_arrayant_vehicular( Ain, Bin, Cin );
        
    case 'parabolic'
        h_qd_arrayant = gen_arrayant_parabolic( Ain, Bin, Cin, Din, Ein, Fin, Gin );
        
    case 'testarray'
        h_qd_arrayant = gen_arrayant_testarray( Ain );
        
    otherwise
        error('QuaDRiGa:qd_arrayant:generate',['??? Array type "',array_type,'" is not supported.']);
end

h_qd_arrayant.name = array_type;

end
