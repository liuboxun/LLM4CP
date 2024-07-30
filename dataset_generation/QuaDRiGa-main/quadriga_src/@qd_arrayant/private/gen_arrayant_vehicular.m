function h_qd_arrayant = gen_arrayant_vehicular( type, fr, option )
%GEN_ARRAYANT_VEHICULAR
%
%   Generates array antennas for vehicle UEs according to 3GPP TR 37.885 V15.1.0
%      * type - vehicle type
%           1: passenger vehicle w/ bumper antennas
%           2: passenger vehicle w/ rooftop antennas
%           3: bus/truck w/ rooftop antennas
%      * fr - frequency range
%           1: below 6 GHz
%           2: above 6 GHz
%      * option - model option
%           1: antennas based on macro BS antenna pattern
%           2: antenna patterns based on simulated vehicle mounted antennas
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

if ~exist('type','var') || isempty( type )
    type = 1;
end

if ~exist('fr','var') || isempty( fr )
    fr = 1;
end

if ~exist('option','var') || isempty( option )
    option = 1;
end

switch type
    case 1
        vehicle_length = 5;
        antenna_height = 0.75;
    case 2
        vehicle_length = 5;
        antenna_height = 1.6;
    case 3
        vehicle_length = 13;
        antenna_height = 3;
end

switch option
    case 1
        switch fr
            case 1
                switch type
                    case {1, 3}
                        % front/rear bumper for vehicle type 1/3 for 6 GHz (FR1) from 3GPP TR 37.885 V15.1.0 (2018-09) Table 6.1.4-8: Antenna element pattern for vehicle UE in Option 1
                        phi_3dB = 120;
                        theta_3dB = 90;
                        SLA_v = 20;
                        A_m = 20;
                        G_dBi = 3;
                    case 2
                        % front/rear rooftop for vehicle type 2 for 6 GHz (FR1) from 3GPP TR 37.885 V15.1.0 (2018-09) Table 6.1.4-8: Antenna element pattern for vehicle UE in Option 1
                        phi_3dB = 1e30;
                        theta_3dB = 90;
                        SLA_v = 20;
                        A_m = 20;
                        G_dBi = 3;
                end
                
            case 2
                % for 30 and 63 GHz GHz (FR2) from 3GPP TR 37.885 V15.1.0 (2018-09) Table 6.1.4-8: Antenna element pattern for vehicle UE in Option 1
                phi_3dB = 90;
                theta_3dB = 90;
                SLA_v = 25;
                A_m = 25;
                G_dBi = 5;
        end
        h_qd_arrayant = qd_arrayant.generate('3gpp', phi_3dB, theta_3dB, SLA_v, A_m, G_dBi);
        h_qd_arrayant.copy_element(1, 2);
        h_qd_arrayant.rotate_pattern(180, 'z', 2);
        
    case 2
        h_qd_arrayant = qd_arrayant.generate('omni');
        h_qd_arrayant.copy_element(1, 2);
        switch type
            case 1
                h_qd_arrayant.Fa(:, :, 1) = 10.^(load_vehicular_option2(fr, 1)./20);
                h_qd_arrayant.Fa(:, :, 2) = 10.^(load_vehicular_option2(fr, 4)./20);
            case {2, 3}
                h_qd_arrayant.Fa(:, :, 1) = 10.^(load_vehicular_option2(fr, 2)./20);
                h_qd_arrayant.Fa(:, :, 2) = 10.^(load_vehicular_option2(fr, 3)./20);
        end
        
end
h_qd_arrayant.element_position(1, :) = vehicle_length/2*[1, -1];
h_qd_arrayant.element_position(3, :) = antenna_height;
% TODO: decide on polarization model

end
